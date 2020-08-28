import pdb
import random
import math
import torch
import numpy as np
import torchvision.transforms as transforms
import ltr.data.processing_utils as prutils

from pytracking import TensorDict


class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), train_transform=None, test_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if train_transform or
                                test_transform is None.
            train_transform - The set of transformations to be applied on the train images. If None, the 'transform'
                                argument is used instead.
            test_transform  - The set of transformations to be applied on the test images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the train and test images.  For
                                example, it can be used to convert both test and train images to grayscale.
        """
        self.transform = {'train': transform if train_transform is None else train_transform,
                          'test':  transform if test_transform is None else test_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class ATOMProcessing(BaseProcessing):
    """ The processing class used for training ATOM. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.

    """
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, proposal_params,
                 mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            proposal_params - Arguments for the proposal generation process. See _generate_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.proposal_params = proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        ''' original implementation
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * self.center_jitter_factor[mode]).item()
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        '''

        ''' my implementation '''
        scale_jitter_factor = self.scale_jitter_factor[mode]
        center_jitter_factor = self.center_jitter_factor[mode]

        scale_jitter_coefficient = torch.exp(torch.randn(2) * scale_jitter_factor)
        center_jitter_coefficient = (scale_jitter_coefficient.prod().sqrt() * torch.Tensor([(box[3]/box[2]).sqrt(), (box[2]/box[3]).sqrt()]) * center_jitter_factor - 1).clamp(0)

        scale_jitter = box[2:4] * scale_jitter_coefficient
        center_jitter = box[0:2] + 0.5 * box[2:4] + (torch.rand(2)-0.5) * box[2:4] * center_jitter_coefficient

        return torch.cat((center_jitter - 0.5 * scale_jitter, scale_jitter), dim=0)

    def _generate_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.proposal_params['boxes_per_frame']
        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.proposal_params['min_iou'],
                                                             sigma_factor=self.proposal_params['sigma_factor']
                                                             )

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals'-
                'proposal_iou'  -
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'] = [self.transform[s](x) for x in crops]
            data[s + '_anno'] = boxes

            ## do flip (my own)
            FLIP = random.random() < 0.5
            if FLIP:
              data[s + '_images'][0] = data[s + '_images'][0].flip(2)
              WIDTH, HEIGHT = data[s + '_images'][0].shape[1], data[s + '_images'][0].shape[2]
              data[s + '_anno'][0][0] = WIDTH - data[s + '_anno'][0][0] - data[s + '_anno'][0][2]
              data[s + '_anno'][0][1] = HEIGHT - data[s + '_anno'][0][1] - data[s + '_anno'][0][3]

        # Generate proposals
        frame2_proposals, gt_iou = zip(*[self._generate_proposals(a) for a in data['test_anno']])

        data['test_proposals'] = list(frame2_proposals)
        data['proposal_iou'] = list(gt_iou)

        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class SBDTProcessing(BaseProcessing):
    """ The processing class used for training SBDT which contains a locator and a scaler. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region ) centered at the jittered 
    target center, and of area search_area_factor^2 times the area of the jittered box is cropped from the image. The reason for jittering the 
    target box is to avoid learning the bias that the target is always at the center of the search region. The search region is then resized to 
    a fixed size given by the argument output_sz. A set of proposals are then generated for the test images by jittering the ground truth box.
    """
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, 
                 scaler_proposal_params, locator_proposal_params, mode='pair', *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scaler_proposal_params - Arguments for the scale proposal generation process. See _generate_scale_proposals for details.
            locator_proposal_params - For the locator proposal generation. See _generate_locator_proposals for details.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.scaler_proposal_params = scaler_proposal_params
        self.locator_proposal_params = locator_proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'train' or 'test' indicating train or test data

        returns:
            torch.Tensor - jittered box
        """

        '''
        a, b = scale_jitter_coefficient[0], scale_jitter_coefficient[1]
        m, n = center_jitter_coefficient[0], center_jitter_coefficient[1]
        f = center_jitter_factor
        sqrt(awbhf^2) / 2 = mw + w/2  ------> m = (f*sqrt(ab)*sqrt(h/w)-1)*0.5
        sqrt(awbhf^2) / 2 = nh + h/2  ------> n = (f*sqrt(ab)*sqrt(w/h)-1)*0.5
        '''
        scale_jitter_factor = self.scale_jitter_factor[mode]
        center_jitter_factor = self.center_jitter_factor[mode]

        scale_jitter_coefficient = torch.exp(torch.randn(2) * scale_jitter_factor).clamp(0.25, 4)
        center_jitter_coefficient = (scale_jitter_coefficient.prod().sqrt() * torch.Tensor([(box[3]/box[2]).sqrt(), (box[2]/box[3]).sqrt()]) * center_jitter_factor - 1).clamp(0)

        scale_jitter = box[2:4] * scale_jitter_coefficient
        center_jitter = box[0:2] + 0.5 * box[2:4] + (torch.rand(2)-0.5) * box[2:4] * center_jitter_coefficient

        return torch.cat((center_jitter - 0.5 * scale_jitter, scale_jitter), dim=0)

    def _get_jittered_box2(self, box):
        scale_jitter_factor = self.locator_proposal_params['scale_jitter_factor']

        jittered_coefficient = torch.exp(torch.randn(2) * scale_jitter_factor).clamp(0.5, 2)
        jittered_size = box[2:4] * jittered_coefficient

        return torch.cat((box[0:2]+box[2:4]*0.5-jittered_size*0.5, jittered_size), dim=0)

    def _generate_scale_proposals(self, box):
        """ Generates proposals by adding noise to the input box
        args:
            box - input box

        returns:
            torch.Tensor - Array of shape (num_proposals, 4) containing proposals
            torch.Tensor - Array of shape (num_proposals,) containing IoU overlap of each proposal with the input box. The
                        IoU is mapped to [-1, 1]
        """
        # Generate proposals
        num_proposals = self.scaler_proposal_params['boxes_per_frame']
        proposals = torch.zeros((num_proposals, 4))
        gt_iou = torch.zeros(num_proposals)

        for i in range(num_proposals):
            proposals[i, :], gt_iou[i] = prutils.perturb_box(box, min_iou=self.scaler_proposal_params['min_iou'],
                                                             sigma_factor=self.scaler_proposal_params['sigma_factor'])

        # Map to [-1, 1]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def _generate_locator_proposals(self, anno):
        WIDTH, HEIGHT = self.output_sz, self.output_sz
        search_area_factor = self.search_area_factor
        sigma_factor = self.locator_proposal_params['label_sigma']
        num_proposals = self.locator_proposal_params['boxes_per_frame']

        width, height = anno[2], anno[3]
        num_proposals_per = int(math.sqrt(num_proposals))
        x_step = ((WIDTH - WIDTH/search_area_factor) / (num_proposals_per-1))
        y_step = ((HEIGHT - HEIGHT/search_area_factor) / (num_proposals_per-1))

        proposals = torch.zeros(num_proposals, 4, dtype=torch.float32)
        proposals[:, 0] = torch.arange(num_proposals_per).repeat(num_proposals_per).type(torch.float32) * x_step + WIDTH/(search_area_factor*2)
        proposals[:, 1] = torch.arange(num_proposals_per).repeat(num_proposals_per,1).t().reshape(-1).type(torch.float32) * y_step + HEIGHT/(search_area_factor*2)
        proposals[:, 2], proposals[:, 3] = width, height

        x_delta = proposals[:, 0] - (anno[0] + anno[2]*0.5)
        y_delta = proposals[:, 1] - (anno[1] + anno[3]*0.5)
        sigma = sigma_factor * math.sqrt(width*height)
        labels = torch.exp(-0.5*(x_delta.pow(2)+y_delta.pow(2)) / sigma**2)

        proposals[:, 0] -= proposals[:, 2] * 0.5
        proposals[:, 1] -= proposals[:, 3] * 0.5

        return proposals, labels
        
    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -

        returns:
            TensorDict - output data block with following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
                'test_proposals'-
                'proposal_iou'  -
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        for s in ['train', 'test']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'],
                                                self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'] = [self.transform[s](x) for x in crops]
            data[s + '_anno'] = boxes

            ## do flip (my own)
            FLIP = random.random() < 0.5
            if FLIP:
              data[s + '_images'][0] = data[s + '_images'][0].flip(2)
              WIDTH, HEIGHT = data[s + '_images'][0].shape[1], data[s + '_images'][0].shape[2]
              data[s + '_anno'][0][0] = WIDTH - data[s + '_anno'][0][0] - data[s + '_anno'][0][2]

        # Generate proposals for scaler
        scale_proposals, gt_iou = zip(*[self._generate_scale_proposals(a) for a in data['test_anno']])

        data['test_scale_proposals'] = list(scale_proposals)
        data['proposal_iou'] = list(gt_iou)

        # Generate train and test proposals for locator
        data['test_anno_jittered'] = [self._get_jittered_box2(a) for a in data['test_anno']]
        train_locator_proposals, train_locator_labels = zip(*[self._generate_locator_proposals(a) for a in data['train_anno']])
        test_locator_proposals, test_locator_labels = zip(*[self._generate_locator_proposals(a) for a in data['test_anno_jittered']])

        data['train_locator_proposals'] = list(train_locator_proposals)
        data['train_locator_labels'] = list(train_locator_labels)
        data['test_locator_proposals'] = list(test_locator_proposals)
        data['test_locator_labels'] = list(test_locator_labels)

        data['train_locator_proposals'][0] = torch.cat((data['train_locator_proposals'][0], data['train_anno'][0].reshape(1,-1)), dim=0)
        data['train_locator_labels'][0] = torch.cat((data['train_locator_labels'][0], torch.Tensor([1.0])), dim=0)
        data['test_locator_proposals'][0] = torch.cat((data['test_locator_proposals'][0], data['test_anno'][0].reshape(1,-1)), dim=0)
        data['test_locator_labels'][0] = torch.cat((data['test_locator_labels'][0], torch.Tensor([1.0])), dim=0)
        
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


class SBDTv2Processing(BaseProcessing):
    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor, 
                 scaler_proposal_params, locator_proposal_params, mode='pair', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.scaler_proposal_params = scaler_proposal_params
        self.locator_proposal_params = locator_proposal_params
        self.mode = mode

    def _get_jittered_box(self, box, mode):
        '''
        a, b = scale_jitter_coefficient[0], scale_jitter_coefficient[1]
        m, n = center_jitter_coefficient[0], center_jitter_coefficient[1]
        f = center_jitter_factor
        sqrt(awbhf^2) / 2 = mw + w/2  ------> m = (f*sqrt(ab)*sqrt(h/w)-1)*0.5
        sqrt(awbhf^2) / 2 = nh + h/2  ------> n = (f*sqrt(ab)*sqrt(w/h)-1)*0.5
        '''
        scale_jitter_factor = self.scale_jitter_factor[mode]
        center_jitter_factor = self.center_jitter_factor[mode]

        scale_jitter_coefficient = torch.exp(torch.randn(2) * scale_jitter_factor).clamp(0.25, 4)
        center_jitter_coefficient = (scale_jitter_coefficient.prod().sqrt() * torch.Tensor([(box[3]/box[2]).sqrt(), (box[2]/box[3]).sqrt()]) * center_jitter_factor - 1).clamp(0)

        scale_jitter = box[2:4] * scale_jitter_coefficient
        center_jitter = box[0:2] + 0.5 * box[2:4] + (torch.rand(2)-0.5) * box[2:4] * center_jitter_coefficient

        return torch.cat((center_jitter - 0.5 * scale_jitter, scale_jitter), dim=0)

    def _get_jittered_box2(self, box):
        scale_jitter_factor = self.locator_proposal_params['scale_jitter_factor']

        jittered_coefficient = torch.exp(torch.randn(2) * scale_jitter_factor).clamp(0.5, 2)
        jittered_size = box[2:4] * jittered_coefficient

        return torch.cat((box[0:2]+box[2:4]*0.5-jittered_size*0.5, jittered_size), dim=0)

    def _generate_scaler_proposals(self, box):
        min_iou = self.scaler_proposal_params['min_iou']
        sigma_factor = self.scaler_proposal_params['sigma_factor']
        num_proposals = self.scaler_proposal_params['boxes_per_frame']

        def iou(reference, proposals):
            tl = torch.max(reference[:,:2], proposals[:,:2])
            br = torch.min(reference[:,:2] + reference[:,2:], proposals[:,:2] + proposals[:,2:])
            sz = (br - tl).clamp(0)

            # Area
            intersection = sz.prod(dim=1)
            union = reference[:,2:].prod(dim=1) + proposals[:,2:].prod(dim=1) - intersection
            return intersection / union

        cx, cy = box[0] + 0.5*box[2], box[1] + 0.5*box[3]
        proposals, gt_iou = box.view(1, 4), torch.Tensor([1.0])

        while proposals.shape[0] < num_proposals:
            # choice sigma randomly
            perturb_factor = torch.sqrt(box[2]*box[3]) * random.choice(sigma_factor)
            # perturb box
            c_x_per = np.random.normal(cx, perturb_factor, int(num_proposals*0.1))
            c_y_per = np.random.normal(cy, perturb_factor, int(num_proposals*0.1))
            w_per = np.random.normal(box[2], perturb_factor, int(num_proposals*0.1))
            h_per = np.random.normal(box[3], perturb_factor, int(num_proposals*0.1))
            # calculate IoU
            box_per = torch.Tensor([c_x_per - 0.5*w_per, c_y_per - 0.5*h_per, w_per, h_per]).t()
            box_iou = iou(box.view(1, 4), box_per)
            # select
            ID = box_iou>min_iou
            proposals = torch.cat((proposals, box_per[ID,:]), dim=0)
            gt_iou = torch.cat((gt_iou, box_iou[ID]))

        proposals = proposals[:num_proposals, :]
        gt_iou = gt_iou[:num_proposals]
        gt_iou = gt_iou * 2 - 1
        return proposals, gt_iou

    def _generate_locator_proposals(self, anno):
        WIDTH, HEIGHT = self.output_sz, self.output_sz
        search_area_factor = self.search_area_factor
        proposals_num = self.locator_proposal_params['boxes_per_frame']
        sigma_factor = self.locator_proposal_params['label_sigma']

        x_tl, y_tl, width, height = anno[0], anno[1], anno[2], anno[3]
        xc, yc = x_tl + 0.5 * width, y_tl + 0.5 * height

        ## Uniform sampling in whole image
        proposals_sqrt = int(math.sqrt(proposals_num))
        x_step = ((WIDTH - WIDTH/search_area_factor) / (proposals_sqrt-1))
        y_step = ((HEIGHT - HEIGHT/search_area_factor) / (proposals_sqrt-1))

        proposals = torch.zeros(proposals_num, 4, dtype=torch.float32)
        proposals[:, 0] = torch.arange(proposals_sqrt).repeat(proposals_sqrt).type(torch.float32) * x_step + WIDTH/(search_area_factor*2)
        proposals[:, 1] = torch.arange(proposals_sqrt).repeat(proposals_sqrt,1).t().reshape(-1).type(torch.float32) * y_step + HEIGHT/(search_area_factor*2)
        proposals[:, 2], proposals[:, 3] = width, height

        ## Gaussian labels
        x_delta = proposals[:, 0] - xc
        y_delta = proposals[:, 1] - yc
        sigma = sigma_factor * math.sqrt(width*height)
        labels = torch.exp(-0.5*(x_delta.pow(2)+y_delta.pow(2)) / sigma**2)

        proposals[:, 0] -= proposals[:, 2] * 0.5
        proposals[:, 1] -= proposals[:, 3] * 0.5
        return proposals, labels
        
    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'train_images'  -
                'test_images'   -
                'train_anno'    -
                'test_anno'     -
        """
        # Apply joint transforms
        if self.transform['joint'] is not None:
            num_train_images = len(data['train_images'])
            all_images = data['train_images'] + data['test_images']
            all_images_trans = self.transform['joint'](*all_images)

            data['train_images'] = all_images_trans[:num_train_images]
            data['test_images'] = all_images_trans[num_train_images:]

        for s in ['train', 'test']:
            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # Crop image region centered at jittered_anno box
            crops, boxes = prutils.jittered_center_crop(data[s + '_images'], jittered_anno, data[s + '_anno'], self.search_area_factor, self.output_sz)

            # Apply transforms
            data[s + '_images'] = [self.transform[s](x) for x in crops]
            data[s + '_anno'] = boxes

            ## random flip
            FLIP = random.random() < 0.5
            if FLIP:
              data[s + '_images'][0] = data[s + '_images'][0].flip(2)
              WIDTH = data[s + '_images'][0].shape[1]
              data[s + '_anno'][0][0] = WIDTH - data[s + '_anno'][0][0] - data[s + '_anno'][0][2]

        # torch.set_printoptions(threshold=20000)
        # Generate train and test proposals for scaler
        train_scaler_proposals, train_scaler_labels = zip(*[self._generate_scaler_proposals(a) for a in data['train_anno']])
        test_scaler_proposals, test_scaler_labels = zip(*[self._generate_scaler_proposals(a) for a in data['test_anno']])

        data['train_scaler_proposals'], data['train_scaler_labels'] = list(train_scaler_proposals), list(train_scaler_labels)
        data['test_scaler_proposals'], data['test_scaler_labels'] = list(test_scaler_proposals), list(test_scaler_labels)

        # Generate train and test proposals for locator
        data['test_anno_jittered'] = [self._get_jittered_box2(a) for a in data['test_anno']]
        train_locator_proposals, train_locator_labels = zip(*[self._generate_locator_proposals(a) for a in data['train_anno']])
        test_locator_proposals, test_locator_labels = zip(*[self._generate_locator_proposals(a) for a in data['test_anno_jittered']])

        data['train_locator_proposals'], data['train_locator_labels'] = list(train_locator_proposals), list(train_locator_labels)
        data['test_locator_proposals'], data['test_locator_labels'] = list(test_locator_proposals), list(test_locator_labels)

        data['train_locator_proposals'][0] = torch.cat((data['train_locator_proposals'][0], data['train_anno'][0].reshape(1,-1)), dim=0)
        data['train_locator_labels'][0] = torch.cat((data['train_locator_labels'][0], torch.Tensor([1.0])), dim=0)
        data['test_locator_proposals'][0] = torch.cat((data['test_locator_proposals'][0], data['test_anno'][0].reshape(1,-1)), dim=0)
        data['test_locator_labels'][0] = torch.cat((data['test_locator_labels'][0], torch.Tensor([1.0])), dim=0)
        
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(prutils.stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data


## debug ##
'''
import os,cv2,numpy
def tensor_to_image(image_tensor):
    image_tensor[0,:,:] *= 0.229
    image_tensor[1,:,:] *= 0.224
    image_tensor[2,:,:] *= 0.225
    image_tensor[0,:,:] += 0.485
    image_tensor[1,:,:] += 0.456
    image_tensor[2,:,:] += 0.406
    image_tensor *= 255
    image_tensor = image_tensor.permute(1,2,0)
    image_tensor = image_tensor.data.numpy().astype(numpy.uint8)
    return image_tensor
output_dir = '/home/zhenglinyu2/SBDT/debug/train_image/'
count = len(os.listdir(output_dir))
count = int(count / 2)
train_image = tensor_to_image(data['train_images'][0])
test_image = tensor_to_image(data['test_images'][0])
train_anno = data['train_anno'][0].data.numpy().astype(int)
test_anno = data['test_anno'][0].data.numpy().astype(int)
test_anno_jittered = data['test_anno_jittered'][0].data.numpy().astype(int)
cv2.imwrite(os.path.join(output_dir,'{}-train.jpg'.format(count+1)),train_image)
cv2.imwrite(os.path.join(output_dir,'{}-test.jpg'.format(count+1)),test_image)
train_image = cv2.imread(os.path.join(output_dir,'{}-train.jpg'.format(count+1)))
test_image = cv2.imread(os.path.join(output_dir,'{}-test.jpg'.format(count+1)))
######## -------------------------------------------------------- ########
train_locator_proposals = data['train_locator_proposals'][0]
test_locator_proposals = data['test_locator_proposals'][0]
train_locator_labels = data['train_locator_labels'][0]
test_locator_labels = data['test_locator_labels'][0]
for i in range(train_locator_proposals.shape[0]):
    cv2.circle(train_image, (train_locator_proposals[i][0]+train_locator_proposals[i][2]*0.5, \
train_locator_proposals[i][1]+train_locator_proposals[i][3]*0.5), 1, (int(255*train_locator_labels[i]),0,0), 2)
for i in range(test_locator_proposals.shape[0]):
    cv2.circle(test_image, (test_locator_proposals[i][0]+test_locator_proposals[i][2]*0.5, \
test_locator_proposals[i][1]+test_locator_proposals[i][3]*0.5), 1, (int(255*test_locator_labels[i]),0,0), 2)
cv2.rectangle(train_image, (train_anno[0],train_anno[1]),(train_anno[0]+train_anno[2],train_anno[1]+train_anno[3]), (0,0,255), 3)
cv2.rectangle(test_image, (test_anno[0],test_anno[1]),(test_anno[0]+test_anno[2],test_anno[1]+test_anno[3]), (0,0,255), 3)
cv2.rectangle(test_image, (test_anno_jittered[0],test_anno_jittered[1]),(test_anno_jittered[0]+test_anno_jittered[2],test_anno_jittered[1]+test_anno_jittered[3]), (0,255,0), 3)
######## -------------------------------------------------------- ########
train_scaler_proposals = data['train_scaler_proposals'][0]
test_scaler_proposals = data['test_scaler_proposals'][0]
for i in range(train_scaler_proposals.shape[0]):
    cv2.rectangle(train_image, (train_scaler_proposals[i][0],train_scaler_proposals[i][1]),(train_scaler_proposals[i][0]+train_scaler_proposals[i][2],\
train_scaler_proposals[i][1]+train_scaler_proposals[i][3]), (0,0,255), 1)
for i in range(test_scaler_proposals.shape[0]):
    cv2.rectangle(test_image, (test_scaler_proposals[i][0],test_scaler_proposals[i][1]),(test_scaler_proposals[i][0]+test_scaler_proposals[i][2],\
test_scaler_proposals[i][1]+test_scaler_proposals[i][3]), (0,0,255), 1)
cv2.imwrite(os.path.join(output_dir,'{}-train.jpg'.format(count+1)),train_image)
cv2.imwrite(os.path.join(output_dir,'{}-test.jpg'.format(count+1)),test_image)
'''

## old one ##
'''
def _generate_scaler_proposals(self, box):
    scale_level = self.scaler_proposal_params['scale_level']
    scale_factor = self.scaler_proposal_params['scale_factor']
    location_level = self.scaler_proposal_params['location_level']
    location_ratio = self.scaler_proposal_params['location_ratio']

    cx, cy = box[0] + box[2] * 0.5, box[1] + box[3] * 0.5

    proposals_num = (scale_level*location_level)**2
    proposals = torch.zeros(proposals_num, 4, dtype=torch.float32)
    proposals[:, 0], proposals[:, 1] = cx, cy
    proposals[:, 2], proposals[:, 3] = box[2], box[3]


    #-1  0  1 -1  0  1 -1  0  1  -1  0  1 -1  0  1 -1  0  1  -1  0  1 -1  0  1 -1  0  1 
    #-1 -1 -1  0  0  0  1  1  1  -1 -1 -1  0  0  0  1  1  1  -1 -1 -1  0  0  0  1  1  1
    #-1 -1 -1 -1 -1 -1 -1 -1 -1   0  0  0  0  0  0  0  0  0   1  1  1  1  1  1  1  1  1
    #-1 -1 -1 -1 -1 -1 -1 -1 -1  -1 -1 -1 -1 -1 -1 -1 -1 -1  -1 -1 -1 -1 -1 -1 -1 -1 -1
        
    proposals_delta = torch.zeros((scale_level*location_level)**2, 4, dtype=torch.float32)
    proposals_delta[:, 0] = (torch.arange(location_level).type(torch.float32).repeat(int(proposals_num/location_level)) - location_level//2) * location_ratio * box[2]
    proposals_delta[:, 1] = (torch.arange(location_level).type(torch.float32).repeat(location_level, 1).t().reshape(-1).repeat(int(proposals_num/location_level**2)) - location_level//2) * location_ratio * box[3]
    proposals_delta[:, 2] = torch.Tensor([scale_factor]).pow(torch.arange(scale_level).type(torch.float32).repeat(int(proposals_num/scale_level**2), 1).t().reshape(-1).repeat(scale_level) - scale_level//2)
    proposals_delta[:, 3] = torch.Tensor([scale_factor]).pow(torch.arange(scale_level).type(torch.float32).repeat(int(proposals_num/scale_level), 1).t().reshape(-1) - scale_level//2)

    proposals[:, 0] += proposals_delta[:, 0]
    proposals[:, 1] += proposals_delta[:, 1]
    proposals[:, 2] *= proposals_delta[:, 2]
    proposals[:, 3] *= proposals_delta[:, 3]

    labels = torch.zeros(proposals_num, 4, dtype=torch.float32)
    labels[:, 0] = (cx - proposals[:, 0]) / proposals[:, 2]
    labels[:, 1] = (cy - proposals[:, 1]) / proposals[:, 3]
    labels[:, 2] = torch.log(box[2] / proposals[:, 2])
    labels[:, 3] = torch.log(box[3] / proposals[:, 3])

    proposals[:, 0] -= proposals[:, 2] * 0.5
    proposals[:, 1] -= proposals[:, 3] * 0.5

    return proposals, labels
'''