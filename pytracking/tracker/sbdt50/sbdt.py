import os
import cv2
import sys
import pdb
import math
import time
import random
import shutil
import numpy as np

import torch
import torch.nn
import torch.nn.functional as F

from pytracking import TensorList
from pytracking.features import augmentation
from pytracking.tracker.base import BaseTracker
from pytracking.utils.plotting import show_tensor
from pytracking.features.preprocessing import sample_patch
from pytracking.features.preprocessing import numpy_to_torch

class SBDT(BaseTracker):

    # ------ MAIN INITIALIZE ------#
    def initialize(self, image, state, *args, **kwargs):
        self.frame_num = 1

        if self.params.output_image:
            # Make
            if not os.path.exists(self.params.output_image_path):
                os.mkdir(self.params.output_image_path)

            NUM = len(os.listdir(self.params.output_image_path))
            self.params.output_image_path = os.path.join(self.params.output_image_path, "%d"%(NUM+1))
            os.mkdir(self.params.output_image_path)

            # For debugging and display only
            image_show = image.copy() 

        # For debugging
        torch.set_printoptions(threshold=20000) 
        
        # Fix random seed
        np.random.seed(1024)
        torch.manual_seed(1024)
        torch.cuda.manual_seed_all(1024)

        # HEIGHT and WIDTH
        self.IMG_HEIGHT, self.IMG_WIDTH = image.shape[0], image.shape[1]

        # Initialize tracking model
        self.params.model.initialize()

        # Get target position and target size (y, x, h, w) (state = [xt, yt, w, h])
        self.target_pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])
        self.initial_target_sz = self.target_sz.clone()

        # Set sample size and target area of search region (N)
        self.img_sample_sz = torch.Tensor([math.sqrt(self.params.img_sample_area)]) * torch.ones(2)
        self.target_sample_area = self.params.img_sample_area / self.params.search_padding**2

        # Get sampling area, sampling ratio and target size
        self.search_area = torch.prod(self.target_sz * self.params.search_padding)
        self.sample_scale = torch.sqrt(self.search_area / self.params.img_sample_area)
        self.target_sample_sz = self.target_sz / self.sample_scale

        # Initialize centers of proposals for locator (N)
        self.locator_proposals_xc, self.locator_proposals_yc, self.locator_labels = self.init_locator_proposals_center_function()
        self.locator_proposals = torch.zeros(1, self.locator_labels.shape[0], 4, device=self.params.device)
        assert(self.locator_labels.max().item()==1.0)

        # Creat output score window (N)
        self.output_window = None
        if getattr(self.params, 'window_output', True):
            self.output_window = self.init_output_window_function()

        # Extract transform samples
        im_tensor = numpy_to_torch(image)
        train_samples = self.generate_init_samples(im_tensor, self.target_pos, self.sample_scale)
        train_samples = train_samples.cuda()

        # Setup scale bounds
        self.image_sz = torch.Tensor([self.IMG_HEIGHT, self.IMG_WIDTH])
        self.min_scale_factor = torch.max(10 / self.target_sz)
        self.max_scale_factor = torch.min(self.image_sz / self.target_sz)

        # Generate locator proposals
        batch_size = train_samples.shape[0]
        locator_proposals = self.get_locator_proposals(self.target_sample_sz)
        locator_proposals = locator_proposals.repeat(batch_size,1,1)

        # Extract backbone features
        backbone_features = self.params.model.extract_backbone_features(train_samples)

        # Extract target iounet features 
        self.target_iou_feat = self.init_iou_net(self.target_pos, self.target_sz, self.sample_scale, backbone_features)

        # Extract locator features and train locator model
        train_locator_features = self.params.model.extract_locator_features(backbone_features, locator_proposals)
        self.locator_XTX = torch.matmul(train_locator_features.permute(0,2,1), train_locator_features).mean(dim=0)
        self.locator_XTY = torch.matmul(train_locator_features.permute(0,2,1), self.locator_labels).mean(dim=0)
        self.locator_regularization = self.params.regularization * torch.eye(self.locator_XTX.shape[1], device=self.params.device)
        self.locator_model = self.train_locator_model(self.locator_XTX+self.locator_regularization, self.locator_XTY)

        # Save initial locator feature model
        self.locator_XTX_initial = self.locator_XTX.clone()
        self.locator_XTY_initial = self.locator_XTY.clone()

        # Initialize the detect region of hard negative samples
        self.hard_negative_region_mask = self.init_hard_negative_region_function()

        # Initialize the weight of first frame
        self.current_initial_frame_weight = 1.0

        # Output result image
        if self.params.output_image:
            self.output_result_image(image_show, state)

    # ------ MAIN TRACK ------#
    def track(self, image):
        self.frame_num += 1

        # For debugging and display only
        if self.params.output_image:
            image_show = image.copy()

        # Initialization
        hard_flag = False
        
        # Conver to tensor and GPU
        image_cuda = self.numpy_to_tensor_gpu(image)

        # ------- LOCALIZATION ------- #
        sample_pos = self.target_pos.clone()
        sample_scale = self.sample_scale.clone()
        target_sample_sz = self.target_sample_sz.clone()

        # Sample and extract backbone features
        test_sample = sample_patch(image_cuda, sample_pos, sample_scale*self.img_sample_sz, self.img_sample_sz)
        test_backbone_features = self.params.model.extract_backbone_features(test_sample)

        # Extract locator features and calcualte the localization score
        test_locator_proposals = self.get_locator_proposals(target_sample_sz)
        test_locator_features = self.params.model.extract_locator_features(test_backbone_features, test_locator_proposals).squeeze()
        test_locator_score = torch.mm(test_locator_features, self.locator_model)

        # Window output and find argmax
        if getattr(self.params, 'window_output', False):
            test_locator_score = test_locator_score * self.output_window
        max_score, max_id = torch.max(test_locator_score, dim=0)
        max_score, max_id = max_score.item(), max_id.item()

        # When target is found
        if max_score > self.params.target_not_found:
            # Update target position
            self.target_pos[1] += (self.locator_proposals_xc[max_id].item() - self.img_sample_sz[1]*0.5) * sample_scale  # x
            self.target_pos[0] += (self.locator_proposals_yc[max_id].item() - self.img_sample_sz[0]*0.5) * sample_scale  # y

            # ------- REFINEMENT ------- # 
            # Extract iou backbone features and refine target box
            test_iou_backbone_features = self.params.model.extract_iou_features(test_backbone_features) 
            new_target_box = self.refine_target_box(self.target_pos, self.target_sz, sample_pos, sample_scale, test_iou_backbone_features)

            # Update target box
            if new_target_box is not None:
                self.target_pos = sample_pos + (new_target_box[:2] + new_target_box[2:]/2 - (self.img_sample_sz - 1) / 2).flip((0,)) * sample_scale
                self.target_sz = self.params.scale_damp * self.target_sz + (1 - self.params.scale_damp) * new_target_box[2:].flip((0,)) * sample_scale

            self.target_sz = torch.min(self.target_sz, self.initial_target_sz*self.max_scale_factor)
            self.target_sz = torch.max(self.target_sz, self.initial_target_sz*self.min_scale_factor)

            # Update the sampling message
            self.search_area = torch.prod(self.target_sz * self.params.search_padding)
            self.sample_scale = torch.sqrt(self.search_area / self.params.img_sample_area)
            self.target_sample_sz = self.target_sz / self.sample_scale

            # ------- UPDAT FEATURE MODEL------- #
            train_sample = sample_patch(image_cuda, self.target_pos, self.sample_scale*self.img_sample_sz, self.img_sample_sz)
            train_backbone_features = self.params.model.extract_backbone_features(train_sample)

            # Extract locator features
            train_locator_proposals = self.get_locator_proposals(self.target_sample_sz)
            train_locator_features = self.params.model.extract_locator_features(train_backbone_features, train_locator_proposals).squeeze()

            # Hard negtive mining and Adaptive learning rate
            if self.params.hard_negative_mining:
                train_locator_score = torch.mm(train_locator_features, self.locator_model)
                max_score = train_locator_score.max()
                train_locator_score = train_locator_score * self.hard_negative_region_mask
                if (train_locator_score.max() > self.params.hard_negative_threshold*max_score) and (train_locator_score.max() > self.params.target_not_found):
                    hard_flag = True
                    learning_rate = self.params.hard_negative_learning_rate
                else:
                    learning_rate = self.params.learning_rate

            # Update locator model
            self.locator_XTX = (1 - learning_rate) * self.locator_XTX + learning_rate * torch.mm(train_locator_features.t(), train_locator_features)
            self.locator_XTY = (1 - learning_rate) * self.locator_XTY + learning_rate * torch.mm(train_locator_features.t(), self.locator_labels)

            # Adjust weight of initial frame
            self.current_initial_frame_weight = (1 - learning_rate) * self.current_initial_frame_weight
            if self.current_initial_frame_weight < self.params.init_samples_minimum_weight:
                diff = self.params.init_samples_minimum_weight - self.current_initial_frame_weight
                coff = diff / (1 - self.current_initial_frame_weight)
                self.locator_XTX = (1 - coff) * self.locator_XTX + coff * self.locator_XTX_initial
                self.locator_XTY = (1 - coff) * self.locator_XTY + coff * self.locator_XTY_initial
                self.current_initial_frame_weight = self.params.init_samples_minimum_weight

        # ------- TRAIN ------- #
        if (self.frame_num % self.params.train_skipping == 0) or (hard_flag):
            self.locator_model = self.train_locator_model(self.locator_XTX+self.locator_regularization, self.locator_XTY, self.locator_model)

        # ------- RETURN ------- #
        # Return new state
        new_state = torch.cat((self.target_pos[[1,0]] - self.target_sz[[1,0]]*0.5, self.target_sz[[1,0]]))
        new_state[0], new_state[1] = new_state[0].clamp(0), new_state[1].clamp(0)
        new_state[2] = new_state[2].clamp(0, self.IMG_WIDTH -new_state[0])
        new_state[3] = new_state[3].clamp(0, self.IMG_HEIGHT-new_state[1])

        # Output result image
        if self.params.output_image:
            self.output_result_image(image_show, new_state)

        return new_state.tolist()

    def numpy_to_tensor_gpu(self, image):
        image = torch.from_numpy(image)
        image = image.cuda()
        image = image.permute(2,0,1).unsqueeze(0).to(torch.float32)
        return image

    def init_locator_proposals_center_function(self):
        search_padding = self.params.search_padding
        target_sample_area = self.target_sample_area
        sigma_factor = self.params.output_sigma_factor
        proposals_num = self.params.proposals_num
        WIDTH, HEIGHT = self.img_sample_sz[1], self.img_sample_sz[0]

        ## uniform proposals
        proposals_sqrt = int(math.sqrt(proposals_num))
        x_step = ((WIDTH - WIDTH/search_padding) / (proposals_sqrt-1))
        y_step = ((HEIGHT - HEIGHT/search_padding) / (proposals_sqrt-1))

        proposals_xc = torch.arange(proposals_sqrt).repeat(proposals_sqrt).type(torch.float32) * x_step + WIDTH/(search_padding*2)
        proposals_yc = torch.arange(proposals_sqrt).repeat(proposals_sqrt,1).t().reshape(-1).type(torch.float32) * y_step + HEIGHT/(search_padding*2)

        ## creat label
        x_dist = proposals_xc - WIDTH*0.5
        y_dist = proposals_yc - HEIGHT*0.5

        sigma = sigma_factor * math.sqrt(target_sample_area)
        proposals_label = torch.exp(-0.5 * (x_dist.pow(2)+y_dist.pow(2)) / sigma**2)
        proposals_label = proposals_label.view(-1,1)

        proposals_xc = proposals_xc.to(self.params.device)
        proposals_yc = proposals_yc.to(self.params.device)
        proposals_label = proposals_label.to(self.params.device)
        return proposals_xc, proposals_yc, proposals_label

    def init_labels_function(self):
        proposals_xc = self.proposals_xc
        proposals_yc = self.proposals_yc
        sigma_factor = self.params.output_sigma_factor
        target_sample_area = self.target_sample_area

        WIDTH, HEIGHT = self.img_sample_sz[0], self.img_sample_sz[1]
        x_dist = proposals_xc - (WIDTH * 0.5).item()
        y_dist = proposals_yc - (HEIGHT * 0.5).item()

        sigma = sigma_factor * math.sqrt(target_sample_area)
        labels = torch.exp(-0.5 * (x_dist.pow(2)+y_dist.pow(2))/sigma**2)
        labels = labels.to(self.params.device).reshape(-1,1)
        return labels

    def init_output_window_function(self):
        proposals_xc = self.locator_proposals_xc
        proposals_yc = self.locator_proposals_yc
        target_sample_area = self.target_sample_area
        sigma_factor = self.params.window_sigma_factor
        window_min_value = self.params.window_min_value
        WIDTH, HEIGHT = self.img_sample_sz[1], self.img_sample_sz[0]

        x_dist = proposals_xc - 0.5*WIDTH.item()
        y_dist = proposals_yc - 0.5*HEIGHT.item()
        sigma = sigma_factor * math.sqrt(target_sample_area)
        output_window = torch.exp(-0.5 * (x_dist.pow(2)+y_dist.pow(2)) / sigma**2)
        output_window = output_window.clamp(window_min_value)
        output_window = output_window.view(-1,1)
        return output_window

    def init_hard_negative_region_function(self):
        proposals_xc = self.locator_proposals_xc
        proposals_yc = self.locator_proposals_yc
        target_sample_area = self.target_sample_area
        distance_ratio = self.params.hard_negative_distance_ratio
        WIDTH, HEIGHT = self.img_sample_sz[1], self.img_sample_sz[0]

        x_dist = proposals_xc - 0.5*WIDTH.item()
        y_dist = proposals_yc - 0.5*HEIGHT.item()
        distance = torch.sqrt(x_dist.pow(2)+y_dist.pow(2))
        distance_threshold = math.sqrt(target_sample_area*distance_ratio**2)

        mask = torch.zeros(proposals_xc.shape, device=self.params.device)
        mask[distance>distance_threshold] = 1.0
        mask = mask.view(-1,1)
        return mask

    def generate_init_samples(self, im: torch.Tensor, target_pos, sample_scale) -> TensorList:
        # Compute augmentation size
        aug_expansion_factor = getattr(self.params, 'augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift operator
        get_rand_shift = lambda: None

        # Create transofmations
        self.transforms = [augmentation.Identity(aug_output_sz)]
        if 'shift' in self.params.augmentation_method:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz) for shift in self.params.augmentation_method['shift']])
        if 'relativeshift' in self.params.augmentation_method:
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz) for shift in self.params.augmentation_method['relativeshift']])
        if 'fliplr' in self.params.augmentation_method and self.params.augmentation_method['fliplr']:
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in self.params.augmentation_method:
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in self.params.augmentation_method['blur']])
        if 'scale' in self.params.augmentation_method:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in self.params.augmentation_method['scale']])
        if 'rotate' in self.params.augmentation_method:
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in self.params.augmentation_method['rotate']])

        init_sample = sample_patch(im, target_pos, sample_scale*aug_expansion_sz, aug_expansion_sz)
        init_samples = torch.cat([T(init_sample) for T in self.transforms])
        if not self.params.augmentation:
            init_samples = init_samples[0:1,...]
        return init_samples

    def get_iounet_box(self, target_pos, target_sz, sample_pos, sample_scale):
        """All inputs in original image coordinates"""
        box_center = (target_pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2
        box_sz = target_sz / sample_scale
        target_ul = box_center - (box_sz - 1) / 2
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])

    def init_iou_net(self, target_pos, target_sz, sample_scale, init_backbone_features):
        # Setup IoU net
        for p in self.params.model.iou_predictor.parameters():
            p.requires_grad = False

        # Get target boxes and convert
        target_boxes = self.get_iounet_box(target_pos, target_sz, target_pos.round(), sample_scale)
        target_boxes = target_boxes.unsqueeze(0).to(self.params.device)

        # Remove other augmentations such as rotation
        iou_backbone_features = TensorList([x[:target_boxes.shape[0],...] for x in init_backbone_features])

        # Extract target IoU feat
        with torch.no_grad():
            target_iou_feat = self.params.model.iou_predictor.get_filter(iou_backbone_features, target_boxes)
            target_iou_feat = TensorList([x.detach().mean(0) for x in target_iou_feat])
        return target_iou_feat

    def get_locator_proposals(self, target_sample_sz):
        proposals = self.locator_proposals
        proposals_xc = self.locator_proposals_xc
        proposals_yc = self.locator_proposals_yc

        proposals[0, :, 0] = proposals_xc - 0.5*target_sample_sz[1].item()
        proposals[0, :, 1] = proposals_yc - 0.5*target_sample_sz[0].item()
        proposals[0, :, 2] = target_sample_sz[1].item()
        proposals[0, :, 3] = target_sample_sz[0].item()
        return proposals

    def train_locator_model(self, model_XTX, model_XTY, model=None):
        if model is None:
            model = torch.potrs(model_XTY, torch.potrf(model_XTX))
        else:
            for _ in range(30):
                model, _ = torch.trtrs(model_XTY - torch.mm(torch.triu(model_XTX, diagonal=1), model), torch.tril(model_XTX, diagonal=0), upper=False)
        return model

    def refine_target_box(self, target_pos, target_sz, sample_pos, sample_scale, iou_backbone_bone_features):
        top_k = self.params.iounet_k
        jitter_sz = self.params.box_jitter_sz
        jitter_pos = self.params.box_jitter_pos
        refine_num = self.params.box_refinement_iter
        max_aspect_ratio = self.params.maximal_aspect_ratio
        num_init_random_boxes = self.params.num_init_random_boxes

        # Initial boxes for refinement
        init_box = self.get_iounet_box(target_pos, target_sz, sample_pos, sample_scale)
        init_boxes = init_box.view(1,4).clone()
        if num_init_random_boxes > 0:
            # Get random initial boxes
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([jitter_pos * torch.ones(2), jitter_sz * torch.ones(2)])
            minimal_edge_size = init_box[2:].min()/3
            rand_bb = (torch.rand(num_init_random_boxes, 4) - 0.5) * rand_factor
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Refine boxes by maximizing iou
        output_boxes, output_iou = self.optimize_boxes(iou_backbone_bone_features, init_boxes)

        # Remove weird boxes with extreme aspect ratios
        output_boxes[:, 2:].clamp_(1)
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]
        keep_ind = (aspect_ratio < max_aspect_ratio) * (aspect_ratio > 1/max_aspect_ratio)
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return None

        # Take average of top k boxes
        top_k = min(top_k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, top_k)
        output_boxes = output_boxes[inds, :].mean(0).cpu()
        return output_boxes 

    def optimize_boxes(self, iou_features, init_boxes):
        # Optimize iounet boxes
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)

        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()
            bb_init.requires_grad = True

            outputs = self.params.model.iou_predictor.predict_iou(self.target_iou_feat, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes = bb_init + (bb_init.grad*100).round()/100 * bb_init[:, :, 2:].repeat(1, 1, 2)
            output_boxes.detach_()

        return output_boxes.view(-1,4), outputs.detach().view(-1)

    def output_result_image(self, image, state):
        if self.params.output_image:
            if not os.path.exists(self.params.output_image_path):
                os.mkdir(self.params.output_image_path)
            cv2.rectangle(image, (int(state[0]),int(state[1])),(int(state[0]+state[2]),int(state[1]+state[3])), (255,0,0), 3)
            cv2.imwrite(os.path.join(self.params.output_image_path,'{}.jpg'.format(self.frame_num)), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

'''
import cv2
for i in range(train_samples.shape[0]):
    output_dir = '/home/zhenglinyu2/SBDT_tracking/debug/transform_image/'
    count = len(os.listdir(output_dir))
    transform_image = train_samples[i,...].permute(1,2,0)
    transform_image = transform_image.data.numpy()
    cv2.imwrite(os.path.join(output_dir,'{}.jpg'.format(count+1)),transform_image))
'''

'''
torch.cuda.synchronize()
start = time.time()
torch.cuda.synchronize()
print(time.time() - start)
'''
