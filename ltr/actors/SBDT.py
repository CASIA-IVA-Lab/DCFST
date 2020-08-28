from . import BaseActor

import torch
import torch.nn as nn

class SBDTActor(BaseActor):
    """ Actor for training the SBDT"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno', 'test_scale_proposals', 
                'proposal_iou', 'train_locator_proposals', 'train_locator_labels', 'test_locator_proposals', 'test_locator_labels'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_scale_proposals' and regression values for each proposal in 'test_locator_proposals'
        iou_pred, locator_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_scale_proposals'],\
        	data['train_locator_proposals'], data['train_locator_labels'], data['test_locator_proposals'])

        iou_pred = iou_pred.squeeze()
        locator_pred = locator_pred.squeeze()
        data['proposal_iou'] = data['proposal_iou'].squeeze()
        data['test_locator_labels'] = data['test_locator_labels'].squeeze()

        # L2 loss for IoU regression in ATOM
        iou_gt = data['proposal_iou']
        iou_loss = nn.MSELoss()(iou_pred, iou_gt)

        # Shrinkage loss for locator regression
        locator_gt = data['test_locator_labels']
        locator_pred = locator_pred.clamp(0-0.5,1+0.5)
        locator_absolute_error = (locator_pred - locator_gt).abs()
        locator_loss = torch.exp(locator_gt) * locator_absolute_error.pow(2) / (1 + torch.exp(10*(0.2-locator_absolute_error)))
        locator_loss = locator_loss.mean()

        # Return training stats
        loss = iou_loss + locator_loss
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': iou_loss.item(),
                 'Loss/locator': locator_loss.item()}

        return loss, stats