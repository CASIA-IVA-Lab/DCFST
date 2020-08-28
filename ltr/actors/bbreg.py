from . import BaseActor


class AtomActor(BaseActor):
    """ Actor for training the IoU-Net in ATOM"""
    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_anno',
                    'test_proposals' and 'proposal_iou'.

        returns:
            loss    - the training loss
            states  -  dict containing detailed losses
        """
        # Run network to obtain IoU prediction for each proposal in 'test_proposals'
        iou_pred = self.net(data['train_images'], data['test_images'], data['train_anno'], data['test_proposals'])

        iou_pred = iou_pred.view(-1, iou_pred.shape[2])
        iou_gt = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])

        # hard mining
        batch_size, sample_num = iou_gt.shape[0], iou_gt.shape[1]
        hard_num = int(0.5 * sample_num)
        total_loss = (iou_pred - iou_gt).pow(2).squeeze()
        loss = 0.0
        for i in range(batch_size):
            _, pred = total_loss[i].topk(hard_num, 0, True, True)
            loss += total_loss[i].index_select(0, pred).mean()
        loss = loss / batch_size
        
        # Compute loss
        #loss = self.objective(iou_pred, iou_gt)

        # Return training stats
        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss.item()}

        return loss, stats