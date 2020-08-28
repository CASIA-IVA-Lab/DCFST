import math
import torch
import torch.nn as nn
from ltr.models.layers.blocks import LinearBlock
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class OnlineRRNet(nn.Module):
    def __init__(self, input_dim=(128,256), pred_input_dim=(128,256)):
        super().__init__()
        self.conv3_1 = conv(input_dim[0], pred_input_dim[0], kernel_size=3, stride=1)
        self.conv3_2 = conv(pred_input_dim[0], pred_input_dim[0], kernel_size=3, stride=1)
        self.conv3_3 = conv(pred_input_dim[0], pred_input_dim[0], kernel_size=3, stride=1)
        self.conv3_4 = conv(pred_input_dim[0], pred_input_dim[0], kernel_size=3, stride=1)
        self.conv4_1 = conv(input_dim[1], pred_input_dim[1], kernel_size=3, stride=1)
        self.conv4_2 = conv(pred_input_dim[1], pred_input_dim[1], kernel_size=3, stride=1)
        self.conv4_3 = conv(pred_input_dim[1], pred_input_dim[1], kernel_size=3, stride=1)
        self.conv4_4 = conv(pred_input_dim[1], pred_input_dim[1], kernel_size=3, stride=1)

        self.prroi_pool3 = PrRoIPool2D(8, 8, 1/8)
        self.prroi_pool4 = PrRoIPool2D(4, 4, 1/16)

        self.fc3 = LinearBlock(pred_input_dim[0], 512, 8, batch_norm=False, relu=False)
        self.fc4 = LinearBlock(pred_input_dim[1], 512, 4, batch_norm=False, relu=False)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, train_feat, test_feat, train_proposals, train_labels, test_proposals):
        assert(train_feat[0].shape[0]==1)

        batch_size = train_feat[0].shape[1]

        # transform
        train_feat = [f[0,...] for f in train_feat]
        test_feat = [f[0,...] for f in test_feat]
        train_labels = train_labels.view(batch_size, -1, 1)
        train_proposals = train_proposals.view(batch_size, -1, 4)
        test_proposals = test_proposals.view(batch_size, -1, 4)

        # Extract features
        train_feat_locator = self.get_locator_feat(train_feat, train_proposals)
        test_feat_locator = self.get_locator_feat(test_feat, test_proposals)

        # Train by solving the ridge regression problem
        train_XTY = torch.matmul(train_feat_locator.permute(0,2,1), train_labels)
        train_XTX = torch.matmul(train_feat_locator.permute(0,2,1), train_feat_locator)
        W, _ = torch.gesv(train_XTY, train_XTX + 0.1*torch.eye(train_feat_locator.shape[2]).to(train_XTX.device))

        # Evaluation
        prediction = torch.matmul(test_feat_locator, W)

        return prediction
        
        
    def get_locator_feat(self, feat, proposals):
        batch_size = feat[0].shape[0]
        num_proposals_per_batch = proposals.shape[1]

        # Convolution
        feat_layer_3 = self.conv3_4(self.conv3_3(self.conv3_2(self.conv3_1(feat[0]))))
        feat_layer_4 = self.conv4_4(self.conv4_3(self.conv4_2(self.conv4_1(feat[1]))))

        # Convert the xywh input proposals to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)
        
        # Add batch index to rois
        batch_index = torch.Tensor([x for x in range(batch_size)]).view(batch_size, 1).to(feat_layer_3.device)
        rois = torch.cat((batch_index.view(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1), proposals_xyxy), dim=2)
        rois = rois.view(-1, 5).to(proposals_xyxy.device)

        # Extract features for each sample roi
        feat_layer_3_roi = self.prroi_pool3(feat_layer_3, rois)
        feat_layer_4_roi = self.prroi_pool4(feat_layer_4, rois)

        # Full connection
        feat_layer_3_fc = self.fc3(feat_layer_3_roi)
        feat_layer_4_fc = self.fc4(feat_layer_4_roi)

        feat_layer_3_fc = feat_layer_3_fc.view(batch_size, num_proposals_per_batch, -1)
        feat_layer_4_fc = feat_layer_4_fc.view(batch_size, num_proposals_per_batch, -1)

        # L2 norm
        layer_3_norm = (torch.sum(feat_layer_3_fc.abs()**2, dim=1, keepdim=True) / (feat_layer_3_fc.shape[1] + 1e-10))**(1/2)
        layer_4_norm = (torch.sum(feat_layer_4_fc.abs()**2, dim=1, keepdim=True) / (feat_layer_4_fc.shape[1] + 1e-10))**(1/2)

        feat_layer_3_fc_norm = feat_layer_3_fc / layer_3_norm
        feat_layer_4_fc_norm = feat_layer_4_fc / layer_4_norm

        return torch.cat((feat_layer_3_fc_norm, feat_layer_4_fc_norm), dim=2)