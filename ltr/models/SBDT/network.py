import torch.nn as nn
import ltr.models.backbone as backbones
import ltr.models.bbreg as bbmodels
import ltr.models.locator as locmodels
from ltr import model_constructor


class SBDTNet(nn.Module):
    """ SBDTNet network module"""
    def __init__(self, feature_extractor, feature_layer, bb_regressor, location_predictor, extractor_grad):
        """
        args:
            feature_extractor - backbone feature extractor
            feature_layer - List containing the name of the layers from feature_extractor, which are input to bb_regressor and location_predictor
            bb_regressor - IoU prediction module
            location_predictor - location prediction module
            extractor_grad - Bool indicating whether backbone feature extractor requires gradients
        """
        super(SBDTNet, self).__init__()

        self.feature_extractor = feature_extractor
        self.feature_layer = feature_layer
        self.bb_regressor = bb_regressor
        self.location_predictor = location_predictor
        
        self.feature_extractor = self.feature_extractor.eval()

        if not extractor_grad:
            for p in self.feature_extractor.parameters():
                p.requires_grad_(False)


    def forward(self, train_imgs, test_imgs, train_bb, test_scale_proposals, train_locator_proposals, train_locator_labels, test_locator_proposals):
        """ Forward pass
        Note: If the training is done in sequence mode, that is, test_imgs.dim() == 5, then the batch dimension
        corresponds to the first dimensions. test_imgs is thus of the form [sequence, batch, feature, row, col]
        """
        num_sequences = train_imgs.shape[-4]
        num_train_images = train_imgs.shape[0] if train_imgs.dim() == 5 else 1
        num_test_images = test_imgs.shape[0] if test_imgs.dim() == 5 else 1

        # Extract backbone features
        train_feat = self.extract_backbone_features(train_imgs.view(-1, train_imgs.shape[-3], train_imgs.shape[-2], train_imgs.shape[-1]))
        test_feat = self.extract_backbone_features(test_imgs.view(-1, test_imgs.shape[-3], test_imgs.shape[-2], test_imgs.shape[-1]))

        # For clarity, send the features to bb_regressor in sequence form, i.e. [sequence, batch, feature, row, col]
        train_feat = [feat.view(num_train_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1]) for feat in train_feat.values()]
        test_feat = [feat.view(num_test_images, num_sequences, feat.shape[-3], feat.shape[-2], feat.shape[-1]) for feat in test_feat.values()]

        # Obtain iou prediction
        iou_pred = self.bb_regressor(train_feat, test_feat,
                                     train_bb.view(num_train_images, num_sequences, 4),
                                     test_scale_proposals.view(num_train_images, num_sequences, -1, 4))

        # Obtain regression value prediction for location
        locator_pred = self.location_predictor(train_feat, test_feat,
                                               train_locator_proposals.view(num_train_images, num_sequences, -1, 4),
                                               train_locator_labels.view(num_train_images, num_sequences, -1, 1),
                                               test_locator_proposals.view(num_train_images, num_sequences, -1, 4))
 
        return iou_pred, locator_pred


    def extract_backbone_features(self, im, layers=None):
        if layers is None:
            layers = list(set(self.feature_layer))
        return self.feature_extractor(im, layers)

    def extract_features(self, im, layers):
        return self.feature_extractor(im, layers)


@model_constructor
def SBDT_resnet18(input_dim=(128, 256), locator_inter_dim=(128,256), iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet18(pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(input_dim=input_dim, pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # locator
    location_predictor = locmodels.OnlineRRNet18(input_dim=input_dim, pred_input_dim=locator_inter_dim)

    # SBDTNet
    net = SBDTNet(feature_extractor=backbone_net, feature_layer=['layer2', 'layer3'], bb_regressor=iou_predictor, location_predictor=location_predictor, extractor_grad=False)

    return net

@model_constructor
def SBDT_resnet50(input_dim=(512, 1024), locator_inter_dim=(128,256), iou_input_dim=(256,256), iou_inter_dim=(256,256), backbone_pretrained=True):
    # backbone
    backbone_net = backbones.resnet50(pretrained=backbone_pretrained)

    # Bounding box regressor
    iou_predictor = bbmodels.AtomIoUNet(input_dim=input_dim, pred_input_dim=iou_input_dim, pred_inter_dim=iou_inter_dim)

    # locator
    location_predictor = locmodels.OnlineRRNet50(input_dim=input_dim, pred_input_dim=locator_inter_dim)

    # SBDTNet
    net = SBDTNet(feature_extractor=backbone_net, feature_layer=['layer2', 'layer3'], bb_regressor=iou_predictor, location_predictor=location_predictor, extractor_grad=False)

    return net
