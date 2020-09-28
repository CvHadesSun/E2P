from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark import layers


# @registry.ROI_KEYPOINT_FEATURE_EXTRACTORS.register("KeypointRCNNFeatureExtractor")


class KeypointRCNNFeatureExtractor(nn.Module):
    def __init__(self,in_channels):
        super(KeypointRCNNFeatureExtractor, self).__init__()

        # resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        # scales = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES
        # sampling_ratio = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        # pooler = Pooler(
        #     output_size=(resolution, resolution),
        #     scales=scales,
        #     sampling_ratio=sampling_ratio,
        # )
        # self.pooler = pooler

        input_features = in_channels
        # layers = tuple(512 for _ in range(5))
        layers=(256,128,64,32,16)
        next_feature = input_features
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "conv_refinenet{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            # nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            # nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)
        self.out_channels = layer_features

    def forward(self, x):
        # print(x)
        # x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        # print(x)
        return x

class KeypointRCNNPredictor2(nn.Module):
    def __init__(self,in_channels,num_keypoints):
        super(KeypointRCNNPredictor2, self).__init__()
        # input_features = in_channels
        # num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES
        # deconv_kernel = 4
        # self.kps_score_lowres = layers.ConvTranspose2d(
        #     input_features,
        #     num_keypoints,
        #     deconv_kernel,
        #     stride=2,
        #     padding=1,
        # )
        self.avgpool2d=nn.AdaptiveAvgPool2d(7)
        self.liner=nn.Linear(in_channels*7*7,2*num_keypoints)
        
        # nn.init.kaiming_normal_(
        #     self.liner.weight, mode="fan_out", nonlinearity="relu"
        # )
        # nn.init.constant_(self.liner.bias, 0)
        # self.up_scale = 2
        # self.out_channels = num_keypoints

    def forward(self, x):
        # x = self.kps_score_lowres(x)
        
        # x = layers.interpolate(
        #     x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        # )
        # print(self.liner)
        b,c,h,w=x.shape
        # print(x)
        x= self.avgpool2d(x).view(b,c*7*7)
        # print(x)
        
        x=self.liner(x)
        # print(x)
        # print(x.shape)

        return x


class KeypointRCNNPredictor(nn.Module):
    def __init__(self,in_channels,num_keypoints):
        super(KeypointRCNNPredictor, self).__init__()
        input_features = in_channels
        # num_keypoints = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES
        deconv_kernel = 4
        self.kps_score_lowres = layers.ConvTranspose2d(
            input_features,
            num_keypoints,
            deconv_kernel,
            stride=2,
            padding=1,
        )
        nn.init.kaiming_normal_(
            self.kps_score_lowres.weight, mode="fan_out", nonlinearity="relu"
        )
        nn.init.constant_(self.kps_score_lowres.bias, 0)
        self.up_scale = 2
        self.out_channels = num_keypoints

    def forward(self, x):
        x = self.kps_score_lowres(x)
        
        x = layers.interpolate(
            x, scale_factor=self.up_scale, mode="bilinear", align_corners=False
        )
        # print(x.shape)
        return x


class RefineNet(nn.Module):
    def __init__(self,feature_extractor,predictor):
        super(RefineNet, self).__init__()
        self.extractor=feature_extractor
        self.predictor=predictor

    def forward(self, x):
        x= self.extractor(x)

        x=self.predictor(x)

        return x