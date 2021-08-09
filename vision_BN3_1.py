# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Vision-related components of the structured video representation model.

These components perform the pixels <--> keypoints transformation.
"""

import numpy as np
import torch
import torch.nn as nn
#from video_structure
import ops
import torch.nn.functional as F




class build_image_encoder(nn.Module):
    """Extracts feature maps from images.

      The encoder iteratively halves the resolution and doubles the number of
      filters until the size of the feature maps is output_map_width by
      output_map_width.

      Args:
        input_shape: Shape of the input image (without batch dimension).
        initial_num_filters: Number of filters to apply at the input resolution.
        output_map_width: Width of the output feature maps.
        layers_per_scale: How many additional size-preserving conv layers to apply
          at each map scale.
        **conv_layer_kwargs: Passed to layers.Conv2D.

      Raises:
        ValueError: If the width of the input image is not compatible with
          output_map_width, i.e. if input_width/output_map_width is not a perfect
          square.
    """
    def __init__(self, input_shape, initial_num_filters=32, output_map_width=16,
                 layers_per_scale=1, **conv_layer_kwargs):
        super(build_image_encoder, self).__init__()
        if np.log2(input_shape[1] / output_map_width) % 1:
            raise ValueError(
                'The ratio of input width and output_map_width must be a perfect '
                'square, but got {} and {} with ratio {}'.format(
                    input_shape[1], output_map_width, input_shape[1] / output_map_width))
        total_modules = []
        modules = [nn.Conv2d(input_shape[0], initial_num_filters,
                             **conv_layer_kwargs),
                   nn.BatchNorm2d(initial_num_filters),
                nn.LeakyReLU(0.2, inplace=True)]

        # Expand image to initial_num_filters maps:
        for _ in range(layers_per_scale):
            modules.extend([nn.Conv2d(initial_num_filters, initial_num_filters,
                      **conv_layer_kwargs),
                nn.BatchNorm2d(initial_num_filters),
                nn.LeakyReLU(0.2, inplace=True)])
        total_modules.append(nn.Sequential(*modules))
        modules = []

        # Apply downsampling blocks until feature map width is output_map_width:
        width = input_shape[2]
        num_filters = initial_num_filters
        while width > output_map_width:
            # Reduce resolution:
            modules.extend([nn.Conv2d(num_filters, num_filters*2,
                                      **conv_layer_kwargs),
                nn.BatchNorm2d(num_filters*2),
                nn.LeakyReLU(0.2, inplace=True)])

            # Apply additional layers:
            for _ in range(layers_per_scale):
                modules.extend([nn.Conv2d(num_filters*2, num_filters*2, **conv_layer_kwargs),
                    nn.BatchNorm2d(num_filters*2),
                    nn.LeakyReLU(0.2, inplace=True)])
            num_filters *= 2
            width //= 2
            total_modules.append(nn.Sequential(*modules))
            modules = []
        self.conv = nn.ModuleList(total_modules)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input):
        h1 = self.conv[0](input)
        h2 = self.conv[1](self.mp(h1))
        h3 = self.conv[2](self.mp(h2))
        return h3, [h2, h1]


class build_image_decoder(nn.Module):
    """Decodes images from feature maps.

      The encoder iteratively doubles the resolution and halves the number of
      filters until the size of the feature maps is output_width.

      Args:
        input_shape: Shape of the input image (without batch dimension).
        output_width: Width of the output image.
        layers_per_scale: How many additional size-preserving conv layers to apply
          at each map scale.
        **conv_layer_kwargs: Passed to layers.Conv2D.

      Raises:
        ValueError: If the width of the input feature maps is not compatible with
          output_width, i.e. if output_width/input_map_width is not a perfect
          square.
    """
    def __init__(self, input_shape, output_width, layers_per_scale=1, **conv_layer_kwargs):
        super(build_image_decoder, self).__init__()
        self.num_levels = np.log2(output_width / input_shape[2])
        if self.num_levels % 1:
            raise ValueError(
                'The ratio of output_width and input width must be a perfect '
                'square, but got {} and {} with ratio {}'.format(
                    output_width, input_shape[2], output_width / input_shape[2]))

        # Expand until we have filters_out channels:
        self.num_filters = input_shape[0]
        num_filters = input_shape[0]
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        total_modules = []
        modules = [nn.Conv2d(num_filters, num_filters, **conv_layer_kwargs),
                   nn.LeakyReLU(0.2, inplace=True)]
        # Expand image to initial_num_filters maps:
        for i in range(layers_per_scale):
            if i < layers_per_scale - 1:
                modules.extend([nn.Conv2d(num_filters, num_filters,
                                          **conv_layer_kwargs),
                                nn.LeakyReLU(0.2, inplace=True)])
            else:
                modules.extend([nn.Conv2d(num_filters, num_filters // 2,
                                          **conv_layer_kwargs),
                                nn.LeakyReLU(0.2, inplace=True)])
        total_modules.append(nn.Sequential(*modules))
        modules = []

        for i in range(int(self.num_levels)):
            modules.extend([nn.Conv2d(num_filters, num_filters // 2, **conv_layer_kwargs),
                            nn.LeakyReLU(0.2, inplace=True)])
            # Apply additional layers:
            for j in range(layers_per_scale):
                if j < layers_per_scale - 1:
                    modules.extend([nn.Conv2d(num_filters // 2, num_filters // 2, **conv_layer_kwargs),
                                    nn.LeakyReLU(0.2, inplace=True)])
                else:
                    if i < layers_per_scale - 1:
                        modules.extend([nn.Conv2d(num_filters // 2, num_filters // 4, **conv_layer_kwargs),
                                        nn.LeakyReLU(0.2, inplace=True)])
                    else:
                        modules.extend([nn.Conv2d(num_filters // 2, num_filters // 2, **conv_layer_kwargs),
                                        nn.LeakyReLU(0.2, inplace=True)])
            num_filters //= 2
            total_modules.append(nn.Sequential(*modules))
            modules = []
        self.out_filters = num_filters
        self.conv = nn.ModuleList(total_modules)

    def forward(self, x, skip):
        d1 = self.conv[0](x)
        up1 = self.up(d1)
        d2 = self.conv[1](torch.cat([up1, skip[0]], 1))
        up2 = self.up(d2)
        d3 = self.conv[2](torch.cat([up2, skip[1]], 1))
        return d3

class build_images_to_keypoints_net(nn.Module):
    """Builds a model that encodes an image into a keypoint.

    The feature maps are then reduced to num_keypoints heatmaps, and
    the heatmaps to (x, y, scale)-keypoints.

    Args:
      cfg: ConfigDict with model hyperparamters.
      image_shape: Image shape tuple: (C, H, W).

    Returns:
    A tf.keras.Model object.
    """
    def __init__(self, cfg, image_shape):
        super(build_images_to_keypoints_net, self).__init__()
        # Adjust channel number to account for add_coord_channels:
        encoder_input_shape = image_shape
        #encoder_input_shape[0] += 2
        # Build feature extractor:
        self.image_encoder = build_image_encoder(
          input_shape=encoder_input_shape,
          initial_num_filters=cfg.num_encoder_filters,
          output_map_width=cfg.heatmap_width,
          layers_per_scale=cfg.layers_per_scale,
          **cfg.conv_layer_kwargs)

        # Build final layer that maps to the desired number of heatmaps:
        self.features_to_keypoint_heatmaps = nn.Sequential(
              nn.Conv2d(cfg.img_w//cfg.heatmap_width*cfg.num_encoder_filters,
                        cfg.num_keypoints, kernel_size=1),
                nn.Sigmoid())

    def forward(self, image, pre_keypoints=None):
        #image = ops.add_coord_channels(image)
        encoded, _ = self.image_encoder(image)
        heatmaps = self.features_to_keypoint_heatmaps(encoded)
        if not pre_keypoints is None:
            pre_gaussian_maps = ops.keypoints_to_maps1(pre_keypoints, sigma=3.0)
        else:
            pre_gaussian_maps = torch.ones_like(heatmaps).cuda()
        #pre_gaussian_maps_np = pre_gaussian_maps.data.cpu().numpy()
        #heatmaps_np = heatmaps.data.cpu().numpy()
        #keymap_np = (heatmaps * pre_gaussian_maps).data.cpu().numpy()
        keypoints = ops.maps_to_keypoints1(heatmaps*pre_gaussian_maps)
        return keypoints, heatmaps

class build_keypoints_to_images_net(nn.Module):
    """Builds a model to reconstructs an image from keypoints.

    Model architecture:

      (keypoints[t], image[0], keypoints[0]) --> reconstructed_image

    For all frames image[t] we also we also concatenate the Gaussian maps for
    the keypoints obtained from the initial frame image[0]. This helps the
    decoder "inpaint" the image regions that are occluded by objects in the first
    frame.

    Args:
      cfg: ConfigDict with model hyperparameters.
      image_shape: Image shape tuple: (C, H, W).

    Returns:
      A tf.keras.Model object.
    """
    def __init__(self, cfg, image_shape):
        super(build_keypoints_to_images_net, self).__init__()
        # Build encoder net to extract appearance features from the first frame:
        self.keypoint_width = cfg.keypoint_width
        self.heatmap_width = cfg.heatmap_width
        self.appearance_feature_extractor = build_image_encoder(
          input_shape=image_shape,
          initial_num_filters=cfg.num_encoder_filters,
          layers_per_scale=cfg.layers_per_scale,
          **cfg.conv_layer_kwargs)

        # Build image decoder that goes from Gaussian maps to reconstructed images:
        num_encoder_output_channels = (
          cfg.num_encoder_filters * image_shape[1] // cfg.heatmap_width)
        input_shape = [num_encoder_output_channels, cfg.heatmap_width,
                       cfg.heatmap_width]
        self.image_decoder = build_image_decoder(
          input_shape=input_shape,
          output_width=image_shape[1],
          layers_per_scale=cfg.layers_per_scale,
          **cfg.conv_layer_kwargs)

        # Build layers to adjust channel numbers for decoder input and output image:
        kwargs = dict(cfg.conv_layer_kwargs)
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        self.adjust_channels_of_decoder_input = nn.Sequential(
            nn.Conv2d(cfg.num_keypoints
                      + cfg.num_encoder_filters*cfg.img_w//cfg.heatmap_width
                      , num_encoder_output_channels, **kwargs),
                nn.LeakyReLU(0.2, inplace=True))

        kwargs = dict(cfg.conv_layer_kwargs)
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        self.adjust_channels_of_output_image = nn.Sequential(
            nn.Conv2d(self.image_decoder.out_filters, image_shape[0], **kwargs))

    def forward(self, keypoints, first_frame, first_frame_keypoints, predicted_gaussian_maps=None):
        # Get features and maps for first frame:
        # Note that we cannot use the Gaussian maps above because the
        # first_frame_keypoints may be different than the keypoints (i.e. obs vs
        # pred).
        first_frame_features, skip = self.appearance_feature_extractor(first_frame)
        first_frame_gaussian_maps = ops.keypoints_to_maps2(first_frame_keypoints,
                                                          sigma=self.keypoint_width,
                                                          heatmap_width=self.heatmap_width)

        # Convert keypoints to pixel maps:
        if predicted_gaussian_maps is None:
            gaussian_maps = ops.keypoints_to_maps2(keypoints,
                                              sigma=self.keypoint_width,
                                              heatmap_width=self.heatmap_width)
        else:
            gaussian_maps = predicted_gaussian_maps

        # Reconstruct image:
        gaussian_maps = gaussian_maps - first_frame_gaussian_maps
        combined_representation = torch.cat((gaussian_maps, first_frame_features), 1)
        #combined_representation = ops.add_coord_channels(combined_representation)
        combined_representation = self.adjust_channels_of_decoder_input(
                      combined_representation)
        decoded_representation = self.image_decoder(combined_representation, skip)
        image = self.adjust_channels_of_output_image(decoded_representation)

        # Add in the first frame of the sequence such that the model only needs to
        # predict the change from the first frame:
        image = image + first_frame

        return image

class build_image_decoder_2image(nn.Module):
    def __init__(self, input_shape, output_width, layers_per_scale=1, **conv_layer_kwargs):
        super(build_image_decoder_2image, self).__init__()
        self.num_levels = np.log2(output_width / input_shape[2])
        if self.num_levels % 1:
            raise ValueError(
                'The ratio of output_width and input width must be a perfect '
                'square, but got {} and {} with ratio {}'.format(
                    output_width, input_shape[2], output_width / input_shape[2]))

        # Expand until we have filters_out channels:
        self.num_filters = input_shape[0]
        num_filters = input_shape[0]
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        total_modules = []
        modules = [nn.Conv2d(num_filters, num_filters, **conv_layer_kwargs),
                   nn.LeakyReLU(0.2, inplace=True)]
        # Expand image to initial_num_filters maps:
        for i in range(layers_per_scale):
            if i < layers_per_scale - 1:
                modules.extend([nn.Conv2d(num_filters, num_filters,
                                          **conv_layer_kwargs),
                                nn.BatchNorm2d(num_filters),
                                nn.LeakyReLU(0.2, inplace=True)])
            else:
                modules.extend([nn.Conv2d(num_filters, num_filters // 2,
                                          **conv_layer_kwargs),
                                nn.BatchNorm2d(num_filters//2),
                                nn.LeakyReLU(0.2, inplace=True)])
        total_modules.append(nn.Sequential(*modules))
        modules = []

        for i in range(int(self.num_levels)):
            modules.extend([nn.Conv2d(num_filters, num_filters // 2, **conv_layer_kwargs),
                                nn.BatchNorm2d(num_filters//2),
                            nn.LeakyReLU(0.2, inplace=True)])
            # Apply additional layers:
            for j in range(layers_per_scale):
                if j < layers_per_scale - 1:
                    modules.extend([nn.Conv2d(num_filters // 2, num_filters // 2, **conv_layer_kwargs),
                                nn.BatchNorm2d(num_filters//2),
                                    nn.LeakyReLU(0.2, inplace=True)])
                else:
                    if i < layers_per_scale - 1:
                        modules.extend([nn.Conv2d(num_filters // 2, num_filters // 4, **conv_layer_kwargs),
                                nn.BatchNorm2d(num_filters//4),
                                        nn.LeakyReLU(0.2, inplace=True)])
                    else:
                        modules.extend([nn.Conv2d(num_filters // 2, num_filters // 2, **conv_layer_kwargs),
                                nn.BatchNorm2d(num_filters//2),
                                        nn.LeakyReLU(0.2, inplace=True)])
            num_filters //= 2
            total_modules.append(nn.Sequential(*modules))
            modules = []
        self.out_filters = num_filters
        self.conv = nn.ModuleList(total_modules)

    def forward(self, x, skip):
        d1 = self.conv[0](x)
        up1 = self.up(d1)
        d2 = self.conv[1](torch.cat([up1, skip[0]], 1))
        up2 = self.up(d2)
        d3 = self.conv[2](torch.cat([up2, skip[1]], 1))
        return d3

class build_keypoints_to_images_net_2image(nn.Module):
    def __init__(self, cfg, image_shape):
        super(build_keypoints_to_images_net_2image, self).__init__()
        # Build encoder net to extract appearance features from the first frame:
        self.keypoint_width = cfg.keypoint_width
        self.heatmap_width = cfg.heatmap_width
        self.appearance_feature_extractor = build_image_encoder(
          input_shape=image_shape,
          initial_num_filters=cfg.num_encoder_filters,
          layers_per_scale=cfg.layers_per_scale,
          **cfg.conv_layer_kwargs)

        # Build image decoder that goes from Gaussian maps to reconstructed images:
        num_encoder_output_channels = (
          cfg.num_encoder_filters * image_shape[1] // cfg.heatmap_width)
        input_shape = [num_encoder_output_channels, cfg.heatmap_width,
                       cfg.heatmap_width]
        self.image_decoder = build_image_decoder_2image(
          input_shape=input_shape,
          output_width=image_shape[1],
          layers_per_scale=cfg.layers_per_scale,
          **cfg.conv_layer_kwargs)

        # Build layers to adjust channel numbers for decoder input and output image:
        kwargs = dict(cfg.conv_layer_kwargs)
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        self.adjust_channels_of_decoder_input = nn.Sequential(
            nn.Conv2d(cfg.num_keypoints*3
                      + cfg.num_encoder_filters*cfg.img_w//cfg.heatmap_width*2
                      , num_encoder_output_channels, **kwargs),
                      nn.BatchNorm2d(num_encoder_output_channels),
                nn.LeakyReLU(0.2, inplace=True))

        kwargs = dict(cfg.conv_layer_kwargs)
        kwargs['kernel_size'] = 1
        kwargs['padding'] = 0
        self.adjust_channels_of_output_image = nn.Sequential(
            nn.Conv2d(self.image_decoder.out_filters, image_shape[0], **kwargs))

    def forward(self, keypoints, first_frame, first_frame_keypoints, sec_frame, sec_frame_keypoints):
        # Get features and maps for first frame:
        # Note that we cannot use the Gaussian maps above because the
        # first_frame_keypoints may be different than the keypoints (i.e. obs vs
        # pred).
        first_frame_features, skip = self.appearance_feature_extractor(first_frame)
        first_frame_gaussian_maps = ops.keypoints_to_maps2(first_frame_keypoints,
                                                          sigma=self.keypoint_width,
                                                          heatmap_width=self.heatmap_width)

        sec_frame_features, _ = self.appearance_feature_extractor(sec_frame)
        sec_frame_gaussian_maps = ops.keypoints_to_maps2(sec_frame_keypoints,
                                                           sigma=self.keypoint_width,
                                                           heatmap_width=self.heatmap_width)

        # Convert keypoints to pixel maps:
        gaussian_maps = ops.keypoints_to_maps2(keypoints,
                                              sigma=self.keypoint_width,
                                              heatmap_width=self.heatmap_width)

        # Reconstruct image:
        combined_representation = torch.cat((gaussian_maps, first_frame_gaussian_maps,
                                             sec_frame_gaussian_maps,
                                             first_frame_features,
                                             sec_frame_features), 1)
        #combined_representation = ops.add_coord_channels(combined_representation)
        combined_representation = self.adjust_channels_of_decoder_input(
                      combined_representation)
        decoded_representation = self.image_decoder(combined_representation, skip)
        image = self.adjust_channels_of_output_image(decoded_representation)

        # Add in the first frame of the sequence such that the model only needs to
        # predict the change from the first frame:
        image = torch.tanh(image)

        return image

