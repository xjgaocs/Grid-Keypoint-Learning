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
"""Losses for the video representation model."""

import torch
import torch.nn as nn
import numpy as np
import ops

def temporal_separation_loss(cfg, coords):
    """Encourages keypoint to have different temporal trajectories.

    If two keypoints move along trajectories that are identical up to a time-
    invariant translation (offset), this suggest that they both represent the same
    object and are redundant, which we want to avoid.

    To measure this similarity of trajectories, we first center each trajectory by
    subtracting its mean. Then, we compute the pairwise distance between all
    trajectories at each timepoint. These distances are higher for trajectories
    that are less similar. To compute the loss, the distances are transformed by
    a Gaussian and averaged across time and across trajectories.

    Args:
      cfg: ConfigDict.
      coords: [time, batch, num_landmarks, 3] coordinate tensor.

    Returns:
      Separation loss.
    """
    x = coords[Ellipsis, 0]
    y = coords[Ellipsis, 1]

    # Center trajectories:
    x = x - torch.mean(x, dim=0, keepdim=True)
    y = y - torch.mean(y, dim=0, keepdim=True)

    # Compute pairwise distance matrix:
    d = ((x[:, :, :, np.newaxis] - x[:, :, np.newaxis, :]) ** 2.0 +
         (y[:, :, :, np.newaxis] - y[:, :, np.newaxis, :]) ** 2.0)

    # Temporal mean:
    d = torch.mean(d, dim=0)

    # Apply Gaussian function such that loss falls off with distance:
    loss_matrix = torch.exp(-d / (2.0 * cfg.separation_loss_sigma ** 2.0))
    loss_matrix = torch.mean(loss_matrix, dim=0)  # Mean across batch.
    loss = torch.sum(loss_matrix)  # Sum matrix elements.

    # Subtract sum of values on diagonal, which are always 1:
    loss = loss - cfg.num_keypoints

    # Normalize by maximal possible value. The loss is now scaled between 0 (all
    # keypoints are infinitely far apart) and 1 (all keypoints are at the same
    # location):
    loss = loss / (cfg.num_keypoints * (cfg.num_keypoints - 1))


    return cfg.separation_loss_scale * loss

def sparse_loss(weight_matrix, cfg):
    """L1-loss on mean heatmap activations, to encourage sparsity."""
    weight_shape = weight_matrix.shape
    assert len(weight_shape) == 5, weight_shape

    heatmap_mean = torch.mean(weight_matrix, dim=(3, 4))
    penalty = torch.mean(torch.abs(heatmap_mean))

    return penalty * cfg.heatmap_regularization

def detect_loss(weight_matrix, gama=0.01):
    weight_shape = weight_matrix.shape
    assert len(weight_shape) == 5, weight_shape
    weight_matrix1 = weight_matrix.view(weight_matrix.size(0),
                                        weight_matrix.size(1), weight_matrix.size(2), -1)
    heatmap_max = torch.max(weight_matrix1, dim=-1)[0]
    heatmap_mean = torch.mean(weight_matrix1, dim=-1)
    max_min = torch.min(heatmap_max-heatmap_mean, dim=-1)[0]
    incite = -torch.sum(max_min)
    return gama*incite
