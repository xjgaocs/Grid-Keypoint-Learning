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
"""PyTorch ops for the structured video representation model."""

import enum
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

EPSILON = 1e-20  # Constant for numerical stability.


class Axis(enum.Enum):
  """Maps axes to image indices, assuming that 0th dimension is the batch."""
  y = 2
  x = 3


def maps_to_keypoints(heatmaps):
  """Turns feature-detector heatmaps into (x, y, scale) keypoints.

  This function takes a tensor of feature maps as input. Each map is normalized
  to a probability distribution and the location of the mean of the distribution
  (in image coordinates) is computed. This location is used as a low-dimensional
  representation of the heatmap (i.e. a keypoint).

  To model keypoint presence/absence, the mean intensity of each feature map is
  also computed, so that each keypoint is represented by an (x, y, scale)
  triplet.

  Args:
    heatmaps: [batch_size, num_keypoints, H, W] tensors.
  Returns:
    A [batch_size, num_keypoints, 3] tensor with (x, y, scale)-triplets for each
    keypoint. Coordinate range is [-1, 1] for x and y, and [0, 1] for scale.
  """

  # Check that maps are non-negative:
  map_min = torch.min(heatmaps)
  if map_min < 0.0:
    print("map_min: ", map_min.detach().cpu().numpy())
  assert map_min >= 0.0

  x_coordinates = _maps_to_coordinates(heatmaps, Axis.x)
  y_coordinates = _maps_to_coordinates(heatmaps, Axis.y)
  map_scales = torch.mean(heatmaps, dim=(2, 3))
  '''
  map_scales_np = map_scales.data.cpu().numpy()
  heatmaps_np = heatmaps.data.cpu().numpy()
  map_scales_signp = torch.mean(torch.sigmoid(heatmaps), dim=(2, 3)).data.cpu().numpy()
  heatmaps_signp = torch.sigmoid(heatmaps).data.cpu().numpy()
  map_scales_max = torch.max(torch.sigmoid(heatmaps), dim = -1, keepdim=True)[0]
  map_scales_max = torch.max(map_scales_max, dim=-2, keepdim=True)[0]
  heatmaps1 = torch.sigmoid(heatmaps) - 0.9 * map_scales_max
  heatmaps1 = F.relu(heatmaps1)*10
  heatmaps1_np = heatmaps1.data.cpu().numpy()
  xy = change2xy(torch.stack((x_coordinates, y_coordinates), dim=-1), width=16)

  #'''

  # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
  # degeneracy between the encoder and decoder heatmap scales and ensures that
  # the scales are in a reasonable range for the RNN:
  map_scales /= (EPSILON + torch.max(map_scales, dim=-1, keepdim=True)[0])

  return torch.stack((x_coordinates, y_coordinates, map_scales), dim=-1)

def maps_to_keypoints1(heatmaps):
  """
  do not use miu
  """

  # Check that maps are non-negative:
  #map_min = torch.min(heatmaps)
  #if map_min < 0.0:
    #print("map_min: ", map_min.detach().cpu().numpy())
  #assert map_min >= 0.0

  #heatmaps_np = heatmaps.data.cpu().numpy()

  x_coordinates = _maps_to_coordinates(heatmaps, Axis.x)
  y_coordinates = _maps_to_coordinates(heatmaps, Axis.y)

  # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
  # degeneracy between the encoder and decoder heatmap scales and ensures that
  # the scales are in a reasonable range for the RNN:


  return torch.stack((x_coordinates, y_coordinates), dim=-1)

def maps_to_keypoints1_1(heatmaps):
  """
  use max value as miu
  """

  # Check that maps are non-negative:
  map_min = torch.min(heatmaps)
  if map_min < 0.0:
    print("map_min: ", map_min.detach().cpu().numpy())
  assert map_min >= 0.0

  #heatmaps_np = heatmaps.data.cpu().numpy()

  x_coordinates = _maps_to_coordinates(heatmaps, Axis.x)
  y_coordinates = _maps_to_coordinates(heatmaps, Axis.y)

  map_scales = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), dim=-1)[0]

  # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
  # degeneracy between the encoder and decoder heatmap scales and ensures that
  # the scales are in a reasonable range for the RNN:


  return torch.stack((x_coordinates, y_coordinates, map_scales), dim=-1)

def maps_to_keypoints2(heatmaps):
  """
  do not use miu
  use max point as coordinate
  """

  # Check that maps are non-negative:
  map_min = torch.min(heatmaps)
  if map_min < 0.0:
    print("map_min: ", map_min.detach().cpu().numpy())
  assert map_min >= 0.0
  #heatmaps_np0 = heatmaps.data.cpu().numpy()
  heatmaps1 = heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1)
  topk = torch.topk(heatmaps1, k=1 + 1, dim=-1)[0]
  heatmaps = heatmaps - topk[:, :, -1, np.newaxis, np.newaxis]
  heatmaps = F.relu(heatmaps)
  #heatmaps_np = heatmaps.data.cpu().numpy()

  x_coordinates = _maps_to_coordinates(heatmaps, Axis.x)
  y_coordinates = _maps_to_coordinates(heatmaps, Axis.y)

  # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
  # degeneracy between the encoder and decoder heatmap scales and ensures that
  # the scales are in a reasonable range for the RNN:
  return torch.stack((x_coordinates, y_coordinates), dim=-1)

def maps_to_keypoints3(heatmaps):
  """
  use miu
  use max point as coordinate
  """

  # Check that maps are non-negative:
  map_min = torch.min(heatmaps)
  if map_min < 0.0:
    print("map_min: ", map_min.detach().cpu().numpy())
  assert map_min >= 0.0
  heatmaps1 = heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1)
  topk = torch.topk(heatmaps1, k=1 + 1, dim=-1)[0]
  heatmaps = heatmaps - topk[:, :, -1, np.newaxis, np.newaxis]
  heatmaps = F.relu(heatmaps)
  x_coordinates = _maps_to_coordinates(heatmaps, Axis.x)
  y_coordinates = _maps_to_coordinates(heatmaps, Axis.y)
  map_scales = topk[:,:,0]

  # Normalize map scales to [0.0, 1.0] across keypoints. This removes a
  # degeneracy between the encoder and decoder heatmap scales and ensures that
  # the scales are in a reasonable range for the RNN:
  map_scales /= (EPSILON + torch.max(map_scales, dim=-1, keepdim=True)[0])
  #map_scales_np = map_scales.data.cpu().numpy()

  return torch.stack((x_coordinates, y_coordinates, map_scales), dim=-1)

def maps_to_keypoints4(heatmaps):
  """
  consider variance
  """
  x_coordinates = _maps_to_coordinates(heatmaps, Axis.x)
  y_coordinates = _maps_to_coordinates(heatmaps, Axis.y)
  #x_variance = _maps_to_variance(heatmaps, Axis.x, x_coordinates)
  #y_variance = _maps_to_variance(heatmaps, Axis.y, y_coordinates)
  return torch.stack((x_coordinates, y_coordinates), dim=-1)

#------------------------------key areas-------------------------------
def maps_to_keyareas(heatmaps):

  # Check that maps are non-negative:
  map_min = torch.min(heatmaps)
  if map_min < 0.0:
    print("map_min: ", map_min.detach().cpu().numpy())
  assert map_min >= 0.0
  map_scales_max = torch.max(heatmaps, dim=-1, keepdim = True)[0]
  map_scales_max = torch.max(map_scales_max, dim=-2, keepdim=True)[0]
  heatmaps = heatmaps/map_scales_max
  #heatmaps_np = heatmaps.data.cpu().numpy()

  return heatmaps

def maps_to_keyareas1_1(heatmaps):

  # Check that maps are non-negative:
  map_min = torch.min(heatmaps)
  if map_min < 0.0:
    print("map_min: ", map_min.detach().cpu().numpy())
  assert map_min >= 0.0
  heatmaps = heatmaps - 0.8
  heatmaps = F.relu(heatmaps)*5
  heatmaps_np = heatmaps.data.cpu().numpy()

  return heatmaps

def maps_to_keyareas1(heatmaps):
  heatmaps1 = heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1)
  topk = torch.topk(heatmaps1, k=48+1, dim=-1)[0]
  heatmaps = heatmaps - topk[:, :, -1, np.newaxis, np.newaxis]
  heatmaps = torch.tanh(heatmaps * 10000)
  heatmaps = F.relu(heatmaps)
  #heatmaps_np = heatmaps.data.cpu().numpy()

  return heatmaps

def maps_to_keyareas2(heatmaps):
  heatmaps1 = heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1)
  topk = torch.topk(heatmaps1, k=1+1, dim=-1)[0]
  heatmaps = heatmaps - topk[:, :, -1, np.newaxis, np.newaxis]
  heatmaps = F.relu(heatmaps)
  heatmaps = torch.tanh(heatmaps * 10000)
  heatmaps = torch.sum(heatmaps, dim=1, keepdim=True)
  #heatmaps_np = heatmaps.data.cpu().numpy()

  return heatmaps


def _maps_to_coordinates(maps, axis):
  """Reduces heatmaps to coordinates along one axis (x or y).

  Args:
    maps: [batch_size, num_keypoints, H, W] tensors.
    axis: Axis Enum.

  Returns:
    A [batch_size, num_keypoints, 2] tensor with (x, y)-coordinates.
  """

  width = maps.shape[axis.value]
  grid = _get_pixel_grid(axis, width)
  shape = [1, 1, 1, 1]
  shape[axis.value] = -1
  grid = grid.view(shape)

  if axis == Axis.x:
    marginalize_dim = 2
  elif axis == Axis.y:
    marginalize_dim = 3

  # Normalize the heatmaps to a probability distribution (i.e. sum to 1):
  weights = torch.sum(maps, dim=marginalize_dim, keepdim=True)

  weights /= torch.sum(weights, dim=axis.value, keepdim=True) + EPSILON
  #weights_np = weights.data.cpu().numpy()

  # Compute the center of mass of the marginalized maps to obtain scalar
  # coordinates:
  coordinates = torch.sum(weights * grid, dim=axis.value, keepdim=True)
  coordinates = torch.squeeze(coordinates, -1)
  coordinates = torch.squeeze(coordinates, -1)

  return coordinates

def _maps_to_variance(maps, axis, miu):
  """Reduces heatmaps to variances along one axis (x or y).
  """

  width = maps.shape[axis.value]
  grid = _get_pixel_grid(axis, width)
  shape = [1, 1, 1, 1]
  shape[axis.value] = -1
  grid = grid.view(shape)

  if axis == Axis.x:
    marginalize_dim = 2
  elif axis == Axis.y:
    marginalize_dim = 3

  # Normalize the heatmaps to a probability distribution (i.e. sum to 1):
  weights = torch.sum(maps + EPSILON, dim=marginalize_dim, keepdim=True)

  weights /= torch.sum(weights, dim=axis.value, keepdim=True)
  miu = miu[:, :, np.newaxis, np.newaxis]
  var = (grid-miu)**2
  variance = torch.sum(weights * var, dim=axis.value, keepdim=True)
  variance = torch.squeeze(variance, -1)
  variance = torch.squeeze(variance, -1)
  return variance

def keypoints_to_maps(keypoints, sigma=1.0, heatmap_width=16):
  """Turns (x, y, scale)-tuples into pixel maps with a Gaussian blob at (x, y).

  Args:
    keypoints: [batch_size, num_keypoints, 3] tensor of keypoints where the last
      dimension contains (x, y, scale) triplets.
    sigma: Std. dev. of the Gaussian blob, in units of heatmap pixels.
    heatmap_width: Width of output heatmaps in pixels.

  Returns:
    A [batch_size, num_keypoints, heatmap_width, heatmap_width] tensor.
  """

  coordinates, map_scales = torch.split(keypoints, 2, dim=-1)

  def get_grid(axis):
    grid = _get_pixel_grid(axis, heatmap_width)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    return grid.view(shape)

  # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
  x_coordinates = coordinates[:, :, np.newaxis, np.newaxis, 0]
  y_coordinates = coordinates[:, :, np.newaxis, np.newaxis, 1]

  # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
  #sigma = torch.FloatTensor(sigma)
  keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0

  x_vec = torch.exp(-(get_grid(Axis.x) - x_coordinates)**2/keypoint_width)
  y_vec = torch.exp(-(get_grid(Axis.y) - y_coordinates)**2/keypoint_width)
  maps = torch.mul(x_vec, y_vec)

  #npmaps0 = maps.data.cpu().numpy()
  maps = maps * map_scales[:, :, np.newaxis, np.newaxis, 0]
  #npmaps = maps.data.cpu().numpy()
  #maps = torch.sum(maps, dim=1, keepdim=True)
  #npmaps1 = maps.detach().cpu().numpy()

  return maps

def keypoints_to_maps1(keypoints, sigma=1.0, heatmap_width=16):
  """
  do not use miu
  """
  def get_grid(axis):
    grid = _get_pixel_grid(axis, heatmap_width)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    return grid.view(shape)

  # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
  x_coordinates = keypoints[:, :, np.newaxis, np.newaxis, 0]
  y_coordinates = keypoints[:, :, np.newaxis, np.newaxis, 1]

  # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
  #sigma = torch.FloatTensor(sigma)
  keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0

  x_vec = torch.exp(-(get_grid(Axis.x) - x_coordinates)**2/keypoint_width)
  y_vec = torch.exp(-(get_grid(Axis.y) - y_coordinates)**2/keypoint_width)
  maps = torch.mul(x_vec, y_vec)
  #maps_np = maps.data.cpu().numpy()

  return maps

def keypoints_to_maps2(keypoints, sigma=1.0, heatmap_width=16):
  """
  do not use miu
  re nomalize max value
  """
  def get_grid(axis):
    grid = _get_pixel_grid(axis, heatmap_width)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    return grid.view(shape)

  # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
  x_coordinates = keypoints[:, :, np.newaxis, np.newaxis, 0]
  y_coordinates = keypoints[:, :, np.newaxis, np.newaxis, 1]

  # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
  #sigma = torch.FloatTensor(sigma)
  keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0

  x_vec = torch.exp(-(get_grid(Axis.x) - x_coordinates)**2/keypoint_width)
  y_vec = torch.exp(-(get_grid(Axis.y) - y_coordinates)**2/keypoint_width)
  maps = torch.mul(x_vec, y_vec)
  maps_max = torch.max(maps.view(maps.size(0), maps.size(1), -1), dim=-1)[0]
  maps = maps / maps_max[:, :, np.newaxis, np.newaxis]
  #maps_np = maps.data.cpu().numpy()

  return maps

def keypoints_to_edgemaps(keypoints, neighbor_link, sigma=1.0, heatmap_width=16):
  """
  do not use miu
  """
  def get_grid(axis):
    grid = _get_pixel_grid(axis, heatmap_width)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    return grid.view(shape)

  # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
  x_coordinates = keypoints[:, :, np.newaxis, np.newaxis, 0]
  y_coordinates = keypoints[:, :, np.newaxis, np.newaxis, 1]

  # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
  #sigma = torch.FloatTensor(sigma)
  keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0

  x_vec = torch.exp(-(get_grid(Axis.x) - x_coordinates)**2/keypoint_width)
  y_vec = torch.exp(-(get_grid(Axis.y) - y_coordinates)**2/keypoint_width)
  maps = torch.mul(x_vec, y_vec)
  edgemaps = []
  for edge in neighbor_link:
    edgemaps.append(maps[:, edge[0]]+maps[:, edge[1]])
  edgemaps = torch.stack(edgemaps, dim = 1)
  #maps_np = maps.data.cpu().numpy()
  #edgemaps_np = edgemaps.data.cpu().numpy()

  return edgemaps

def keypoints_to_edgemaps1(keypoints, neighbor_link, sigma=1.0, heatmap_width=16):
  """
  do not use miu
  """
  def get_grid(axis):
    grid = _get_pixel_grid(axis, heatmap_width)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    return grid.view(shape)

  # Expand to [batch_size, num_keypoints, 1, 1] for broadcasting later:
  x_coordinates = keypoints[:, :, np.newaxis, np.newaxis, 0]
  y_coordinates = keypoints[:, :, np.newaxis, np.newaxis, 1]

  # Create two 1-D Gaussian vectors (marginals) and multiply to get a 2-d map:
  #sigma = torch.FloatTensor(sigma)
  keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0

  x_vec = torch.exp(-(get_grid(Axis.x) - x_coordinates)**2/keypoint_width)
  y_vec = torch.exp(-(get_grid(Axis.y) - y_coordinates)**2/keypoint_width)
  maps = torch.mul(x_vec, y_vec)
  #maps_np = maps.data.cpu().numpy()
  edgemaps = []
  d = 1.5/(heatmap_width-1)*2
  for edge in neighbor_link:
    edgemaps.append(maps[:, edge[0]] + maps[:, edge[1]])
    dist = ((keypoints[:, edge[0], 0] - keypoints[:, edge[1], 0]) ** 2+
            (keypoints[:, edge[0], 1] - keypoints[:, edge[1], 1]) ** 2)**0.5
    for j in range(keypoints.size(0)):
      if dist[j]>d*2:
        x_coordinates1 = 0.5 * (keypoints[j, edge[0], 0] +
                                keypoints[j, edge[1], 0])
        y_coordinates1 = 0.5 * (keypoints[j, edge[0], 1] +
                                keypoints[j, edge[1], 1])
        x_vec1 = torch.exp(-(get_grid(Axis.x) - x_coordinates1) ** 2 / keypoint_width)
        y_vec1 = torch.exp(-(get_grid(Axis.y) - y_coordinates1) ** 2 / keypoint_width)
        maps1 = torch.mul(x_vec1, y_vec1)
        edgemaps[-1][j] += maps1[0,0]
    '''
    x_coordinates1 = 0.5*(keypoints[:, edge[0], np.newaxis, np.newaxis, np.newaxis, 0]+
                          keypoints[:, edge[1], np.newaxis, np.newaxis, np.newaxis, 0])
    y_coordinates1 = 0.5*(keypoints[:, edge[0], np.newaxis, np.newaxis, np.newaxis, 1]+
                          keypoints[:, edge[1], np.newaxis, np.newaxis, np.newaxis, 1])
    x_vec1 = torch.exp(-(get_grid(Axis.x) - x_coordinates1) ** 2 / keypoint_width)
    y_vec1 = torch.exp(-(get_grid(Axis.y) - y_coordinates1) ** 2 / keypoint_width)
    maps1 = torch.mul(x_vec1, y_vec1)
    '''
    #maps1_np = maps1.data.cpu().numpy()
  edgemaps = torch.stack(edgemaps, dim = 1)
  #edgemaps_np = edgemaps.data.cpu().numpy()

  return edgemaps

def keypoints_to_edgemaps2(keypoints, neighbor_link, sigma=1.0, heatmap_width=16):
  """
  do not use miu
  rotation Gaussian
  """
  def get_grid(axis):
    grid = _get_pixel_grid(axis, heatmap_width)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    return grid.view(shape)

  keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0

  edgemaps = []
  S = torch.FloatTensor([[1,0],[0,1/4]]).cuda()
  for edge in neighbor_link:
    x_coordinates1 = 0.5*(keypoints[:, edge[0], np.newaxis, np.newaxis, np.newaxis, 0]+
                          keypoints[:, edge[1], np.newaxis, np.newaxis, np.newaxis, 0])
    y_coordinates1 = 0.5*(keypoints[:, edge[0], np.newaxis, np.newaxis, np.newaxis, 1]+
                          keypoints[:, edge[1], np.newaxis, np.newaxis, np.newaxis, 1])
    theta = torch.atan((keypoints[:, edge[0], 1] - keypoints[:, edge[1], 1])/\
                       (keypoints[:, edge[0], 0] - keypoints[:, edge[1], 0] + EPSILON))
    cos_theta = torch.cos(-theta)
    sin_theta = torch.sin(-theta)
    R1 = torch.stack([cos_theta, -sin_theta], dim=-1)
    R2 = torch.stack([sin_theta, cos_theta], dim=-1)
    R = torch.stack([R1, R2], dim=1)
    RT = torch.einsum('...ij->...ji', R)
    Sigma = torch.einsum('bij, jk, bkm->bim', R, S, RT)
    x_vec1 = -Sigma[:, np.newaxis, np.newaxis, np.newaxis, 0, 0]*(get_grid(Axis.x) - x_coordinates1) ** 2 / keypoint_width
    y_vec1 = -Sigma[:, np.newaxis, np.newaxis, np.newaxis, 1, 1]*(get_grid(Axis.y) - y_coordinates1) ** 2 / keypoint_width
    xy_vec1 = -2*Sigma[:, np.newaxis, np.newaxis, np.newaxis, 0, 1]*(get_grid(Axis.x) - x_coordinates1)*(get_grid(Axis.y) - y_coordinates1) / keypoint_width
    maps1 = torch.exp(x_vec1+y_vec1+xy_vec1)
    #maps1_np = maps1.data.cpu().numpy()
    edgemaps.append(maps1)
  edgemaps = torch.stack(edgemaps, dim = 1).squeeze()
  #edgemaps = torch.sum(edgemaps, dim=1, keepdim=True)
  edgemaps_max = torch.max(edgemaps.view(edgemaps.size(0),edgemaps.size(1), -1), dim=-1)[0]
  edgemaps = edgemaps/edgemaps_max[:, :, np.newaxis, np.newaxis]
  #edgemaps_np = edgemaps.data.cpu().numpy()

  return edgemaps

def keypoints_to_edgemaps3(keypoints, neighbor_link, sigma=1.0, heatmap_width=16):
  """
  do not use miu
  rotation Gaussian
  changable variance
  """
  def get_grid(axis):
    grid = _get_pixel_grid(axis, heatmap_width)
    shape = [1, 1, 1, 1]
    shape[axis.value] = -1
    return grid.view(shape)

  keypoint_width = 2.0 * (sigma / heatmap_width) ** 2.0

  edgemaps = []
  for edge in neighbor_link:
    dist = ((keypoints[:, edge[0], 0] - keypoints[:, edge[1], 0]) ** 2 +
            (keypoints[:, edge[0], 1] - keypoints[:, edge[1], 1]) ** 2) ** 0.5
    x_coordinates1 = 0.5 * (keypoints[:, edge[0], np.newaxis, np.newaxis, np.newaxis, 0] +
                            keypoints[:, edge[1], np.newaxis, np.newaxis, np.newaxis, 0])
    y_coordinates1 = 0.5 * (keypoints[:, edge[0], np.newaxis, np.newaxis, np.newaxis, 1] +
                            keypoints[:, edge[1], np.newaxis, np.newaxis, np.newaxis, 1])
    theta = torch.atan((keypoints[:, edge[0], 1] - keypoints[:, edge[1], 1])/\
                       (keypoints[:, edge[0], 0] - keypoints[:, edge[1], 0] + EPSILON))
    cos_theta = torch.cos(-theta)
    sin_theta = torch.sin(-theta)
    R1 = torch.stack([cos_theta, -sin_theta], dim=-1)
    R2 = torch.stack([sin_theta, cos_theta], dim=-1)
    R = torch.stack([R1, R2], dim=1)
    RT = torch.einsum('...ij->...ji', R)
    S = [torch.FloatTensor([[1,0],[0,1/4/(1+dist[i]*10)]]).cuda() for i in range(keypoints.size(0))]
    S = torch.stack(S)
    Sigma = torch.einsum('bij, bjk, bkm->bim', R, S, RT)
    x_vec1 = -Sigma[:, np.newaxis, np.newaxis, np.newaxis, 0, 0]*(get_grid(Axis.x) - x_coordinates1) ** 2 / keypoint_width
    y_vec1 = -Sigma[:, np.newaxis, np.newaxis, np.newaxis, 1, 1]*(get_grid(Axis.y) - y_coordinates1) ** 2 / keypoint_width
    xy_vec1 = -2*Sigma[:, np.newaxis, np.newaxis, np.newaxis, 0, 1]*(get_grid(Axis.x) - x_coordinates1)*(get_grid(Axis.y) - y_coordinates1) / keypoint_width
    maps1 = torch.exp(x_vec1+y_vec1+xy_vec1)
    #maps1_np = maps1.data.cpu().numpy()
    edgemaps.append(maps1)
  edgemaps = torch.stack(edgemaps, dim = 1).squeeze()
  edgemaps = torch.sum(edgemaps, dim=1, keepdim=True)
  edgemaps_max = torch.max(edgemaps.view(edgemaps.size(0), -1), dim=-1)[0]
  edgemaps = edgemaps / edgemaps_max[:, np.newaxis, np.newaxis, np.newaxis]
  #edgemaps_np = edgemaps.data.cpu().numpy()

  return edgemaps

def _get_pixel_grid(axis, width):
  """Returns an array of length `width` containing pixel coordinates."""
  if axis == Axis.x:
    return torch.linspace(-1.0, 1.0, width).cuda()  # Left is negative, right is positive.
  elif axis == Axis.y:
    return torch.linspace(1.0, -1.0, width).cuda()  # Top is positive, bottom is negative.


def change2xy(cfg, keypoints):
  xy = keypoints[:,:,:2].detach().cpu().numpy()
  xy[:,:,0] = np.round((1+xy[:,:,0])*(cfg.img_w-1)/2)
  xy[:,:,1] = np.round((1-xy[:,:,1])*(cfg.img_w-1)/2)
  return np.uint8(xy)

def add_keypoints(image, key_points, radius = 1, miu=None):   # image in [-0.5, 0.5]
  im_key = torch.ones_like(image).cuda()
  if not (miu is None):
    im_key1 = torch.zeros_like(image).cuda()

  for i in range(len(key_points)):
    x = [np.clip(key_points[i][0]-radius, 0, 63), np.clip(key_points[i][0]+radius, 0, 63)]
    y = [np.clip(key_points[i][1] - radius, 0, 63), np.clip(key_points[i][1] + radius, 0, 63)]
    im_key[:, y[0]:y[1] + 1, key_points[i][0]] = 0
    im_key[:, key_points[i][1], x[0]:x[1] + 1] = 0
    if not (miu is None):
      if miu[i] > 0:
        im_key1[:, y[0]:y[1] + 1, key_points[i][0]] = miu[i]
        im_key1[:, key_points[i][1], x[0]:x[1] + 1] = miu[i]

  if not (miu is None):
    image *= im_key
    im_key = 1 - im_key
    image += -0.5 * im_key
    image += im_key1
  else:
    #im_key_np = im_key.data.cpu().numpy()
    image *= im_key
    im_key = 1 - im_key
    image += 0.5 * im_key
  return image

def show_keypoints(image, key_points, img_w, radius = 1):
  im_key = torch.zeros_like(image).cuda()

  for i in range(len(key_points)):
    x = [np.clip(key_points[i][0]-radius, 0, img_w-1), np.clip(key_points[i][0]+radius, 0, img_w-1)]
    y = [np.clip(key_points[i][1] - radius, 0, img_w-1), np.clip(key_points[i][1] + radius, 0, img_w-1)]
    im_key[:, y[0]:y[1] + 1, key_points[i][0]] = 1
    im_key[:, key_points[i][1], x[0]:x[1] + 1] = 1
  return im_key-0.5


