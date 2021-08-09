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
r"""Minimal example for training a video_structure model.

See README.md for installation instructions. To run on GPU device 0:

CUDA_VISIBLE_DEVICES=0 python -m video_structure.train
"""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import torch
import torch.optim as optim
import torch.nn as nn
import os, subprocess
import random
from torch.utils.data import DataLoader
import numpy as np

import utils,hyperparameters,losses,vision_BN3_1,ops

'''
Builds the complete model with image encoder plus dynamics model.

  This architecture is meant for testing/illustration only.

  Model architecture:

    image --> keypoints --> reconstructed_image

  The model takes a [batch_size, timesteps, H, W, C] image sequence as input. It
  "observes" all frames, detects keypoints, and reconstructs the images. The
  dynamics model learns to predict future keypoints based on the detected
  keypoints.
'''
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True,
         stdout=subprocess.PIPE).stdout.readlines()]))

cfg = hyperparameters.get_config()
name = 'model=AE_BN3_kth_1_2'

# do not use miu
# increase detect intense
# normalize max map to 1
# simple skip connections
# Adam optimizer
# no training clamp(-0.5, 0.5)
# incite loss
# discrete to 64x64 grid
cfg.dataset = 'kth'
cfg.num_keypoints = 12


load_model = False
log_dir = 'logs/struc'
log_dir = '%s-%s' % (log_dir, name)

os.makedirs('%s/gen/' % log_dir, exist_ok=True)
os.makedirs('%s/plots/' % log_dir, exist_ok=True)

print("Random Seed: ", cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
dtype = torch.cuda.FloatTensor

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss(reduction='sum')

def CalDelta_xy(keypoints_np):
    keypoints_np1 = np.zeros_like(keypoints_np)
    keypoints_np1[:, :, 0] = np.round((1 + keypoints_np[:, :, 0]) * (cfg.img_w-1) / 2)
    keypoints_np1[:, :, 1] = np.round((1 - keypoints_np[:, :, 1]) * (cfg.img_w-1) / 2)
    keypoints_np1[:, :, 0] = keypoints_np1[:, :, 0] / (cfg.img_w-1) * 2 - 1
    keypoints_np1[:, :, 1] = 1 - keypoints_np1[:, :, 1] / (cfg.img_w-1) * 2

    delta_xy = keypoints_np1 - keypoints_np
    return delta_xy

if load_model:
    saved_model = torch.load('%s/model0.pth' % log_dir)
    cfg.learning_rate /= 2
    build_images_to_keypoints_net = saved_model['build_images_to_keypoints_net']
    keypoints_to_images_net = saved_model['keypoints_to_images_net']
else:
    build_images_to_keypoints_net = vision_BN3_1.build_images_to_keypoints_net(
      cfg, [1, cfg.img_h, cfg.img_w])
    keypoints_to_images_net = vision_BN3_1.build_keypoints_to_images_net(
      cfg, [1, cfg.img_h, cfg.img_w])
    build_images_to_keypoints_net.apply(utils.init_weights)
    keypoints_to_images_net.apply(utils.init_weights)

build_images_to_keypoints_net_optimizer = optim.Adam(build_images_to_keypoints_net.parameters(),
                                           lr=cfg.learning_rate, weight_decay=cfg.reg_lambda)
keypoints_to_images_net_optimizer = optim.Adam(keypoints_to_images_net.parameters(),
                                     lr=cfg.learning_rate, weight_decay=cfg.reg_lambda)

# --------- transfer to gpu ------------------------------------
build_images_to_keypoints_net.cuda()
keypoints_to_images_net.cuda()


# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(cfg)

train_loader = DataLoader(train_data,
                          num_workers=4,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=1,
                         batch_size=cfg.test_batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(cfg, dtype, sequence)
            yield batch

training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(cfg, dtype, sequence)
            yield batch

testing_batch_generator = get_testing_batch()

# --------- plotting funtions ------------------------------------
def plot(x, epoch):
    gen_seq = []
    gen_keypointsxy = []
    gt_seq = [x[i] for i in range(len(x))]

    observed_keypoints = []
    for i in range(cfg.n_eval):
        keypoints, _ = build_images_to_keypoints_net(x[i])
        observed_keypoints.append(keypoints.detach())

    for i in range(cfg.n_eval):
        reconstructed_image = keypoints_to_images_net(observed_keypoints[i], x[0], observed_keypoints[0])
        gen_keypointsxy.append(ops.change2xy(observed_keypoints[i]))
        gen_seq.append(reconstructed_image.detach())

    to_plot = []
    gifs = [[] for t in range(cfg.n_eval)]

    nrow = min(cfg.test_batch_size, 10)
    for i in range(nrow):
        # ground truth sequence
        row = []
        for t in range(cfg.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        row = []
        for t in range(cfg.n_eval):
            row.append(ops.add_keypoints(gen_seq[t][i].clone(), gen_keypointsxy[t][i]))
        to_plot.append(row)

        for t in range(cfg.n_eval):
            row = []
            row.append(gt_seq[t][i])
            row.append(gen_seq[t][i])
            gifs[t].append(row)

    fname = '%s/gen/sample_%d.png' % (log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen/sample_%d.gif' % (log_dir, epoch)
    utils.save_gif(fname, gifs)

# --------- training funtions ------------------------------------
def train(x):
    build_images_to_keypoints_net.zero_grad()
    keypoints_to_images_net.zero_grad()

    mse = 0
    observed_keypoints = []
    observed_heatmaps = []
    for i in range(cfg.observed_steps + cfg.predicted_steps):
        keypoints, heatmaps = build_images_to_keypoints_net(x[i])
        keypoints_np = keypoints.data.cpu().numpy()
        delta_xy = CalDelta_xy(keypoints_np)
        delta_xypt = torch.FloatTensor(delta_xy).cuda()
        keypoints = keypoints + delta_xypt
        observed_keypoints.append(keypoints)
        observed_heatmaps.append(heatmaps)

    for i in range(cfg.observed_steps + cfg.predicted_steps):
        reconstructed_image = keypoints_to_images_net(observed_keypoints[i], x[0], observed_keypoints[0])
        mse += 0.5*mse_criterion(reconstructed_image, x[i])

    mse /= (cfg.observed_steps + cfg.predicted_steps)*cfg.batch_size
    separation_loss = losses.temporal_separation_loss(
        cfg, torch.stack(observed_keypoints[:cfg.observed_steps]))
    sparse_loss = losses.sparse_loss(torch.stack(observed_heatmaps), cfg)
    incite = losses.detect_loss(torch.stack(observed_heatmaps))

    loss = mse + incite #+ separation_loss + sparse_loss
    loss.backward()

    torch.nn.utils.clip_grad_norm_(build_images_to_keypoints_net.parameters(), cfg.clipnorm)
    torch.nn.utils.clip_grad_norm_(keypoints_to_images_net.parameters(), cfg.clipnorm)

    build_images_to_keypoints_net_optimizer.step()
    keypoints_to_images_net_optimizer.step()

    return mse.data.cpu().numpy(), separation_loss.data.cpu().numpy(), sparse_loss.data.cpu().numpy(), \
           incite.data.cpu().numpy()

# --------- training loop ------------------------------------
result = open(log_dir+'/'+"result.txt","w")
for epoch in range(cfg.num_epochs):
    build_images_to_keypoints_net.train()
    keypoints_to_images_net.train()
    epoch_mse = 0
    epoch_sep = 0
    epoch_spa = 0
    epoch_inc = 0

    for i in range(cfg.steps_per_epoch):
        x = next(training_batch_generator)  # sequence and the sequence class number

        # train frame_predictor
        mse, separation_loss, sparse_loss, incite = train(x)
        epoch_mse += mse
        epoch_sep += separation_loss
        epoch_spa += sparse_loss
        epoch_inc += incite

    print('[%02d] mse loss: %.5f | separation loss: %.5f | sparse loss: %.5f | incite loss: %.5f (%s)' % (
        epoch, epoch_mse / cfg.steps_per_epoch, epoch_sep / cfg.steps_per_epoch,
        epoch_spa / cfg.steps_per_epoch, epoch_inc / cfg.steps_per_epoch,name))
    result.write('[%02d] mse loss: %.5f | separation loss: %.5f | sparse loss: %.5f | incite loss: %.5f (%s)\n' % (
        epoch, epoch_mse / cfg.steps_per_epoch, epoch_sep / cfg.steps_per_epoch,
        epoch_spa / cfg.steps_per_epoch, epoch_inc / cfg.steps_per_epoch,name))

    # plot some stuff
    build_images_to_keypoints_net.eval()
    keypoints_to_images_net.eval()

    if (epoch+1)%100==0 or epoch==0:
        x = next(testing_batch_generator)
        plot(x, epoch)
    if (epoch + 1) % 500 == 0:
        cfg.learning_rate /= 4
        utils.set_learning_rate(build_images_to_keypoints_net_optimizer, cfg.learning_rate)
        utils.set_learning_rate(keypoints_to_images_net_optimizer, cfg.learning_rate)

    # save the model
    torch.save({
        'build_images_to_keypoints_net': build_images_to_keypoints_net,
        'keypoints_to_images_net': keypoints_to_images_net},
        '%s/model.pth' % log_dir)
result.close()


