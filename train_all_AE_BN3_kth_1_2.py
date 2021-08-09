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
from torch.autograd import Variable
import numpy as np
import utils,hyperparameters,losses,ops,dynanmics
import time
import imageio, cv2
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
# convlstm
# pi loss
# maxpooling map
# auto-regressive model

cfg.dataset = 'kth'
cfg.observed_steps = 10
cfg.predicted_steps = 10
cfg.num_keypoints = 12
cfg.batch_size = 32
cfg.num_epochs = 1500
cfg.learning_rate = 1e-3
cfg.kl_loss_scale = 0.05 #----------------------------
cfg.test_N = 256
cfg.test_batch_size = 32
cfg.nsample = 4
cfg.reso = cfg.img_w

load_model = False
log_dir = 'logs/struc'
log_dir = '%s-%s' % (log_dir, name)

os.makedirs('%s/gen1/' % log_dir, exist_ok=True)

print("Random Seed: ", cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)
dtype = torch.cuda.FloatTensor

# --------- loss functions ------------------------------------
def CalDelta_xy(keypoints_np, width=64):
    keypoints_np1 = np.zeros_like(keypoints_np)
    keypoints_np1[:, :, 0] = np.round((1 + keypoints_np[:, :, 0]) * (width-1) / 2)
    keypoints_np1[:, :, 1] = np.round((1 - keypoints_np[:, :, 1]) * (width-1) / 2)
    keypoints_np1[:, :, 0] = keypoints_np1[:, :, 0] / (width-1) * 2 - 1
    keypoints_np1[:, :, 1] = 1 - keypoints_np1[:, :, 1] / (width-1) * 2

    delta_xy = keypoints_np1 - keypoints_np
    return delta_xy

mse_criterion = nn.MSELoss(reduction='sum')
mse_criterion_none = nn.MSELoss(reduction='none')
def kl_criterion(mu1, stds1, mu2, stds2):
    # KL( N(mu_1, sigma2_1) || N(mu_2, sigma2_2)) =
    #   log( sqrt(
    #
    kld = torch.log(stds2/stds1) + (stds1**2 + (mu1 - mu2)**2)/(2*stds2**2) - 1/2
    return kld.sum()


repeat=0
saved_model = torch.load('%s/model.pth' % log_dir)
build_images_to_keypoints_net = saved_model['build_images_to_keypoints_net']
keypoints_to_images_net =  saved_model['keypoints_to_images_net']
rnn_cell = dynanmics.convlstm_rnn_p(cfg, map_width=cfg.img_w, add_dim=1)
keypoint_decoder = dynanmics.convlstm_decoder_p(cfg, add_dim=1)
prior_net = dynanmics.prior_net_cnn(cfg)
posterior_net = dynanmics.posterior_net_cnn(cfg)

rnn_cell.apply(utils.init_weights)
keypoint_decoder.apply(utils.init_weights)
prior_net.apply(utils.init_weights)
posterior_net.apply(utils.init_weights)


rnn_cell_optimizer = optim.Adam(rnn_cell.parameters(),
                                     lr=cfg.learning_rate, weight_decay=cfg.reg_lambda)
keypoint_decoder_optimizer = optim.Adam(keypoint_decoder.parameters(),
                                     lr=cfg.learning_rate, weight_decay=cfg.reg_lambda)
prior_net_optimizer = optim.Adam(prior_net.parameters(),
                                     lr=cfg.learning_rate, weight_decay=cfg.reg_lambda)
posterior_net_optimizer = optim.Adam(posterior_net.parameters(),
                                     lr=cfg.learning_rate, weight_decay=cfg.reg_lambda)

# --------- transfer to gpu ------------------------------------
build_images_to_keypoints_net.cuda()
keypoints_to_images_net.cuda()
rnn_cell.cuda()
keypoint_decoder.cuda()
prior_net.cuda()
posterior_net.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(cfg)

train_loader = DataLoader(train_data,
                          num_workers=4,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=4,
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
    gen_keypoints = []
    gen_seq = []
    gt_seq = [x[i] for i in range(len(x))]

    observed_keypoints = []
    for i in range(cfg.observed_steps):
        keypoints, _ = build_images_to_keypoints_net(x[i])
        keypoints_np = keypoints.data.cpu().numpy()
        delta_xy = CalDelta_xy(keypoints_np, width=cfg.img_w)
        delta_xypt = torch.FloatTensor(delta_xy).cuda()
        keypoints = keypoints + delta_xypt
        observed_keypoints.append(keypoints.detach())

    rnn_cell.batch_size = cfg.test_batch_size
    rnn_cell.hidden = rnn_cell.init_hidden()
    rnn_cell.batch_size = cfg.batch_size
    rnn_state = rnn_cell.hidden[0][0]

    for i in range(cfg.n_eval):
        if i < cfg.observed_steps:
            observed_keypoints_np = observed_keypoints[i].data.cpu().numpy()
            observed_keypoints_np[:, :, 0] = np.round((1 + observed_keypoints_np[:, :, 0])
                                                      * (cfg.img_w - 1) / 2)
            observed_keypoints_np[:, :, 1] = np.round((1 - observed_keypoints_np[:, :, 1])
                                                      * (cfg.img_w - 1) / 2)
            observed_keypoints_np = np.clip(observed_keypoints_np, 0, cfg.img_w - 1)
            observed_keypoints_id = observed_keypoints_np[:, :, 0] + \
                                    observed_keypoints_np[:, :, 1] * cfg.img_w
            best_keypoints_id_flat = observed_keypoints_id.flatten().astype(int)
            observed_keypoints_map_np = np.zeros((cfg.test_batch_size, cfg.num_keypoints,
                                                  cfg.img_w, cfg.img_w))
            observed_keypoints_map_np_flat = observed_keypoints_map_np.reshape(cfg.test_batch_size *
                                                                               cfg.num_keypoints, -1)
            observed_keypoints_map_np_flat[range(cfg.test_batch_size * cfg.num_keypoints),
                                           best_keypoints_id_flat[range(cfg.test_batch_size
                                                                        * cfg.num_keypoints)]] = 1
            observed_keypoints_map_np = observed_keypoints_map_np_flat.reshape(cfg.test_batch_size,
                                                                               cfg.num_keypoints,
                                                                               cfg.img_w, -1)
            observed_keypoints_map_batch = torch.FloatTensor(observed_keypoints_map_np).cuda()
            observed_keypoints_batch = observed_keypoints[i]
            mean, std = posterior_net(rnn_state, observed_keypoints_map_batch)
            if i == 0:
                observed_keypoints_batch0 = observed_keypoints_batch
        else:
            mean_prior, std_prior = prior_net(rnn_state)
            mean = mean_prior.detach()
            std = std_prior.detach()
        eps = Variable(std.data.new(std.size()).normal_())
        eps = eps * std + mean
        if i < cfg.observed_steps:
            keypoints = observed_keypoints_batch
        else:
            sampled_keypoints_flat = keypoint_decoder(rnn_state, eps).detach()
            sampled_keypoints_flat = torch.exp(sampled_keypoints_flat).data.cpu().numpy()
            sampled_keypoints_id = np.argmax(sampled_keypoints_flat, axis=-1)
            keypoints = np.zeros((cfg.test_batch_size, cfg.num_keypoints, 2))
            keypoints[:, :, 0] = sampled_keypoints_id % cfg.img_w / (cfg.img_w - 1) * 2 + (-1)
            keypoints[:, :, 1] = 1 - sampled_keypoints_id // cfg.img_w / (cfg.img_w - 1) * 2
            keypoints = torch.FloatTensor(keypoints).cuda()
            observed_keypoints_map_np = np.zeros((cfg.test_batch_size, cfg.num_keypoints,
                                                  cfg.img_w, cfg.img_w))
            observed_keypoints_map_np_flat = observed_keypoints_map_np.reshape(cfg.test_batch_size *
                                                                               cfg.num_keypoints, -1)
            sampled_keypoints_id_flat = sampled_keypoints_id.flatten().astype(int)
            observed_keypoints_map_np_flat[range(cfg.test_batch_size * cfg.num_keypoints),
                                           sampled_keypoints_id_flat[range(cfg.test_batch_size
                                                                        * cfg.num_keypoints)]] = 1
            observed_keypoints_map_np = observed_keypoints_map_np_flat.reshape(cfg.test_batch_size,
                                                                               cfg.num_keypoints,
                                                                               cfg.img_w, -1)
            observed_keypoints_map_batch = torch.FloatTensor(observed_keypoints_map_np).cuda()
        reconstructed_image = keypoints_to_images_net(keypoints, x[0], observed_keypoints_batch0).detach()
        rnn_state = rnn_cell(observed_keypoints_map_batch, eps)
        gen_keypoints.append(ops.change2xy(keypoints))
        gen_seq.append(reconstructed_image)

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
            row.append(ops.add_keypoints(gen_seq[t][i].clone(), gen_keypoints[t][i]))
        to_plot.append(row)

        for t in range(cfg.n_eval):
            row = []
            row.append(gt_seq[t][i])
            row.append(gen_seq[t][i])
            gifs[t].append(row)

    fname = '%s/gen1/sample_%d.png' % (log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

    fname = '%s/gen1/sample_%d.gif' % (log_dir, epoch)
    utils.save_gif(fname, gifs)

def val(x, epoch):
    observed_keypoints = []
    for i in range(cfg.observed_steps):
        keypoints, _ = build_images_to_keypoints_net(x[i])
        keypoints_np = keypoints.data.cpu().numpy()
        delta_xy = CalDelta_xy(keypoints_np, width=cfg.img_w)
        delta_xypt = torch.FloatTensor(delta_xy).cuda()
        keypoints = keypoints + delta_xypt
        observed_keypoints.append(keypoints.detach())

    ssim = np.zeros((cfg.test_batch_size, cfg.nsample, cfg.n_eval))
    psnr = np.zeros((cfg.test_batch_size, cfg.nsample, cfg.n_eval))
    all_gen = []
    all_gen_keypoints = []
    for s in range(cfg.nsample):
        gen_seq = []
        gt_seq = []
        all_gen.append([])
        all_gen_keypoints.append([])
        rnn_cell.batch_size = cfg.test_batch_size
        rnn_cell.hidden = rnn_cell.init_hidden()
        rnn_cell.batch_size = cfg.batch_size
        rnn_state = rnn_cell.hidden[0][0]
        for i in range(cfg.n_eval):
            if i < cfg.observed_steps:
                observed_keypoints_np = observed_keypoints[i].data.cpu().numpy()
                observed_keypoints_np[:, :, 0] = np.round((1 + observed_keypoints_np[:, :, 0])
                                                          * (cfg.reso - 1) / 2)
                observed_keypoints_np[:, :, 1] = np.round((1 - observed_keypoints_np[:, :, 1])
                                                          * (cfg.reso - 1) / 2)
                observed_keypoints_np = np.clip(observed_keypoints_np, 0, cfg.reso - 1)
                observed_keypoints_id = observed_keypoints_np[:, :, 0] + \
                                        observed_keypoints_np[:, :, 1] * cfg.reso
                best_keypoints_id_flat = observed_keypoints_id.flatten().astype(int)
                observed_keypoints_map_np = np.zeros((cfg.test_batch_size, cfg.num_keypoints,
                                                      cfg.reso, cfg.reso))
                observed_keypoints_map_np_flat = observed_keypoints_map_np.reshape(cfg.test_batch_size *
                                                                                   cfg.num_keypoints, -1)
                observed_keypoints_map_np_flat[range(cfg.test_batch_size * cfg.num_keypoints),
                                               best_keypoints_id_flat[range(cfg.test_batch_size
                                                                            * cfg.num_keypoints)]] = 1
                observed_keypoints_map_np = observed_keypoints_map_np_flat.reshape(cfg.test_batch_size,
                                                                                   cfg.num_keypoints,
                                                                                   cfg.reso, -1)
                observed_keypoints_map_batch = torch.FloatTensor(observed_keypoints_map_np).cuda()
                observed_keypoints_batch = observed_keypoints[i]
                mean, std = posterior_net(rnn_state, observed_keypoints_map_batch)
            else:
                mean_prior, std_prior = prior_net(rnn_state)
                mean = mean_prior.detach()
                std = std_prior.detach()
            eps = Variable(std.data.new(std.size()).normal_())
            eps = eps * std + mean
            if i < cfg.observed_steps:
                keypoints = observed_keypoints_batch
            else:
                sampled_keypoints_flat = keypoint_decoder(rnn_state, eps).detach()
                sampled_keypoints_flat = torch.exp(sampled_keypoints_flat).data.cpu().numpy()
                sampled_keypoints_id = np.argmax(sampled_keypoints_flat, axis=-1)
                keypoints = np.zeros((cfg.test_batch_size, cfg.num_keypoints, 2))
                keypoints[:, :, 0] = sampled_keypoints_id % cfg.reso / (cfg.reso - 1) * 2 + (-1)
                keypoints[:, :, 1] = 1 - sampled_keypoints_id // cfg.reso / (cfg.reso - 1) * 2
                keypoints = torch.FloatTensor(keypoints).cuda()
                observed_keypoints_map_np = np.zeros((cfg.test_batch_size, cfg.num_keypoints,
                                                      cfg.reso, cfg.reso))
                observed_keypoints_map_np_flat = observed_keypoints_map_np.reshape(cfg.test_batch_size *
                                                                                   cfg.num_keypoints, -1)
                sampled_keypoints_id_flat = sampled_keypoints_id.flatten().astype(int)
                observed_keypoints_map_np_flat[range(cfg.test_batch_size * cfg.num_keypoints),
                                               sampled_keypoints_id_flat[range(cfg.test_batch_size
                                                                               * cfg.num_keypoints)]] = 1
                observed_keypoints_map_np = observed_keypoints_map_np_flat.reshape(cfg.test_batch_size,
                                                                                   cfg.num_keypoints,
                                                                                   cfg.reso, -1)
                observed_keypoints_map_batch = torch.FloatTensor(observed_keypoints_map_np).cuda()
            rnn_state = rnn_cell(observed_keypoints_map_batch, eps)
            reconstructed_image = keypoints_to_images_net(keypoints, x[cfg.observed_steps - 1],
                                                          observed_keypoints[cfg.observed_steps - 1]).detach()
            reconstructed_image = torch.clamp(reconstructed_image, -0.5, 0.5)

            all_gen_keypoints[s].append(ops.change2xy(keypoints))
            all_gen[s].append(reconstructed_image)
            gen_seq.append(reconstructed_image.data.cpu().numpy() + 0.5)
            gt_seq.append(x[i].data.cpu().numpy() + 0.5)
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    return ssim, psnr

# --------- training funtions ------------------------------------
def train(x, train_step):
    rnn_cell.zero_grad()
    keypoint_decoder.zero_grad()
    prior_net.zero_grad()
    posterior_net.zero_grad()

    kl_divergence = torch.FloatTensor([0]).cuda()
    mse = torch.FloatTensor([0]).cuda()
    keypoints_pi = torch.FloatTensor([0]).cuda()
    observed_keypoints = []
    for i in range(cfg.observed_steps + cfg.predicted_steps):
        keypoints, _ = build_images_to_keypoints_net(x[i])
        keypoints_np = keypoints.data.cpu().numpy()
        delta_xy = CalDelta_xy(keypoints_np, width=cfg.img_w)
        delta_xypt = torch.FloatTensor(delta_xy).cuda()
        keypoints = keypoints + delta_xypt
        observed_keypoints.append(keypoints.detach())
    #'''
    rnn_cell.hidden = rnn_cell.init_hidden()
    rnn_state = rnn_cell.hidden[0][0]

    for i in range(cfg.observed_steps + cfg.predicted_steps):
        observed_keypoints_map_np = np.zeros((cfg.batch_size, cfg.num_keypoints,
                                              cfg.img_w, cfg.img_w))
        observed_keypoints_np = observed_keypoints[i].data.cpu().numpy()
        observed_keypoints_np[:, :, 0] = np.round((1 + observed_keypoints_np[:, :, 0])
                                                  * (cfg.img_w - 1) / 2)
        observed_keypoints_np[:, :, 1] = np.round((1 - observed_keypoints_np[:, :, 1])
                                                  * (cfg.img_w - 1) / 2)
        observed_keypoints_np = np.clip(observed_keypoints_np, 0, cfg.img_w - 1)
        eye = np.eye(cfg.img_w ** 2)
        observed_keypoints_id = observed_keypoints_np[:, :, 0] + \
                            observed_keypoints_np[:, :, 1] * cfg.img_w
        best_keypoints_id_flat = observed_keypoints_id.flatten().astype(int)
        probs = eye[best_keypoints_id_flat]
        probs = probs.reshape(cfg.batch_size, cfg.num_keypoints, -1)
        probs = torch.FloatTensor(probs).cuda()
        observed_keypoints_map_np_flat = observed_keypoints_map_np.reshape(cfg.batch_size*cfg.num_keypoints, -1)
        observed_keypoints_map_np_flat[range(cfg.batch_size*cfg.num_keypoints),
                                       best_keypoints_id_flat[range(cfg.batch_size*cfg.num_keypoints)]
                                 ] = 1
        observed_keypoints_map_np = observed_keypoints_map_np_flat.reshape(cfg.batch_size, cfg.num_keypoints,
                                                                        cfg.img_w, -1)
        observed_keypoints_map_batch = torch.FloatTensor(observed_keypoints_map_np).cuda()
        mean_prior, std_prior = prior_net(rnn_state)
        mean, std = posterior_net(rnn_state, observed_keypoints_map_batch)
        if i>0:
            kl_divergence += kl_criterion(mean_prior, std_prior, mean, std)

        # Conduct BestOfMany
        sampled_latent_list = []
        sample_losses = []
        for j in range(cfg.num_samples_for_bom):
            eps = Variable(std.data.new(std.size()).normal_())
            eps = eps * std + mean
            sampled_latent_list.append(eps)
            sampled_keypoints_flat = keypoint_decoder(rnn_state, eps).detach()
            sample_losses.append(torch.sum(-probs * sampled_keypoints_flat, dim=(1,2)).detach())
        _, best_sample_ind = torch.min(torch.stack(sample_losses), dim=0)
        best_sample_ind = best_sample_ind.detach().cpu().numpy()
        best_latent = torch.stack([sampled_latent_list[best_sample_ind[j]][j]
                                       for j in range(cfg.batch_size)])
        best_keypoints_flat = keypoint_decoder(rnn_state, best_latent)
        keypoints_pi += torch.sum(-probs * best_keypoints_flat)
        rnn_state = rnn_cell(observed_keypoints_map_batch, best_latent)

    #'''
    keypoints_pi /= (cfg.observed_steps+cfg.predicted_steps)*cfg.num_keypoints*cfg.batch_size
    kl_divergence /= (cfg.observed_steps+cfg.predicted_steps-1)*cfg.batch_size

    loss = keypoints_pi + cfg.kl_loss_scale*kl_divergence
    loss.backward()

    torch.nn.utils.clip_grad_norm_(rnn_cell.parameters(), cfg.clipnorm)
    torch.nn.utils.clip_grad_norm_(keypoint_decoder.parameters(), cfg.clipnorm)
    torch.nn.utils.clip_grad_norm_(prior_net.parameters(), cfg.clipnorm)
    torch.nn.utils.clip_grad_norm_(posterior_net.parameters(), cfg.clipnorm)

    rnn_cell_optimizer.step()
    keypoint_decoder_optimizer.step()
    prior_net_optimizer.step()
    posterior_net_optimizer.step()

    return kl_divergence.data.cpu().numpy(), mse.data.cpu().numpy(), \
           keypoints_pi.data.cpu().numpy()


# --------- training loop ------------------------------------
train_step = 0 + repeat*cfg.num_epochs*cfg.steps_per_epoch
for epoch in range(cfg.num_epochs):
    build_images_to_keypoints_net.eval()
    keypoints_to_images_net.eval()
    rnn_cell.train()
    keypoint_decoder.train()
    prior_net.train()
    posterior_net.train()

    epoch_kl = 0
    epoch_mse = 0
    epoch_keypoints_pi = 0

    #cfg.steps_per_epoch = 1
    for i in range(cfg.steps_per_epoch):
        x = next(training_batch_generator)  # sequence and the sequence class number

        # train frame_predictor
        kl, mse, keypoints_pi = train(x, train_step)
        train_step += 1
        epoch_kl += kl
        epoch_mse += mse
        epoch_keypoints_pi += keypoints_pi

    print('[%02d] kl: %.5f | mse: %.5f | future_key_pi: %.5f (%s_all)' % (
        epoch, epoch_kl / cfg.steps_per_epoch,
        epoch_mse / cfg.steps_per_epoch,
        epoch_keypoints_pi / cfg.steps_per_epoch, name))
    with open(log_dir+'/'+"result_all_best.txt","a") as result:
        if epoch==0:
            result.write('\n')
        result.write('[%02d] kl: %.5f | mse: %.5f | future_key_pi: %.5f (%s_all)\n' % (
        epoch, epoch_kl / cfg.steps_per_epoch,
        epoch_mse / cfg.steps_per_epoch,
        epoch_keypoints_pi / cfg.steps_per_epoch, name))

    # plot some stuff
    build_images_to_keypoints_net.eval()
    keypoints_to_images_net.eval()
    rnn_cell.eval()
    keypoint_decoder.eval()
    prior_net.eval()
    posterior_net.eval()

    if (epoch+1)%100==0 or epoch==0:
        x = next(testing_batch_generator)
        plot(x, epoch)
        psnr_total = np.zeros((cfg.test_N, cfg.n_eval))
        ssim_total = np.zeros((cfg.test_N, cfg.n_eval))
        for i in range(0, cfg.test_N, cfg.test_batch_size):
            # plot test
            test_x = next(testing_batch_generator)
            ssim, psnr = val(test_x, epoch)
            for j in range(0, cfg.test_batch_size):
                psnr_total[i + j, :] = psnr[j, np.argmax(np.mean(psnr[j], axis=1)), :]
                ssim_total[i + j, :] = ssim[j, np.argmax(np.mean(ssim[j], axis=1)), :]

        ssim_val = int("{:3.0f}".format(np.mean(ssim_total[:, cfg.observed_steps:]) * 1000))
        psnr_val = int("{:4.0f}".format(np.mean(psnr_total[:, cfg.observed_steps:]) * 100))
        # save the model
        torch.save({
                'build_images_to_keypoints_net': build_images_to_keypoints_net,
                'keypoints_to_images_net': keypoints_to_images_net,
                'rnn_cell': rnn_cell,
                'keypoint_decoder': keypoint_decoder,
                'prior_net': prior_net,
                'posterior_net': posterior_net},
                '%s/model_all_epoch%d_kl%.2f_ssim%d_psnr_%d.pth' % (log_dir, epoch, cfg.kl_loss_scale, ssim_val, psnr_val))


    if (epoch + 1) % 500 == 0:
        cfg.learning_rate /= 4
        utils.set_learning_rate(rnn_cell_optimizer, cfg.learning_rate)
        utils.set_learning_rate(keypoint_decoder_optimizer, cfg.learning_rate)
        utils.set_learning_rate(prior_net_optimizer, cfg.learning_rate)
        utils.set_learning_rate(posterior_net_optimizer, cfg.learning_rate)




