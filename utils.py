import math
import torch
import socket
import argparse
import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from PIL import Image, ImageDraw

from torchvision import datasets, transforms
from torch.autograd import Variable
from torch import nn
import imageio
import random

def get_minibatches_idx(n,
                        minibatch_size,
                        shuffle=False,
                        min_frame=None,
                        trainfiles=None,
                        del_list=None):
  """
  Used to shuffle the dataset at each iteration.
  """
  idx_list = np.arange(n, dtype="int32")

  if min_frame != None:
    if del_list == None:
      del_list = list()
      for i in idx_list:
        vid_path = trainfiles[i].split()[0]
        length = len([f for f in listdir(vid_path) if f.endswith('.png')])
        if length < min_frame:
          del_list.append(i)
      print('[!] Discarded %d samples from training set!' % len(del_list))
    idx_list = np.delete(idx_list, del_list)

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0
  for i in range(n // minibatch_size):
    minibatches.append(
        idx_list[minibatch_start:minibatch_start + minibatch_size])
    minibatch_start += minibatch_size

  if (minibatch_start != n):
    # Make a minibatch out of what is left
    minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches), del_list

def load_dataset(opt):
    if opt.dataset == 'smmnist':
        from data.moving_mnist import MovingMNIST
        train_data = MovingMNIST(
            train=True,
            data_root=opt.data_root,
            seq_len=opt.observed_steps + opt.predicted_steps,
            image_size=opt.img_w,
            deterministic=False,
            num_digits=opt.num_digits)
        test_data = MovingMNIST(
            train=False,
            data_root=opt.data_root,
            seq_len=opt.n_eval,
            image_size=opt.img_w,
            deterministic=False,
            num_digits=opt.num_digits)
    elif opt.dataset == 'bair':
        from data.bair import RobotPush
        train_data = RobotPush(
            data_root=opt.data_root,
            train=True,
            seq_len=opt.observed_steps + opt.predicted_steps,
            image_size=opt.img_w)
        test_data = RobotPush(
            data_root=opt.data_root,
            train=False,
            seq_len=opt.n_eval,
            image_size=opt.img_w)
    elif opt.dataset == 'kth':
        from data.kth import KTH
        train_data = KTH(
            train=True,
            data_root=opt.data_root,
            seq_len=opt.observed_steps + opt.predicted_steps,
            image_size=opt.img_w)
        test_data = KTH(
            train=False,
            data_root=opt.data_root,
            seq_len=opt.n_eval,
            image_size=opt.img_w)
    elif opt.dataset == 'JIGSAWS-Suturing':
        from data.Suturning import Suturning
        train_data = Suturning(
            train=True,
            data_root=opt.data_root,
            seq_len=opt.observed_steps + opt.predicted_steps,
            image_size=opt.img_w)
        test_data = Suturning(
            train=False,
            data_root=opt.data_root,
            seq_len=opt.n_eval,
            image_size=opt.img_w)
    elif opt.dataset == 'Human3.6m':
        from data.human import Human
        train_data = Human(
            train=True,
            data_root=opt.data_root,
            seq_len=opt.observed_steps + opt.predicted_steps,
            image_size=opt.img_w)
        test_data = Human(
            train=False,
            data_root=opt.data_root,
            seq_len=opt.n_eval,
            image_size=opt.img_w)
    elif opt.dataset == 'Human3.6m_hr':
        from data.human_hr import Human
        train_data = Human(
            train=True,
            data_root=opt.data_root,
            seq_len=opt.observed_steps + opt.predicted_steps,
            image_size=opt.img_w)
        test_data = Human(
            train=False,
            data_root=opt.data_root,
            seq_len=opt.n_eval,
            image_size=opt.img_w)
    elif opt.dataset == 'kitti':
        from data.kitti import Kitti
        train_data = Kitti(
            train=True,
            data_root=opt.data_root,
            seq_len=opt.observed_steps + opt.predicted_steps,
            image_size=opt.img_w)
        test_data = Kitti(
            train=False,
            data_root=opt.data_root,
            seq_len=opt.n_eval,
            image_size=opt.img_w)
    elif opt.dataset == 'penn':
        from data.penn import Penn
        train_data = Penn(
            train=True,
            data_root=opt.data_root,
            seq_len=opt.observed_steps + opt.predicted_steps,
            image_size=opt.img_w)
        test_data = Penn(
            train=False,
            data_root=opt.data_root,
            seq_len=opt.n_eval,
            image_size=opt.img_w)

    return train_data, test_data

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def change_to_video(sequence):
    sequence.transpose_(0, 1).transpose_(1, 2)
    return sequence

def normalize_data(opt, dtype, sequence):

    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)

    #sequence.transpose_(0, 1)

    return sequence_input(sequence, dtype)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if not (m.bias is None):
            m.bias.data.fill_(0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight)
        if not (m.bias is None):
            m.bias.data.fill_(0)

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def make_image(tensor, colored_kp=False, key_points=None):
    tensor = tensor.cpu().clamp(-0.5, 0.5) + 0.5
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    tensor = tensor.transpose(0, 1).transpose(1, 2).numpy() * 255

    if colored_kp:
        radius = 1
        color_bar = np.array([[248,248,24],[246,221,41],[254,191,60],
                              [213,190,39],[148,202,73],[83,204,125],
                              [41,195,170],[1,183,202],[33,164,227],
                              [45,142,242],[58,115,255],[71,86,247]])
        for i in range(len(key_points)):
            x = [np.clip(key_points[i][0] - radius, 0, 63), np.clip(key_points[i][0] + radius, 0, 63)]
            y = [np.clip(key_points[i][1] - radius, 0, 63), np.clip(key_points[i][1] + radius, 0, 63)]
            tensor[y[0]:y[1] + 1, key_points[i][0]] = color_bar[i]
            tensor[key_points[i][1], x[0]:x[1] + 1] = color_bar[i]
    # pdb.set_trace()
    return Image.fromarray(np.uint8(tensor))

def save_image(filename, tensor, colored_kp=False, key_points=None):
    img = make_image(tensor, colored_kp, key_points)
    img.save(filename)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def save_gif(filename, inputs, duration=0.15, colored_kp=False, key_points=None):
    images = []
    m = 0
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(-0.5, 0.5)
        img = (img.numpy()+0.5) * 255
        if colored_kp:
            img += 255
            radius = 1
            kp = key_points[m]
            color_bar = np.array([[248, 248, 24], [246, 221, 41], [254, 191, 60],
                                  [213, 190, 39], [148, 202, 73], [83, 204, 125],
                                  [41, 195, 170], [1, 183, 202], [33, 164, 227],
                                  [45, 142, 242], [58, 115, 255], [71, 86, 247]])
            for i in range(len(kp)):
                x = [np.clip(kp[i][0] - radius, 0, 63), np.clip(kp[i][0] + radius, 0, 63)]
                y = [np.clip(kp[i][1] - radius, 0, 63), np.clip(kp[i][1] + radius, 0, 63)]
                img[y[0]:y[1] + 1, kp[i][0]] = color_bar[i]
                img[kp[i][1], x[0]:x[1] + 1] = color_bar[i]
            m+=1
        images.append(img.astype('uint8'))
    imageio.mimsave(filename, images, duration=duration)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif_with_text(filename, inputs, text, duration=0.25):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1).numpy()*255
        images.append(img.astype('uint8'))
    imageio.mimsave(filename, images, duration=duration)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w] = image

    return img


def transform(input_):
    return 2 * input_ - 1.


def inverse_transform(input_):
    return (input_ + 1.) / 2.


def imsave(images, size, path):
    return imageio.imwrite(path, merge(images, size))

def gauss2D_mask(center, shape, sigma=0.5):
  m, n = [ss - 1 for ss in shape]
  y, x = np.ogrid[0:m + 1, 0:n + 1]
  y = y - center[0]
  x = x - center[1]
  z = x * x + y * y
  h = np.exp(-z / (2. * sigma * sigma/(shape[0]**2)))
  sumh = h.sum()
  if sumh != 0:
    h = h / sumh
  return h

def visualize_lm(posex, posey, image_size, num_keypoints):
    posey = inverse_transform(posey) * image_size
    posex = inverse_transform(posex) * image_size
    cpose = np.zeros((image_size, image_size, num_keypoints))
    for j in range(num_keypoints):
        gmask = gauss2D_mask(
            (posey[j], posex[j]), (image_size, image_size), sigma=8.)
        cpose[:, :, j] = gmask / gmask.max()

    return np.amax(cpose, axis=2)

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

def l1_metric(x1, x2):
    err = np.sum(abs(x1 - x2))
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    l1 = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]): # calculate for each channel respectively
                ssim[i, t] += ssim_metric(gt[t][i][c], pred[t][i][c], data_range=1.0)
                psnr[i, t] += psnr_metric(gt[t][i][c], pred[t][i][c], data_range=1.0)
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            l1[i, t] = l1_metric(gt[t][i], pred[t][i])

    return l1, ssim, psnr

def eval_seq1(gt, pred, LPIPSmodel):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    l1 = np.zeros((bs, T))
    lpips = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]): # calculate for each channel respectively
                ssim[i, t] += ssim_metric(gt[t][i][c], pred[t][i][c], data_range=1.0)
                psnr[i, t] += psnr_metric(gt[t][i][c], pred[t][i][c], data_range=1.0)
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            l1[i, t] = l1_metric(gt[t][i], pred[t][i])
    for t in range(T):
        gt_im = 2*gt[t]-1 # normalize to [-1, 1]
        pred_im = 2*pred[t]-1
        if gt_im.shape[1]==1:
            gt_im = np.tile(gt_im, (1,3,1,1))
            pred_im = np.tile(pred_im, (1,3,1,1))
        gt_im = torch.FloatTensor(gt_im).cuda()
        pred_im = torch.FloatTensor(pred_im).cuda()
        lpips[:, t] = LPIPSmodel.forward(gt_im, pred_im).squeeze().data.cpu().numpy()
    return l1, ssim, psnr, lpips

def changed_keypoints(keypoints, weight):
    keypoints = keypoints* weight
    return keypoints

def changed_keypoints1(keypoints, weight):
    keypoints = torch.clamp(0.95*keypoints + 0.05*weight, min=0, max=1.0)
    return keypoints