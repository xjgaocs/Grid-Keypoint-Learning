import torch
import torch.nn as nn
from torch.autograd import Variable
import ops
import torch.nn.functional as F

%================================================================================
class convLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: int
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(convLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=False)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next    


class convlstm_rnn_p(nn.Module):
    def __init__(self, cfg, map_width, add_dim=0, scale_factor=1, input_dim = None):
        super(convlstm_rnn_p, self).__init__()
        if input_dim is None:
            self.input_dim = cfg.num_keypoints
        else:
            self.input_dim = input_dim
        self.hidden_size = 128//scale_factor
        self.batch_size = cfg.batch_size
        self.keypoint_width = cfg.keypoint_width
        self.map_width = map_width//4
        self.n_layers = 1
        self.convlayer1 = nn.Sequential(nn.Conv2d(self.input_dim,
                                                  32//scale_factor, kernel_size=3,
                                                  stride=1, padding=1),
                          nn.LeakyReLU(0.2, inplace=True))
        self.convlayer2 = nn.Sequential(nn.Conv2d(32//scale_factor,
                                                  64//scale_factor, kernel_size=3,
                                                  stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.convlayer3 = nn.Sequential(nn.Conv2d(64//scale_factor,
                                                  self.hidden_size, kernel_size=3,
                                                  stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.convlstm = nn.ModuleList([convLSTMCell(input_dim=self.hidden_size+add_dim,
                               hidden_dim=self.hidden_size,
                               kernel_size=3) for i in range(self.n_layers)])
        self.hidden = self.init_hidden()
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def init_hidden(self):
        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(self.batch_size, self.hidden_size,
                                          self.map_width, self.map_width).cuda(),
                           torch.zeros(self.batch_size, self.hidden_size,
                                       self.map_width, self.map_width).cuda()))
        return hidden

    def forward(self, gaussian_maps, latent_code=None, update_hidden=True):
        h_in = self.convlayer1(gaussian_maps)
        h_in = self.convlayer2(self.mp(h_in))
        h_in = self.convlayer3(self.mp(h_in))
        if not (latent_code is None):
            h_in = torch.cat((h_in, latent_code), 1)
        for i in range(self.n_layers):
            if update_hidden:
                self.hidden[i] = self.convlstm[i](h_in, self.hidden[i])
                h_in = self.hidden[i][0]
            else:
                hidden = self.convlstm[i](h_in, self.hidden[i])
                h_in = hidden[0]
        return h_in


class convlstm_decoder_p(nn.Module):
    def __init__(self, cfg, add_dim=0, scale_factor=1):
        super(convlstm_decoder_p, self).__init__()
        self.convlayer1 = nn.Sequential(nn.Conv2d(128//scale_factor+add_dim,
                                                  128//scale_factor, kernel_size=3,
                                                  stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.convlayer2 = nn.Sequential(nn.Conv2d(128//scale_factor,
                                                  64//scale_factor, kernel_size=3,
                                                  stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.convlayer3 = nn.Sequential(nn.Conv2d(64//scale_factor,
                                                  32//scale_factor, kernel_size=3,
                                                  stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.adjust_channels_of_output = nn.Sequential(nn.Conv2d(32//scale_factor, cfg.num_keypoints,
                                                               kernel_size=1))
        self.LogSoftmax = nn.LogSoftmax(dim=2)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, rnn_state, latent_code=None):
        if not (latent_code is None):
            rnn_state = torch.cat((rnn_state, latent_code), 1)
        h_out = self.convlayer1(rnn_state)
        h_out = self.convlayer2(self.up(h_out))
        h_out = self.convlayer3(self.up(h_out))
        gaussian_maps = self.adjust_channels_of_output(h_out)
        gaussian_maps_flat = gaussian_maps.view(gaussian_maps.size(0),
                                              gaussian_maps.size(1),
                                              -1)
        gaussian_maps_flat = self.LogSoftmax(gaussian_maps_flat)
        return gaussian_maps_flat    

        
class prior_net_cnn(nn.Module):
  def __init__(self, cfg, scale_factor=1):
      super(prior_net_cnn, self).__init__()
      self.embed = nn.Sequential(nn.Conv2d(128//scale_factor, 128//scale_factor, kernel_size=3,
                                            stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True))
      self.embed1 = nn.Sequential(nn.Conv2d(128//scale_factor, 1, kernel_size=3,
                                            stride=1, padding=1))
      self.embed2 = nn.Sequential(nn.Conv2d(128//scale_factor, 1, kernel_size=3,
                                            stride=1, padding=1),
                                  nn.Softplus())

  def forward(self, rnn_state):
      hidden = self.embed(rnn_state)
      means = self.embed1(hidden)
      stds = self.embed2(hidden) + 1e-4
      return means, stds   
      

class posterior_net_cnn(nn.Module):
  def __init__(self, cfg, scale_factor=1, input_dim=None):
      super(posterior_net_cnn, self).__init__()
      self.num_keypoints = cfg.num_keypoints
      if input_dim is None:
          self.input_dim = cfg.num_keypoints
      else:
          self.input_dim = input_dim
      self.convlayer1 = nn.Sequential(nn.Conv2d(self.input_dim,
                                                16//scale_factor, kernel_size=3,
                                                stride=1, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))
      self.convlayer2 = nn.Sequential(nn.Conv2d(16//scale_factor,
                                                32//scale_factor, kernel_size=3,
                                                stride=1, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))
      self.convlayer3 = nn.Sequential(nn.Conv2d(32//scale_factor,
                                                64//scale_factor, kernel_size=3,
                                                stride=1, padding=1),
                                      nn.LeakyReLU(0.2, inplace=True))
      self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
      self.embed = nn.Sequential(nn.Conv2d(128//scale_factor+64//scale_factor, 128//scale_factor, kernel_size=3,
                                            stride=1, padding=1),
                                        nn.LeakyReLU(0.2, inplace=True))
      self.embed1 = nn.Sequential(nn.Conv2d(128//scale_factor, 1, kernel_size=3,
                                            stride=1, padding=1))
      self.embed2 = nn.Sequential(nn.Conv2d(128//scale_factor, 1, kernel_size=3,
                                            stride=1, padding=1),
                                  nn.Softplus())

  def forward(self, rnn_state, gaussian_maps):
      gaussian_maps = self.convlayer1(gaussian_maps)
      gaussian_maps = self.mp(self.convlayer2(gaussian_maps))
      gaussian_maps = self.mp(self.convlayer3(gaussian_maps))
      hidden = self.embed(torch.cat((rnn_state, gaussian_maps), dim=1))
      means = self.embed1(hidden)
      stds = self.embed2(hidden) + 1e-4
      return means, stds        
      
