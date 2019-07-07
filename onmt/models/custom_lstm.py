# encoding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit as jit
from torch.nn import Parameter
import time

from onmt.analy_tools.distance import adjacent_cos_distance


class LSTMCell(nn.Module):
  def __init__(self, input_size, hidden_size, layer_id):
    super(LSTMCell, self).__init__()

    MODEL_PATH = "/home/murphyhuang/dev/mldata/OpenNMT/Summarization/model/harvard_pretrained_gigaword_nocopy_acc_51.33_ppl_12.74_e20.pt"
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model_params = checkpoint["model"]

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.layer_id = layer_id
    self.weight_ih = Parameter(model_params["encoder.rnn.weight_ih_l{}".format(self.layer_id)])
    self.weight_hh = Parameter(model_params["encoder.rnn.weight_hh_l{}".format(self.layer_id)])
    self.bias_ih = Parameter(model_params["encoder.rnn.bias_ih_l{}".format(self.layer_id)].view(1, 2000))
    self.bias_hh = Parameter(model_params["encoder.rnn.bias_hh_l{}".format(self.layer_id)].view(1, 2000))

  def forward(self, input, state):
    hx, cx = state
    gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
             torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, (hy, cy)


class LSTMLayer(nn.Module):
  def __init__(self, cell, *cell_args):
    super(LSTMLayer, self).__init__()
    self.cell = cell(*cell_args)

  def forward(self, input, state):
    inputs = input.unbind(0)
    outputs = []
    for i in range(len(inputs)):
      out, state = self.cell(inputs[i], state)
      outputs += [out]
    return torch.stack(outputs), state


class LSTMInnerRecLayer(nn.Module):
  def __init__(self, cell, *cell_args):
    super(LSTMInnerRecLayer, self).__init__()
    self.cell = cell(*cell_args)

  def forward(self, input, state):
    forward_time_stamp = int(round(time.time() * 1000))
    inputs = input.unbind(0)
    outputs = []

    words_tmp_outs = []

    for i in range(len(inputs)):
      word_tmp_outs = []

      dumb_input = torch.zeros(inputs[i].size())

      out, state = self.cell(dumb_input, state)
      outputs += [out]

      word_tmp_outs.append(out)
      out_tmp, state_tmp = self.cell(dumb_input, state)
      word_tmp_outs.append(out_tmp)
      for _ in range(100):
        out_tmp, state_tmp = self.cell(dumb_input, state_tmp)
        word_tmp_outs.append(out_tmp)

      word_tmp_outs = torch.stack(word_tmp_outs)
      words_tmp_outs.append(word_tmp_outs)

    words_tmp_outs = torch.stack(words_tmp_outs)
    torch.save(words_tmp_outs, "/tmp/opennmt/rec_mat_{}_{}.pt".format(self.cell.layer_id, forward_time_stamp))

    return torch.stack(outputs), state
