# encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from scipy.spatial.distance import cosine

from matplotlib import pyplot as plt


def adjacent_cos_distance(memory_bank):

  memory_bank = torch.squeeze(memory_bank).detach().numpy()
  num_vectors = memory_bank.shape[0]

  adjacent_distances = []
  for vec_i in range(num_vectors - 1):
    cos_tmp_distance = cosine(memory_bank[vec_i, :], memory_bank[vec_i + 1, :])
    adjacent_distances.append(cos_tmp_distance)

  plt.plot(adjacent_distances)
  plt.show()
