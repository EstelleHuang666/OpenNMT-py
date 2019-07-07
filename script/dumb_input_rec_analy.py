# encoding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

from heatmap import *


words_list = ['by', 'tossing', 'gerrymandering', 'back', 'to', 'congress', 'and', 'the', 'states', ',', 'the', 'supreme', 'court', 'may', 'have', 'emboldened', 'regional', 'lawmakers', 'to', 'carry', 'out', 'partisan', 'mapping', 'after', 'the', 'next', 'census', 'is', 'complete', ',', 'in', 'a', 'move', 'some', 'say', 'will', 'result', 'in', 'noncompetitive', 'elections', '.']


def adjacent_cos_distance(memory_bank):

  memory_bank = torch.squeeze(memory_bank).detach().numpy()
  num_vectors = memory_bank.shape[0]

  adjacent_distances = []
  for vec_i in range(num_vectors - 1):
    cos_tmp_distance = cosine(memory_bank[vec_i, :], memory_bank[vec_i + 1, :])
    adjacent_distances.append(cos_tmp_distance)
  return adjacent_distances


def whole_sentence_cos_distance_mat(memory_bank):

  memory_bank = torch.squeeze(memory_bank).detach().numpy()
  num_vectors = memory_bank.shape[0]

  distance_mat = []
  for vec_i in range(num_vectors):
    distance_tmp_mat = []
    for vec_j in range(num_vectors):
      distance_tmp_mat.append(cosine(memory_bank[vec_i, :], memory_bank[vec_j, :]))
    distance_mat.append(distance_tmp_mat)

  return distance_mat


def same_word_across_cycles_distance(memory_banks):

  words_first_cycle_distances = []
  words_second_cycle_distances = []
  for memory_bank in memory_banks:
    whole_tmp_distances = adjacent_cos_distance(memory_bank)
    words_first_cycle_distances.append(whole_tmp_distances[0])
    words_second_cycle_distances.append(whole_tmp_distances[1])

  x_pos = np.arange(len(words_first_cycle_distances))
  plt.figure(figsize=(30, 3))
  p1 = plt.bar(x_pos, words_first_cycle_distances, align='center', alpha=0.5, width=0.3)
  p2 = plt.bar(x_pos, words_second_cycle_distances, align='center', alpha=0.5, width=0.3)
  plt.legend((p1[0], p2[0]), ('First Cycle', 'Second Cycle'))
  plt.xticks(x_pos, words_list)

  plt.show()


def same_cycle_of_different_words_distance(memory_banks):

  cycle_of_interest = 0
  memory_bank = memory_banks[:, cycle_of_interest, :, :]

  distances_between_adjacent_words = adjacent_cos_distance(memory_bank)

  x_pos = np.arange(len(distances_between_adjacent_words)) + 0.5
  plt.figure(figsize=(40, 3))
  plt.bar(x_pos, distances_between_adjacent_words, align='center', alpha=0.5, width=0.3)
  plt.xticks(x_pos - 0.5, words_list)

  plt.show()

  distance_between_words_mat = whole_sentence_cos_distance_mat(memory_bank)
  distance_between_words_mat = np.asarray(distance_between_words_mat)
  fig, ax = plt.subplots()
  heatmap(distance_between_words_mat, None, None, ax=ax, cmap="YlGn", cbarlabel="cosine distance")
  fig.tight_layout()
  plt.show()


def single_word_trend(memory_banks):

  words_num = memory_banks.shape[0]
  for word_idx in range(words_num):
    memory_bank = memory_banks[word_idx, :, :, :]
    adjacent_tmp_distance = adjacent_cos_distance(memory_bank)

    plt.plot(adjacent_tmp_distance)
    plt.title(words_list[word_idx])

    plt.show()


if __name__ == "__main__":
  memory_banks = torch.load("/tmp/opennmt/rec_mat_0_1562092755677.pt")
  same_cycle_of_different_words_distance(memory_banks)
