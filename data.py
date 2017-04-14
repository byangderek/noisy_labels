from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
install_aliases()

import numpy as np
import numpy.random as npr
import os
import gzip
import struct
import array

from urllib.request import urlretrieve

npr.seed(23)

# noise type:
# 0: no noise
# 1: white noise
# 2: label permutation
class mnist:
    def __init__(self, set='train', noise_type=0, noise_ratio=0.5, noise_prior=0.1, gt_prior=False, use_init=False, is_train=False):
        base_url = 'http://yann.lecun.com/exdb/mnist/'

        def parse_labels(filename):
            with gzip.open(filename, 'rb') as fh:
                magic, num_data = struct.unpack(">II", fh.read(8))
                return np.array(array.array("B", fh.read()), dtype=np.uint8)

        def parse_images(filename):
            with gzip.open(filename, 'rb') as fh:
                magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
                return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

        def download(url, filename):
            if not os.path.exists('data'):
                os.makedirs('data')
            out_file = os.path.join('data', filename)
            if not os.path.isfile(out_file):
                urlretrieve(url, out_file)

        for filename in ['train-images-idx3-ubyte.gz',
                         'train-labels-idx1-ubyte.gz',
                         't10k-images-idx3-ubyte.gz',
                         't10k-labels-idx1-ubyte.gz']:
            download(base_url + filename, filename)

        self._set = set
        validation_size = 5000
        train_images = parse_images('data/train-images-idx3-ubyte.gz')
        train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
        test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
        test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

        if set == 'train':
            self.data = train_images[validation_size:]
            self.labels = train_labels[validation_size:].astype(np.float32)
        elif set == 'val':
            self.data = train_images[:validation_size]
            self.labels = train_labels[:validation_size].astype(np.float32)
        else:
            self.data = test_images
            self.labels = test_labels.astype(np.float32)

        self._is_train = is_train
        self._noise_type = noise_type
        self._noise_ratio = noise_ratio
        self.add_noise()
        self.compute_noise_matrix(noise_prior, gt_prior, use_init)
        self.shuffle_inds()


    def size(self):
        return self.data.shape[0]


    def compute_noise_matrix(self, noise_prior, gt_prior, use_init):
        if use_init:
            m = np.load('weights/lr_m_%.1f.npy' % self._noise_ratio)
            self._noise_matrix = m
        elif self._noise_type == 0:
            self._noise_matrix = np.eye(10)
        elif self._noise_type == 2 and gt_prior:
            perm = np.array([7, 9, 0, 4, 2, 1, 3, 5, 6, 8])
            r = self._noise_ratio
            m = np.zeros((10, 10), dtype=np.float32)
            for i in xrange(10):
                m[i, i] = 1 - r
                m[i, perm[i]] = r
            self._noise_matrix = m
        else:
            if gt_prior:
                r = self._noise_ratio
            else:
                r = noise_prior
            r0 = r * 0.1
            r1 = 1 - r + r * 0.1
            m = r0 * np.ones((10, 10), dtype=np.float32)
            for i in xrange(10):
                m[i, i] = r1
            self._noise_matrix = m


    def add_noise(self):
        if self._noise_type == 1:
            prob = npr.rand(self.size())
            noise = npr.rand(self.size())
            mask = np.asarray(prob < self._noise_ratio)
            noise_labels = np.asarray(np.floor(noise/0.1))
            self.labels[mask] = noise_labels[mask]
        elif self._noise_type == 2:
            perm = np.array([7, 9, 0, 4, 2, 1, 3, 5, 6, 8])
            noise = perm[self.labels.astype(int)]
            from sklearn.model_selection import StratifiedShuffleSplit
            _, noise_idx = next(iter(StratifiedShuffleSplit(n_splits=1,
                                                            test_size=self._noise_ratio,
                                                            random_state=23).split(self.data, self.labels)))
            self.labels[noise_idx] = noise[noise_idx]


    def shuffle_inds(self):
        """Randomly permute the training roidb."""
        if self._is_train:
            self._perm = npr.permutation(np.arange(self.size()))
        else:
            self._perm = np.arange(self.size())
        self._cur = 0


    def pre_process(self, data, labels):
        partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
        one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=np.float32)
        return partial_flatten(data) / 255.0, one_hot(labels, 10)


    def get_data(self, batch_size=1):
        if batch_size == 0:
            return self.pre_process(self.data, self.labels)
        if self._cur + batch_size > self.size():
            self.shuffle_inds()
        db_inds = self._perm[self._cur:self._cur + batch_size]
        self._cur += batch_size
        return self.pre_process(self.data[db_inds, :, :], self.labels[db_inds])