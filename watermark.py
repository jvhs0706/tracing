import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import numpy as np
import pickle as pk
import sys, os

class Watermark(nn.Module):
	def __init__(self, locations: np.array):
		'''
		locations: (N, 3) [[cha0, row0, col0], [cha1, row1, col1], [cha2, row2, col2], ...]
		'''
		super().__init__()
		assert len(locations.shape) == 2 and locations.shape[1] == 3
		self.locations = locations
		
			
	def forward(self, X):
		C, H, W = X.shape[-3:]
		if isinstance(X, torch.Tensor):
			mask = torch.ones_like(X, dtype = X.dtype, device = X.device)
			mask[..., self.locations[:, 0], self.locations[:, 1], self.locations[:, 2]] = 0.0
			return X * mask
		
		elif isinstance(X, np.ndarray):
			out = X.copy()
			out[..., self.locations[:, 0], self.locations[:, 1], self.locations[:, 2]] = 0.0
			return out
		
		else:
			raise TypeError

	def get_values(self, X):
		return X[..., self.locations[:, 0], self.locations[:, 1], self.locations[:, 2]]

	def save(self, fn):
		np.save(fn, self.locations)

	def get_values(self, X):
		return X[..., self.locations[:, 0], self.locations[:, 1], self.locations[:, 2]]
			
	@staticmethod
	def load(fn):
		return Watermark(np.load(fn))

	@staticmethod
	def random(num_masked_dims, C, H, W):
		indices = np.random.choice(C * H * W, size = num_masked_dims, replace = False)
		watermark = Watermark(np.stack([indices // (H * W), (indices // W) % H, indices % W], axis = -1))
		return watermark 

if __name__ == '__main__':
	pass