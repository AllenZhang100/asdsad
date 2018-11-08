"""
Module that contains that some utility functions
"""

import os
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

class ActionDataset(Dataset):
	def __init__(self, data_dir, crop=None):
		self.data_dir = data_dir
		self.crop = crop
		
		self.trajs = os.listdir(data_dir)
		
		self._cache = {}
		
	
	def __len__(self):
		return len(self.trajs)//2
		
	def __getitem__(self, idx):
		if idx not in self._cache:
			imgs = np.load(os.path.join(self.data_dir, '%04d_img.npy'%idx))
			actions = np.load(os.path.join(self.data_dir, '%04d_action.npy'%idx))
			
			self._cache[idx] = (imgs, actions)
		
		imgs, actions = self._cache[idx]
		
		if self.crop is not None:
			s = np.random.choice(len(imgs) - self.crop + 1)
			imgs = imgs[s:s+self.crop]
			actions = actions[s:s+self.crop]
		
		imgs = (imgs - [4.417,3.339,4.250]) / [23.115,19.552,20.183]

		return imgs, np.unpackbits(actions[None], axis=0)[2:]
		
		
def load(data_filepath, num_workers=0, batch_size=32, **kwargs):
	dataset = ActionDataset(data_filepath, **kwargs)
	return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

class SummaryWriter:
	def __init__(self, *args, **kwargs):
		print("tensorboardX not found. You need to install it to use the SummaryWriter.")
		print("try: pip3 install tensorboardX")
		raise ImportError
try:
	from tensorboardX import SummaryWriter
except ImportError:
	pass