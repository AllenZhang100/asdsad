import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import *
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

def test(iterations):
	# Load the model
	model = Model().cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, 'model.th')))
	model.eval()
	
	# Load the data
	data = ActionDataset('val')
	loss = nn.BCEWithLogitsLoss()
	
	loss_vals = []
	for i in range(iterations):
		obs, actions = data[i]
		obs = torch.as_tensor(obs).float().cuda()
		actions = torch.as_tensor(actions).float().cuda()
		
		obs = obs[None].float().permute(0,1,4,2,3).cuda()
		actions = actions[None].float().permute(0,2,1).cuda()
		
		# print (obs.size(), actions.size())
		
		pred = model.policy()
		
		for i in range(obs.shape[1]):
			# Evaluate the prediction accuracy
			model_output = pred(obs[0,i,...])
			action = actions[0,i,...]
			
			# print (model_output, action)
			
			l = float(loss(model_output, action))
			loss_vals.append(l)
	print('Median log-likelihood loss', np.median(loss_vals))
	print('Mean log-likelihood loss', np.mean(loss_vals))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--iterations', type=int, default=25)
	args = parser.parse_args()

	print ('[I] Testing')
	test(args.iterations)
