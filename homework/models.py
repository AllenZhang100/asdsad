from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class Policy:
	"""
	Class used for evaluation. Will take a single observation as input in the __call__ function and need to output the l6 dimensional logits for next action
	"""
	def __init__(self, model):
		self.model = model
		self.hs = None
		
	def __call__(self, obs):
		callreta, callretb = self.model(obs[None,:,None], self.hs)
		#print(callreta.size())
		#print(callreta)
		#print("crb")
		#print(callretb.size())

		#print(callretb)
		#print("craa")
		self.hs = callretb
		#prediction
		#print(callreta[0,:,-1])
		return callreta[0,:,-1]
		
class Model(nn.Module):
	def __init__(self):
		super().__init__()
		
		self.lstm = nn.LSTM(
			input_size=6,
			hidden_size=64,
			num_layers=1,
			batch_first = True
		)
		self.linear = nn.Linear(64*4096, 6)

		self.xlin = nn.Linear(4096, 128)

		#convnet
		self.fconv1 = nn.Conv2d(3, 32, 4, 2, 1)
		self.fconv2 = nn.Conv2d(32, 64, 4, 2, 1)
		self.fconv3 = nn.Conv2d(64, 128, 4, 2, 1)
		self.relu = nn.ReLU()
		#upcon
		self.uconv1 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
		self.uconv2 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
		self.uconv3 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
		self.finalconv = nn.Conv2d(16, 6, 1, 1, 0)


		
	def forward(self, hist):
		'''
		Your code here
		Input size: (batch_size, sequence_length, channels, height, width)
		Output size: (batch_size, sequence_length, 6)
		'''

		dims = list(hist.size())
		batchsize = dims[0]
		seqsize = dims[1]

		hist = hist.view(-1, 3, 64, 64)
		print(hist.size())
		'''
		x = hist
		#fcon
		x = self.fconv1(x)
		x = self.relu(x)
		c1 = x
		x = self.fconv2(x)
		x = self.relu(x)
		c2 = x
		#print(up1.size())
		#print(c2.size())
		up2 = self.uconv2(c2) #64
		#print(up2.size())
		#print(c1.size())
		up3 = self.uconv3(up2 + c1) #32
		#?
		up3 = self.finalconv(up3)
		convout = up3
		'''
		x = hist
		#fcon
		x = self.fconv1(x)
		x = self.relu(x)
		c1 = x
		x = self.fconv2(x)
		x = self.relu(x)
		c2 = x
		x = self.fconv3(x)
		x = self.relu(x)
		c3 = x
		#upcon
		up1 = self.uconv1(c3)
		#print(up1.size())
		#print(c2.size())
		up2 = self.uconv2(up1 + c2) #64
		#print(up2.size())
		#print(c1.size())
		up3 = self.uconv3(up2 + c1) #32
		#?
		up3 = self.finalconv(up3)
		convout = up3

		#print("  convout: " + str(convout.size()))
		convout = convout.view(convout.size()[:2] + (-1, ))
		#print("  flat: " + str(convout.size()))

		#?
		#convout = self.xlin(convout)

		#output, ohidden = self.lstm(convout.permute(0,2,1,3,4), hidden)
		output, ohidden = self.lstm(convout.permute(0,2,1))

		output = output.permute(0,2,1)
		#print("ooutput" + str(output.size()))
		output = output.contiguous().view(batchsize,seqsize, -1)
		#output = output.view(16,20,-1)
		#print("oooutput" + str(output.size()))
		output = self.linear(output.contiguous())
		#print("  output: " + str(output.size()))


		'''
		dims = list(hist.size())
		batchsize = dims[0]
		#seqlen = dims[1]
		finoutput = torch.FloatTensor(16,20,6)
		#print(finoutput.size())
		#print(finoutput)

		for batchc in range(0,batchsize):
			print("batchc: " + str(batchc))
			#print("  " + str(hist.size()))
			curbatch = hist.select(0,batchc)
			#print("  " + str(curbatch.size()))

			x = curbatch
			#fcon
			x = self.fconv1(x)
			x = self.relu(x)
			c1 = x
			x = self.fconv2(x)
			x = self.relu(x)
			c2 = x
			x = self.fconv3(x)
			x = self.relu(x)
			c3 = x
			#upcon
			up1 = self.uconv1(c3)
			#print(up1.size())
			#print(c2.size())
			up2 = self.uconv2(up1 + c2) #64
			#print(up2.size())
			#print(c1.size())
			up3 = self.uconv3(up2 + c1) #32
			#?
			up3 = self.finalconv(up3)
			convout = up3


			#print("  convout: " + str(convout.size()))
			convout = convout.view(convout.size()[:2] + (-1, ))
			print("  flat: " + str(convout.size()))

			#output, ohidden = self.lstm(convout.permute(0,2,1,3,4), hidden)
			output, ohidden = self.lstm(convout.permute(0,2,1))

			print("ooutput" + str(output.size()))
			output = self.linear(output.contiguous())
			output = output.permute(0,2,1)
			print("  output: " + str(output.size()))
		'''


		#return output, ohidden
		return output

	def policy(self):
		return Policy(self)