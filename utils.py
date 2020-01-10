import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image


def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)


class GenerativeReplay(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, h_dim=5, z_dim=3):
		super(GenerativeReplay, self).__init__()

		self.action_shape = action_shape
		self.state_shape = state_shape
		self.feature_size = self.action_shape + (2 * self.state_shape) + 2
		self.action_low = action_low
		self.action_high = action_high
		self.state_low = state_low
		self.state_high = state_high
		self.reward_low = -20.0
		self.reward_high = 0.0
		self.z_dim = z_dim
		self.encoder = nn.Sequential(
			nn.Linear(self.feature_size, h_dim),
			nn.LeakyReLU(0.2),
			nn.Linear(h_dim, z_dim * 2)
		)

		self.decoder = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, self.feature_size),
			nn.Tanh()
		)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def normalise(self, x):
		(((x[:,0].sub_(self.state_low[0])).div_((self.state_high[0] - self.state_low[0]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 1].sub_(self.state_low[1])).div_((self.state_high[1] - self.state_low[1]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 2].sub_(self.state_low[2])).div_((self.state_high[2] - self.state_low[2]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 3].sub_(self.action_low)).div_((self.action_high - self.action_low))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 4].sub_(self.state_low[0])).div_((self.state_high[0] - self.state_low[0]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 5].sub_(self.state_low[1])).div_((self.state_high[1] - self.state_low[1]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 6].sub_(self.state_low[2])).div_((self.state_high[2] - self.state_low[2]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 7].sub_(self.reward_low)).div_((self.reward_high - self.reward_low))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 8].sub_(0.0)).div_(1.0)).mul_(2.0)).sub_(1.0).cuda()
		return x

	def descale(self, x):
		(((x[:, 0].add_(1.0)).div_(2.0)).mul_(self.state_high[0] - self.state_low[0])).add_(self.state_low[0]).cuda()
		(((x[:, 1].add_(1.0)).div_(2.0)).mul_(self.state_high[1] - self.state_low[1])).add_(self.state_low[1]).cuda()
		(((x[:, 2].add_(1.0)).div_(2.0)).mul_(self.state_high[2] - self.state_low[2])).add_(self.state_low[2]).cuda()
		(((x[:, 3].add_(1.0)).div_(2.0)).mul_(self.action_high - self.action_low)).add_(self.action_low).cuda()
		(((x[:, 4].add_(1.0)).div_(2.0)).mul_(self.state_high[0] - self.state_low[0])).add_(self.state_low[0]).cuda()
		(((x[:, 5].add_(1.0)).div_(2.0)).mul_(self.state_high[1] - self.state_low[1])).add_(self.state_low[1]).cuda()
		(((x[:, 6].add_(1.0)).div_(2.0)).mul_(self.state_high[2] - self.state_low[2])).add_(self.state_low[2]).cuda()
		(((x[:, 7].add_(1.0)).div_(2.0)).mul_(self.reward_high - self.reward_low)).add_(self.reward_low).cuda()
		((x[:, 8].add_(1.0)).div_(2.0)).round_().cuda()

		return x

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		z = mu + std * esp
		return z

	def activation(self, z):
		for i in range(self.state_shape[0]):
			if i == 0:
				state = nn.Hardtanh(min_val=self.state_low[i], max_val=self.state_high[i])(z[i])
			else:
				state = torch.cat((state, nn.Hardtanh(min_val=self.state_low[i], max_val=self.state_high[i])(z[i])), 0)

		state = nn.Hardtanh(min_val=self.action_low, max_val=self.action_high)(z[0: self.state_shape[0]])
		action = nn.Hardtanh(min_val=self.action_low, max_val=self.action_high)(z[self.state_shape[0]: self.state_shape[0] + self.action_shape[0]])

		for i in range(self.state_shape[0] + self.action_shape[0], 2 * self.state_shape[0] + self.action_shape[0]):
			if i == (self.state_shape[0] + self.action_shape[0]):
				next_state = nn.Hardtanh(min_val=self.state_low[i], max_val=self.state_high[i])(z[i])
			else:
				next_state = torch.cat((state, nn.Hardtanh(min_val=self.state_low[i], max_val=self.state_high[i])(z[i])), 0)

		reward = z[-2]
		done = nn.Sigmoid()(z[-1])

		return torch.cat((state, action, next_state, reward, done), 0)

	def forward(self, x):

		h = self.encoder(x)
		mu, logvar = torch.chunk(h, 2, dim=1)
		z = self.reparameterize(mu, logvar)
		z = self.decoder(z)

		return z, mu, logvar  ## split z to get all components

	def sample(self, batch_size):

		sample = Variable(torch.randn(batch_size, self.z_dim))
		recon_x = np.arctanh(self.decoder(sample).detach().numpy())
		result = self.descale(torch.Tensor(recon_x))

		## descale

		return (
			torch.FloatTensor(result[:, 0:3]).to(self.device),
			torch.FloatTensor(result[:, 3]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, 4:7]).to(self.device),
			torch.FloatTensor(result[:, -2]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, -1]).unsqueeze(1).to(self.device)
		)
