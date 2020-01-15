import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pygit2 import Repository

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
		(((x[:, 0].sub_(self.state_low[0])).div_((self.state_high[0] - self.state_low[0]))).mul_(2.0)).sub_(1.0).cuda()
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
		action = nn.Hardtanh(min_val=self.action_low, max_val=self.action_high)(
			z[self.state_shape[0]: self.state_shape[0] + self.action_shape[0]])

		for i in range(self.state_shape[0] + self.action_shape[0], 2 * self.state_shape[0] + self.action_shape[0]):
			if i == (self.state_shape[0] + self.action_shape[0]):
				next_state = nn.Hardtanh(min_val=self.state_low[i], max_val=self.state_high[i])(z[i])
			else:
				next_state = torch.cat(
					(state, nn.Hardtanh(min_val=self.state_low[i], max_val=self.state_high[i])(z[i])), 0)

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
		# recon_x = np.arctanh(self.decoder(sample).detach().numpy())
		# result = self.descale(torch.FloatTensor(recon_x))
		result = self.descale(self.decoder(sample).detach())
		## descale

		return (
			torch.FloatTensor(result[:, 0:3]).to(self.device),
			torch.FloatTensor(result[:, 3]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, 4:7]).to(self.device),
			torch.FloatTensor(result[:, -2]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, -1]).unsqueeze(1).to(self.device)
		)


class Generator_XY(nn.Module):
	def __init__(self, state_shape, action_shape, latent_size):
		super(Generator_XY, self).__init__()

		self.state_shape = state_shape
		self.action_shape = action_shape
		self.latent_size = latent_size

		self.model = nn.Sequential(
			nn.Linear(self.state_shape + self.latent_size, 2),
			nn.LeakyReLU(0.2),
			nn.Linear(2, self.action_shape),
			nn.Tanh()
		)

	def forward(self, x):
		# Concatenate label embedding and image to produce input

		img = self.model(x)
		img = img.view(-1, self.action_shape)
		return img


class Generator_YX(nn.Module):
	def __init__(self, state_shape, action_shape, latent_size):
		super(Generator_YX, self).__init__()

		self.state_shape = state_shape
		self.action_shape = action_shape
		self.latent_size = latent_size

		self.model = nn.Sequential(
			nn.Linear(self.action_shape + self.latent_size, 5),
			nn.LeakyReLU(0.2),
			nn.Linear(5, self.state_shape),
			nn.Tanh()
		)

	def forward(self, x):
		# Concatenate label embedding and image to produce input

		img = self.model(x)
		img = img.view(-1, self.state_shape)
		return img


class Discriminator(nn.Module):

	def __init__(self, state_shape, action_space):
		super(Discriminator, self).__init__()
		self.state_shape = state_shape
		self.action_space = action_space
		self.model = nn.Sequential(
			nn.Linear(self.state_shape + self.action_space, 6),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(6, 4),
			nn.Linear(4, 5),
		)

	def forward(self, state, action):
		# Concatenate label embedding and image to produce input
		xy = torch.cat((state, action), 1)
		validity = self.model(xy)
		return validity


class DiscriminatorX(nn.Module):

	def __init__(self, state_shape):
		super(DiscriminatorX, self).__init__()
		self.state_shape = state_shape
		self.model = nn.Sequential(
			nn.Linear(self.state_shape, 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(2, 1),
		)

	def forward(self, x1):
		# Concatenate label embedding and image to produce input
		validity = self.model(x1)
		return validity


class DiscriminatorY(nn.Module):
	def __init__(self, action_shape):
		super(DiscriminatorY, self).__init__()
		self.action_shape = action_shape
		self.model = nn.Sequential(
			nn.Linear(self.action_shape, 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(2, 1),
		)

	def forward(self, y1):
		# Concatenate label embedding and image to produce input
		validity = self.model(y1)
		return validity


class JointGANTrainer():
	def __init__(self, state_shape, action_shape, batch_size, latent_size, action_low, action_high, state_low, state_high):
		self.state_shape = state_shape
		self.action_shape = action_shape
		self.batch_size = batch_size
		self.latent_size = latent_size
		self.action_low = action_low
		self.action_high = action_high
		self.state_low = state_low
		self.state_high = state_high
		self.gen_xy = Generator_XY(self.state_shape, self.action_shape, self.latent_size).cuda()
		self.gen_yx = Generator_YX(self.state_shape, self.action_shape, self.latent_size).cuda()
		self.discriminator = Discriminator(self.state_shape, self.action_shape).cuda()
		self.discriminatorX = DiscriminatorX(self.state_shape).cuda()
		self.discriminatorY = DiscriminatorY(self.action_shape).cuda()

		self.G_params = list(self.gen_xy.parameters()) + list(self.gen_yx.parameters())
		self.D_params = list(self.discriminator.parameters()) + list(self.discriminatorX.parameters()) + list(self.discriminatorY.parameters())

		self.optimizer_G = torch.optim.Adam(self.G_params, lr=0.0002)  ## param sharing between generators ??
		self.optimizer_D = torch.optim.Adam(self.D_params, lr=0.0002)


	def normalise(self, x):
		(((x[:, 0].sub_(self.state_low[0])).div_((self.state_high[0] - self.state_low[0]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 1].sub_(self.state_low[1])).div_((self.state_high[1] - self.state_low[1]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 2].sub_(self.state_low[2])).div_((self.state_high[2] - self.state_low[2]))).mul_(2.0)).sub_(1.0).cuda()
		(((x[:, 3].sub_(self.action_low)).div_((self.action_high - self.action_low))).mul_(2.0)).sub_(1.0).cuda()

		return x

	def descale(self, x):
		(((x[:, 0].add_(1.0)).div_(2.0)).mul_(self.state_high[0] - self.state_low[0])).add_(self.state_low[0]).cuda()
		(((x[:, 1].add_(1.0)).div_(2.0)).mul_(self.state_high[1] - self.state_low[1])).add_(self.state_low[1]).cuda()
		(((x[:, 2].add_(1.0)).div_(2.0)).mul_(self.state_high[2] - self.state_low[2])).add_(self.state_low[2]).cuda()
		(((x[:, 3].add_(1.0)).div_(2.0)).mul_(self.action_high - self.action_low)).add_(self.action_low).cuda()

		return x

	def compute_D_losses(self, state, action, a_x, b_y, p2, p1, p4, p3):

		D1 = self.discriminator(state, action)
		#D2 = self.discriminator()
		D6 = self.discriminatorX(p2)
		D8 = self.discriminatorX(p4)
		D9 = self.discriminatorX(state)

		D10 = self.discriminatorY(p1)
		D12 = self.discriminatorY(p3)
		D13 = self.discriminatorY(action)

		# lables are float
		D1_label = torch.zeros((int(D1.size()[0])), dtype=torch.long).fill_(0).cuda()
		D1_loss = nn.CrossEntropyLoss(reduction='mean')(D1, D1_label)


		real_labels = torch.ones((self.batch_size, 1)).cuda()
		fake_labels = torch.zeros((self.batch_size, 1)).cuda()

		D9_loss = nn.MSELoss()(D9, real_labels)
		D13_loss = nn.MSELoss()(D13, real_labels)

		D6_loss = nn.MSELoss()(D6, fake_labels)
		D8_loss = nn.MSELoss()(D8, fake_labels)
		D10_loss = nn.MSELoss()(D10, fake_labels)
		D12_loss = nn.MSELoss()(D12, fake_labels)

		d_loss = D1_loss + D6_loss + D8_loss + D9_loss + D10_loss + D12_loss + D13_loss

		return d_loss

	def compute_G_losses(self, state, action, a_x, b_y, p2, p1, p4, p3):

		g1 = self.discriminator(p2, action)
		g2 = self.discriminator(state, p1)
		g3 = self.discriminator(a_x, p3)
		g4 = self.discriminator(p4, b_y)
		#g5 = self.discriminator(state, action)

		# lables are float
		g1_label = torch.zeros((int(g1.size()[0])), dtype=torch.long).fill_(0).cuda()
		g1_loss = nn.CrossEntropyLoss(reduction='mean')(g1, g1_label)

		g2_label = torch.zeros((int(g2.size()[0])), dtype=torch.long).fill_(1).cuda()
		g2_loss = nn.CrossEntropyLoss(reduction='mean')(g2, g2_label)

		g3_label = torch.zeros((int(g3.size()[0])), dtype=torch.long).fill_(2).cuda()
		g3_loss = nn.CrossEntropyLoss(reduction='mean')(g3, g3_label)

		g4_label = torch.zeros((int(g4.size()[0])), dtype=torch.long).fill_(3).cuda()
		g4_loss = nn.CrossEntropyLoss(reduction='mean')(g4, g4_label)

		#g5_label = torch.zeros((int(g5.size()[0])), dtype=torch.long).fill_(4).cuda()
		#g5_loss = nn.CrossEntropyLoss(reduction='mean')(g5, g5_label)

		g_loss = g1_loss + g2_loss + g3_loss + g4_loss
		return g_loss

	def train(self, train_data):
		states = train_data[:, 0: self.state_shape]
		actions = train_data[:, -1]
		count = 0
		for epoch in range(10):
			count = count + 1
			perm = torch.randperm(actions.size()[0])
			action = actions[perm]
			state = states[perm]
			action = torch.FloatTensor(action[0:self.batch_size]).unsqueeze(1).cuda()
			state = torch.FloatTensor(state[0:self.batch_size]).cuda()

			self.optimizer_G.zero_grad()

			# P2(x, y) = q(y) p(x|y) phi
			z1 = torch.FloatTensor(torch.randn((self.batch_size, self.latent_size))).cuda()
			self.P2 = self.gen_yx(torch.cat((action, z1), 1))

			# P(y)
			beta_0 = torch.FloatTensor(torch.zeros((self.batch_size, self.state_shape))).cuda()
			z2 = torch.FloatTensor(torch.randn((self.batch_size, self.latent_size))).cuda()
			self.beta_y = self.gen_xy(torch.cat((beta_0, z2), 1))

			# P4(x, y) = P(y) P(x|y) phi
			z3 = torch.FloatTensor(torch.randn((self.batch_size, self.latent_size))).cuda()
			self.P4 = self.gen_yx(torch.cat((self.beta_y, z3), 1))

			# P(x)
			alpha_0 = torch.FloatTensor(torch.zeros((self.batch_size, self.action_shape))).cuda()
			z4 = torch.FloatTensor(torch.randn((self.batch_size, self.latent_size))).cuda()
			self.alpha_x = self.gen_yx(torch.cat((alpha_0, z4), 1))

			# P3(x, y) = P(x) p(y|x) theta
			z5 = torch.FloatTensor(torch.randn((self.batch_size, self.latent_size))).cuda()
			self.P3 = self.gen_xy(torch.cat((self.alpha_x, z5), 1))

			# P1(x, y) = q(x) p(y|x) theta
			z6 = torch.FloatTensor(torch.randn((self.batch_size, self.latent_size))).cuda()
			self.P1 = self.gen_xy(torch.cat((state, z6), 1))

			generator_loss = self.compute_G_losses(state, action, self.alpha_x, self.beta_y, self.P2, self.P1, self.P4, self.P3)

			generator_loss.backward()

			self.optimizer_G.step()

			self.optimizer_D.zero_grad()
			discriminator_loss = self.compute_D_losses(state, action, self.alpha_x.detach(), self.beta_y.detach(), self.P2.detach(), self.P1.detach(), self.P4.detach(), self.P3.detach())
			'''
			# discriminator_loss = self.compute_D_losses(state, action, self.alpha_x, self.beta_y, self.P2, self.P1, self.P4, self.P3)
			D1 = self.discriminator(self.P2.detach(), action)
			D2 = self.discriminator(state, self.P1.detach())
			D3 = self.discriminator(self.alpha_x.detach(), self.P3.detach())
			D4 = self.discriminator(self.P4.detach(), self.beta_y.detach())
			D5 = self.discriminator(state, action)

			D6 = self.discriminatorX(self.P2.detach())
			D7 = self.discriminatorX(self.alpha_x.detach())
			D8 = self.discriminatorX(self.P4.detach())
			D9 = self.discriminatorX(state)

			D10 = self.discriminatorX(self.P1.detach())
			D11 = self.discriminatorX(self.beta_y.detach())
			D12 = self.discriminatorX(self.P3.detach())
			D13 = self.discriminatorX(action)

			# lables are float
			D1_label = torch.zeros((int(D1.size()[0])), dtype=torch.long).fill_(0).cuda()
			D1_loss = nn.CrossEntropyLoss(reduction='mean')(D1, D1_label)

			D2_label = torch.zeros((int(D2.size()[0])), dtype=torch.long).fill_(1).cuda()
			D2_loss = nn.CrossEntropyLoss(reduction='mean')(D2, D2_label)

			D3_label = torch.zeros((int(D3.size()[0])), dtype=torch.long).fill_(2).cuda()
			D3_loss = nn.CrossEntropyLoss(reduction='mean')(D3, D3_label)

			D4_label = torch.zeros((int(D4.size()[0])), dtype=torch.long).fill_(3).cuda()
			D4_loss = nn.CrossEntropyLoss(reduction='mean')(D4, D4_label)

			D5_label = torch.zeros((int(D5.size()[0])), dtype=torch.long).fill_(4).cuda()
			D5_loss = nn.CrossEntropyLoss(reduction='mean')(D5, D5_label)

			real_labels = torch.ones((self.batch_size, 1)).cuda()
			fake_labels = torch.zeros((self.batch_size, 1)).cuda()



			discriminator_loss = D1_loss + D2_loss + D3_loss + D4_loss + D5_loss
			
			'''
			discriminator_loss.backward()

			self.optimizer_D.step()

		return discriminator_loss.item(), generator_loss.item(), self.gen_xy.state_dict(), self.gen_yx.state_dict(), \
			   self.discriminator.state_dict(), self.optimizer_G.state_dict(), self.optimizer_D.state_dict()

	def sample(self, size):
		noise = torch.randn(size, self.latent_size)
		z = torch.zeros((size, self.action_shape))
		state = self.gen_yx(torch.cat((z, noise), 1).cuda())
		result = self.descale(state.detach())

		action = self.descale(self.gen_xy(torch.cat((result, noise), 1)).detach())

		return torch.cat((result, action), 1).cuda()
