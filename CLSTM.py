import torch
import torch.nn as nn
from torch.autograd import Variable
import sys
from Unets import transition_UNet, transition_UNet_small, UNet, UNet_small, transition_UNet_large, UNet_large

from collections import OrderedDict
# Batch x NumChannels x Height x Width
# UNET --> BatchSize x 1 (3?) x 240 x 240
# BDCLSTM --> BatchSize x 64 x 240 x240

def conv_batchNorm(in_channels, out_channels):
	return nn.Sequential(
		nn.Conv3d(in_channels, out_channels, 3, padding=1),
		nn.BatchNorm3d(out_channels),
		nn.ReLU(inplace=True),
		nn.Conv3d(out_channels, out_channels, 3, padding=1),
		nn.BatchNorm3d(out_channels),
		nn.ReLU(inplace=True),
	)

''' Class CLSTMCell.
	This represents a single node in a CLSTM series.
	It produces just one time (spatial) step output.
'''
class CLSTMCell(nn.Module):

	# Constructor
	def __init__(self, input_channels, hidden_channels,
				 kernel_size, bias=True, device='cuda:0'):
		super(CLSTMCell, self).__init__()

		assert hidden_channels % 2 == 0

		self.input_channels = input_channels
		self.hidden_channels = hidden_channels
		self.bias = bias
		self.kernel_size = kernel_size
		self.num_features = 4

		self.padding = (kernel_size - 1) // 2
		self.conv = nn.Conv3d(self.input_channels + self.hidden_channels,
							  self.num_features * (self.hidden_channels),
							  self.kernel_size,
							  1,
							  self.padding)#.to(device)

	# Forward propogation formulbdclstmation
	def forward(self, x, h, c):
		combined = torch.cat((x, h), dim=1)
		A = self.conv(combined)

		# NOTE: A? = xz * Wx? + hz-1 * Wh? + b? where * is convolution
		(Ai, Af, Ao, Ag) = torch.split(A,
									   A.size()[1]//self.num_features,
									   dim=1)

		i = torch.sigmoid(Ai)     # input gate
		f = torch.sigmoid(Af)     # forget gate
		o = torch.sigmoid(Ao)     # output gate
		g = torch.tanh(Ag)
		c = c * f + i * g           # cell activation state
		h = o * torch.tanh(c)     # cell hidden state

		return h, c

	@staticmethod
	def init_hidden(batch_size, hidden_c, shape):
		try:
			return(Variable(torch.zeros(batch_size,
									hidden_c,
									shape[0],
									shape[1],
									shape[2])).cuda(),
			   Variable(torch.zeros(batch_size,
									hidden_c,
									shape[0],
									shape[1],
									shape[2])).cuda())
		except:
			return(Variable(torch.zeros(batch_size,
									hidden_c,
									shape[0],
									shape[1],
									shape[2])),
					Variable(torch.zeros(batch_size,
									hidden_c,
									shape[0],
									shape[1],
									shape[2])))


''' Class CLSTM.
	This represents a series of CLSTM nodes (one direction)
'''
class CLSTM(nn.Module):
	# Constructor
	def __init__(self, input_channels=64, hidden_channels=[64],
				 kernel_size=5, bias=True):
		super(CLSTM, self).__init__()

		# store stuff
		self.input_channels = [input_channels] + hidden_channels
		self.hidden_channels = hidden_channels
		self.kernel_size = kernel_size
		self.num_layers = len(hidden_channels)

		self.bias = bias
		self.all_layers = []

		# create a node for each layer in the CLSTM
		self.cell = CLSTMCell(self.input_channels[0],
							 self.hidden_channels[0],
							 self.kernel_size,
							 self.bias)

	# Forward propogation
	# x --> BatchSize x NumSteps x NumChannels x Height x Width
	#       BatchSize x 2 x 64 x 48 x 48
	def forward(self, x):
		# print('in CLSTM, x size:', x.size())
		bsize, steps, _, depth, height, width = x.size()
		internal_state = []
		outputs = []
		for step in range(steps):
			input = x[:, step, :, :, :, :]
			for layer in range(self.num_layers):
				# populate hidden states for all layers
				if step == 0:
					(h, c) = CLSTMCell.init_hidden(bsize,
												   self.hidden_channels[layer],
												   (depth, height, width))
					internal_state.append((h, c))

				# do forward
				name = 'cell{}'.format(layer)
				(h, c) = internal_state[layer]

				input, c = self.cell(input, h, c)  # forward propogation call
				internal_state[layer] = (input, c)

			outputs.append(input) # 2x1x32x64x64x64

		return outputs


class BDCLSTM_unet(nn.Module):
	# Constructor
	def __init__(self, network, network_size, unet_path=None, input_channels=64, hidden_channels=[64],
				 last_fc = 64, kernel_size=5, bias=True, num_classes=1, pretrained='False',
				 parameter_fix='False', version='0'):
		super(BDCLSTM_unet, self).__init__()
		self.last_fc = last_fc
		self.version = version

		if network == 'unet':
			if network_size == 'normal':
				self.unet = UNet(1, ngf=input_channels)
			elif network_size == 'large':
				self.unet = UNet_large(1, ngf=input_channels)
			else:
				self.unet = UNet_small(1, ngf=input_channels)
		else:
			if network_size == 'normal':
				self.unet = transition_UNet(1, ngf=input_channels)
				# self.unet_center = transition_UNet(1, ngf=input_channels)
			elif network_size == 'large':
				self.unet = transition_UNet_large(1, ngf=input_channels)
			else:
				self.unet = transition_UNet_small(1, ngf=input_channels)
		
		if pretrained == 'True' and unet_path != None:
			print('load pretrained Unet!')
			pre_dict = torch.load(unet_path, map_location='cpu')#['state_dict']
			new_state_dict = OrderedDict()
			for k, v in pre_dict.items():
				name = k[7:]
				new_state_dict[name] = v
			# model_dict = self.unet_center.state_dict()

			# pretrained_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
			# model_dict.update(pretrained_dict)
			# self.unet_center.load_state_dict(model_dict)
			self.unet.load_state_dict(new_state_dict)
			if parameter_fix == 'True':
				for param in self.unet.parameters():
					param.requires_grad = False

		self.CLSTM_list = nn.ModuleList()
		for i in range(6):
			self.CLSTM_list.append(
				CLSTM(last_fc, hidden_channels, kernel_size, bias)
			)
		# version 5
		if version == '5':
			self.last_conv = conv_batchNorm(hidden_channels[-1], num_classes)

		# version 4
		if version == '4':
			self.conv_x = nn.Conv3d(
				2 * hidden_channels[-1], hidden_channels[-1], kernel_size=1)
			self.batch_x = nn.BatchNorm3d(hidden_channels[-1])
			self.conv_y = nn.Conv3d(
				2 * hidden_channels[-1], hidden_channels[-1], kernel_size=1)
			self.batch_y = nn.BatchNorm3d(hidden_channels[-1])
			self.conv_z = nn.Conv3d(
				2 * hidden_channels[-1], hidden_channels[-1], kernel_size=1)
			self.batch_z = nn.BatchNorm3d(hidden_channels[-1])
		
			self.conv1 = nn.Conv3d(
				3 * hidden_channels[-1], hidden_channels[-1], kernel_size=1)
			self.batch1 = nn.BatchNorm3d(hidden_channels[-1])
			self.relu = nn.ReLU(inplace=True)

			self.conv2 = nn.Conv3d(
				hidden_channels[-1], num_classes, kernel_size=1)
		
		# version 3
		if version == '3':
			self.conv = nn.Conv3d(
				6 * hidden_channels[-1], num_classes, kernel_size=1)
		
		# version 2
		if version == '2':
			self.conv1 = nn.Conv3d(
				6 * hidden_channels[-1], hidden_channels[-1], kernel_size=1)
			self.conv2 = nn.Conv3d(
				last_fc + hidden_channels[-1], num_classes, kernel_size=1)

		# version 1
		if version == '1':
			self.conv1 = nn.Conv3d(
				6 * hidden_channels[-1], 3 * hidden_channels[-1], kernel_size=1)
			self.batch1 = nn.BatchNorm3d(3 * hidden_channels[-1])

			self.conv2 = nn.Conv3d(
			 	3 * hidden_channels[-1], hidden_channels[-1], kernel_size=1)
			self.batch2 = nn.BatchNorm3d(hidden_channels[-1])

			self.conv3 = nn.Conv3d(
				hidden_channels[-1], num_classes, kernel_size=1)
			self.relu = nn.ReLU(inplace=True)
		

	# Forward propogation
	# feature_maps -- > batch x temporal x channel x depth x height x width
	# 					batch x 7 x 64 x 48 x 48 x 48
	def forward(self, input):

		feature_list = []
		input_center = input[:, 0, :, :, :, :]
		if self.last_fc == 1:
			unet_feature_center = self.unet(input_center, return_features=False)
			# unet_feature_center = self.unet_center(input_center, return_features=False)
		else:
			unet_feature_center = self.unet(input_center, return_features=True)
			# unet_feature_center = self.unet_center(input_center, return_features=True)
		feature_list.append(unet_feature_center)
		for t in range(1, 7):
			x_input = input[:, t, :, :, :, :]
			if self.last_fc != 1:
				unet_feature = self.unet(x_input, return_features=True)
			else:
				unet_feature = self.unet(x_input, return_features=False)
			feature_list.append(unet_feature)
		feature_maps = torch.stack(feature_list, dim=1)

		_, temporal, ch, d, h, w = feature_maps.size()

		x_center = feature_maps[:, 0, :, :, :, :]
		x_center = torch.unsqueeze(x_center, dim=1)

		lstm_result_list = []
		for t in range(1, temporal):
			x = feature_maps[:, t, :, :, :, :] # b x 64 x 48 x 48 x 48
			# print('featuremap size:', x.size())
			x = torch.unsqueeze(x, dim=1)

			x_merged = torch.cat((x, x_center), dim=1) # like xforward
			x_lstm = self.CLSTM_list[t-1](x_merged)
			lstm_result_list.append(x_lstm)

		x_start = lstm_result_list[0][-1]
		if self.version != '5':
			for i in range(1, len(lstm_result_list)):
				x_start = torch.cat((lstm_result_list[i][-1], x_start), dim=1)
		
		# version 5
		if self.version == '5':
			for i in range(1, len(lstm_result_list)):
				x_start += lstm_result_list[i][-1]
			x_start /= len(lstm_result_list)
			y = self.last_conv(x_start)

		# version 4
		if self.version == '4':
			temp = torch.cat((lstm_result_list[0][-1], lstm_result_list[1][-1]), dim=1)
			# print(temp.size())
			lstm_x = self.relu(self.batch_x(self.conv_x(torch.cat((lstm_result_list[0][-1], lstm_result_list[1][-1]), dim=1))))
			lstm_y = self.relu(self.batch_y(self.conv_y(torch.cat((lstm_result_list[2][-1], lstm_result_list[3][-1]), dim=1))))
			lstm_z = self.relu(self.batch_z(self.conv_z(torch.cat((lstm_result_list[4][-1], lstm_result_list[5][-1]), dim=1))))
			lstm_result = torch.cat((lstm_x, lstm_y, lstm_z), dim=1)
		
			y = self.conv1(lstm_result)
			y = self.relu(self.batch1(y))
			y = self.conv2(y)

		# x_start shape : batch x 6*64 x 48 x 48 x 48

		# version 3 / using 1 cnn
		if self.version == '3':
			y = self.conv(x_start)

		# version 2
		if self.version == '2':
			y = self.conv1(x_start)
			y = torch.cat((feature_maps[:, 0, :, :, :], y), dim=1)
			y = self.conv2(y)

		# version 1
		if self.version == '1':
			y = self.conv1(x_start)
			y = self.batch1(y)
			y = self.relu(y)
			y = self.conv2(y)
			y = self.batch2(y)
			y = self.relu(y)
			y = self.conv3(y)

		return y


if __name__ == "__main__":
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	net = BDCLSTM(input_channels=64, hidden_channels=[64], num_classes=1, device=device).cuda()


	CT = torch.randn(1, 7, 64, 48, 48, 48).to(device)
	out = net(CT)
	print(out.shape)
	pass