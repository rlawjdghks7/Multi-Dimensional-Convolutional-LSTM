import os, sys
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import pytorch_unet
from Unets import transition_UNet, UNet, transition_UNet_large, UNet_large, encoder_decoer
from CLSTM import BDCLSTM

import torch.nn as nn

from collections import defaultdict
import torch.nn.functional as F

from loss import calc_loss
from utils import *

import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from tqdm import tqdm
import argparse
from dataloader import LSTMDataset, UnetDataset, CV_Unet_Dataset, CV_LSTMDataset


def train_unet(model, dataloaders, optimizer, scheduler, temporal_model_dir, unet_device, bce_weight=0.25, num_epochs=10):
	best_model_wts = copy.deepcopy(model.state_dict())
	best_loss = 1
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-'*10)

		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				for param_group in optimizer.param_groups:
					print('LR', param_group['lr'])
				model.train()
			else:
				model.eval()

			metrics = defaultdict(float)
			epoch_samples = 0

			for inputs, labels, _, _ in tqdm(dataloaders[phase]):
				inputs = inputs.to(unet_device)
				labels = labels.to(unet_device)
				optimizer.zero_grad()
				
				with torch.set_grad_enabled(phase=='train'):
					outputs = model(inputs)
					print(inputs.get_device())
					print(labels.get_device())
					print(outputs.get_device())
					loss, _ = calc_loss(outputs, labels, metrics, bce_weight=bce_weight)

					if phase == 'train':
						loss.backward()
						optimizer.step()
				epoch_samples += inputs.size(0)
			print_metrics(metrics, epoch_samples, phase)
			epoch_loss = metrics['dice'] / epoch_samples
			if phase == 'val' and epoch_loss < best_loss:
				print('best loss changed!')
				best_loss = epoch_loss
				best_model_wts = copy.deepcopy(model.state_dict())
	model.load_state_dict(best_model_wts)
	return model


def Arg():
	parser = argparse.ArgumentParser(description='CT image train')
	parser.add_argument('-d', '--root_dir', dest='root_dir', default='/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA',
						help='set root_dir, default is "/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA"')
	parser.add_argument('-t', '--target', dest='target', default='pancreas',
						help='choose target, liver or pancreas, default is liver')
	parser.add_argument('-nf', '--n_fold', dest='n_fold', default=4, type=int,
						help='set number of folds, default is 4')
	parser.add_argument('-g', '--unet_gpu', dest='unet_gpu', default='0',
						help='choose gpu id, 0~7, default=0, if you want use multi-gpu write "0,1,2,3"')
	parser.add_argument('-e', '--unet_epochs', dest='unet_epochs', default=30, type=int,
						help="choose num epoch, default is 30")
	parser.add_argument('-b', '--batch_size', dest='batch_size', default=8, type=int,
						help='set batch size')
	parser.add_argument('-p', '--patch_size', dest='patch_size', default=64, type=int,
						help='set patch size')
	parser.add_argument('-v', dest='version', default='old',
						help='old version means clip the intensity value [-300, 350], new version is [-100, 240], default is old')
						
	parser.add_argument('--padding_size', dest='padding_size', default='stride',
						help='set padding_size. stride or patch, default is stride')
	parser.add_argument('--stride_size', dest='stride_size', default='half',
						help='set stride_size. half or triple, default is half')

	parser.add_argument('--network_size', dest='network_size', default='normal',
						help='set network_size. normal or small, default is normal')
	parser.add_argument('--optim', dest='optim', default='Adam',
						help='set optim. SGD or Adam, default is Adam')
	parser.add_argument('--input_channels', dest='input_channels', default=64, type=int,
						help='set input_channels. default is 64')
	parser.add_argument('--bce_weight', dest='bce_weight', default=0.25, type=float,
						help='set bce_weight. default is 0.25, 0 means only use dice loss')
	parser.add_argument('--unet_type', dest='unet_type', default='transition',
						help='choose what Unet network training(for using bdclstm), unet or transition or encoder_decoder, defulat is transition')
	return parser.parse_args()


def main():
	args = Arg()
	root_dir = args.root_dir
	data_root = '../datasets'
	unet_device = torch.device('cuda:{}'.format(args.unet_gpu) if torch.cuda.is_available() else 'cpu')

	model_dir = os.path.join(root_dir, 'models')
	cur_model_dir = make_dir(model_dir, '3D_lstm')

	# split dataset to N_fold
	pre_dir = os.path.join(data_root, 'preprocessed')
	if args.stride_size == 'half':
		data_path = os.path.join(pre_dir, '{}_patch_{}_padding:{}_{}_version'.format(args.target, args.patch_size, args.padding_size, args.version))
	else:
		data_path = os.path.join(pre_dir, '{}_patch_{}_padding:{}_stride:{}'.format(args.target, args.patch_size, args.padding_size, args.stride_size))
	fold_list = split_subject_for_cv(data_path, n_fold=args.n_fold)

	unet_device_ids = []
	for id in args.unet_gpu.split(','):
		unet_device_ids.append(int(id))
	torch.cuda.set_device(unet_device_ids[0])

	Unets_model_dir = make_dir(cur_model_dir, 'Unets_{}'.format(args.optim))

	if args.stride_size == 'half':
		hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_input:{}_version:{}_185'.format('Unet', 
						args.unet_type, args.network_size, args.patch_size, args.unet_epochs, 'stride', args.input_channels, args.version)
	else:
		hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_stride:{}_input:{}'.format('Unet', 
						args.unet_type, args.network_size, args.patch_size, args.unet_epochs, args.padding_size, args.stride_size, args.input_channels)
	print('hyper parameter set:', hyper_name)
	for i in range(3, args.n_fold):
		subject_paths = split_fold_to_train(fold_list, i)	
		
		print('[{}] fold training start!'.format(i))
		
		if args.unet_type == 'transition':
			if args.network_size == 'large':
				unet = transition_UNet_large(1, args.input_channels).to(unet_device)
			else:
				unet = transition_UNet(1, args.input_channels).to(unet_device)
			unetType_model_dir = make_dir(Unets_model_dir, 'transition')
		elif args.unet_type == 'unet':
			if args.network_size == 'large':
				unet = UNet_large(1, args.input_channels).to(unet_device)
			else:
				unet = UNet(1, args.input_channels).to(unet_device)
			unetType_model_dir = make_dir(Unets_model_dir, 'unet')

		elif args.unet_type == 'encoder_decoder':
			unet = encoder_decoer(1, args.input_channels)#.to(unet_device)
			unetType_model_dir = make_dir(Unets_model_dir, 'encoder_decoder')

		hyper_dir = make_dir(unetType_model_dir, hyper_name)
		Unet_name = 'fold:{}.ckpt'.format(i)
		final_model_path = os.path.join(hyper_dir, Unet_name)
		
		unet = nn.DataParallel(unet, device_ids=unet_device_ids)

		datasets = {x: CV_Unet_Dataset(subject_paths[x]) for x in ['train', 'val']}
		dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True)
					for x in ['train', 'val']}
		if args.optim == 'Adam':
			optimizer_ft = optim.Adam(unet.parameters(), lr=1e-3)
		elif args.optim == 'SGD':
			optimizer_ft = optim.SGD(unet.parameters(), lr=1e-3, momentum=0.9)

		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)
		
		model = train_unet(unet, dataloaders, optimizer_ft, exp_lr_scheduler, unetType_model_dir, unet_device, bce_weight=args.bce_weight, num_epochs=args.unet_epochs)

		torch.save(model.state_dict(), final_model_path)


if __name__ == '__main__':
	main()

