import os, sys
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torchsummary import summary
import pytorch_unet
from Unets import transition_UNet, UNet
from CLSTM import BDCLSTM_unet
from without_lstm import no_lstm

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
import shutil

def train_bdclstm(bdclstm, dataloaders, optimizer, scheduler, device, bce_weight, final_model_path, num_epochs=40):
	best_model_wts = copy.deepcopy(bdclstm.state_dict())
	best_loss = 1

	l1_crit = nn.L1Loss(size_average=False)
	factor = 0.0005
	
	for epoch in range(num_epochs):
		print('Epoch {}/{}'.format(epoch+1, num_epochs))
		print('-'*10)

		for phase in ['train', 'val']:
			if phase == 'train':
				scheduler.step()
				for param_group in optimizer.param_groups:
					print('LR', param_group['lr'])
				bdclstm.train()
			else:
				bdclstm.eval()

			metrics = defaultdict(float)
			epoch_samples = 0
			# cnt = 0
			for inputs, labels, _, _ in tqdm(dataloaders[phase]):
				feature_list = []
				# labels = labels[:, 0, :, :, :, :].to(bdclstm_device)
				labels = labels.to(device)
				inputs = inputs.to(device)					
				
				optimizer.zero_grad()
				with torch.set_grad_enabled(phase=='train'):
					outputs = bdclstm(inputs)
					loss, _ = calc_loss(outputs, labels, metrics, bce_weight=bce_weight)
					reg_loss = 0
					# for param in bdclstm.parameters():
					# 	reg_loss += l1_crit(param, torch.zeros_like(param))
					# loss += factor * reg_loss
					if phase == 'train':
						loss.backward()
						optimizer.step()
				epoch_samples += inputs.size(0)
			print_metrics(metrics, epoch_samples, phase)
			
			epoch_loss = metrics['dice'] / epoch_samples
			if phase == 'val' and epoch_loss < best_loss:
				print('best loss changed!')
				best_loss = epoch_loss
				best_model_wts = copy.deepcopy(bdclstm.state_dict())
		
		print('svae temproal model!')
		torch.save(best_model_wts, final_model_path)
		# 	torch.save(model.state_dict(), final_model_path)

	bdclstm.load_state_dict(best_model_wts)
	return bdclstm


def Arg():
	parser = argparse.ArgumentParser(description='CT image train')
	parser.add_argument('-d', '--root_dir', dest='root_dir', default='/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA',
						help='set root_dir, default is "/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA"')
	parser.add_argument('-t', '--target', dest='target', default='pancreas',
						help='choose target, liver or pancreas, default is liver')
	parser.add_argument('-nf', '--n_fold', dest='n_fold', default=4, type=int,
						help='set number of folds, default is 4')
	parser.add_argument('-bg', '--bdclstm_gpu', dest='bdclstm_gpu', default='1',
						help='choose gpu id, 0~7, default=1, if you want use multi-gpu write "4,5,6,7"')

	parser.add_argument('-ue', '--unet_epochs', dest='unet_epochs', default=30, type=int,
						help="choose num epoch, default is 30")
	parser.add_argument('-be', '--bdclstm_epochs', dest='bdclstm_epochs', default=30, type=int,
						help="choose num epoch, default is 30")
	
	parser.add_argument('-b', '--batch_size', dest='batch_size', default=8, type=int,
						help='set batch size')
	parser.add_argument('-p', '--patch_size', dest='patch_size', default=64, type=int,
						help='set patch size')
	parser.add_argument('-v', dest='version', default='old',
						help='old version means clip the intensity value [-300, 350], new version is [-100, 240], default is old')

	parser.add_argument('--network_size', dest='network_size', default='normal',
						help='set network_size. normal or small, default is normal')
	parser.add_argument('--optim', dest='optim', default='Adam',
						help='set optim. SGD or Adam, default is Adam')
	parser.add_argument('--padding_size', dest='padding_size', default='stride',
						help='set padding_size. stride or patch, default is stride')
	parser.add_argument('--stride_size', dest='stride_size', default='half',
						help='set stride_size. half or triple, default is half')
	parser.add_argument('--retrain', dest='retrain', default='False',
						help='set retrain. default is False')
	parser.add_argument('--unet_pretrain', dest='unet_pretrain', default='True',
						help='set unet pretrain mode, default is True')
	parser.add_argument('--unet_parameter_fix', dest='unet_parameter_fix', default='True',
						help='set unet_parameter_fix mode, default is True')                        
	parser.add_argument('--input_channels', dest='input_channels', default=64, type=int,
						help='set input_channels. default is 64')
	parser.add_argument('--hidden_channels', dest='hidden_channels', default=32, type=int,
						help='set hidden_channels. default is 32')
	parser.add_argument('--last_fc', dest='last_fc', default=64, type=int,
						help='set last_fc. default is 64')
	parser.add_argument('--bce_weight', dest='bce_weight', default=0.25, type=float,
						help='set bce_weight. default is 0.25, 0 means only use dice loss')
	parser.add_argument('--unet_type', dest='unet_type', default='transition',
						help='choose what Unet network training(for using bdclstm), Unet or transition, defulat is transition')
	parser.add_argument('--lstm_version', dest='lstm_version', default='1',
						help='set lstm version(1~4), defulat is 1')
	parser.add_argument('--without_lstm', dest='without_lstm', default='False',
						help='set without_lstm, defulat is False')
	return parser.parse_args()


def main():
	args = Arg()
	root_dir = args.root_dir
	data_root = '../datasets'
	device = torch.device('cuda:{}'.format(args.bdclstm_gpu) if torch.cuda.is_available() else 'cpu')

	model_dir = os.path.join(root_dir, 'models')
	cur_model_dir = make_dir(model_dir, '3D_lstm')

	# split dataset to N_fold
	pre_dir = os.path.join(data_root, 'preprocessed')
	if args.stride_size == 'half':
		data_path = os.path.join(pre_dir, '{}_patch_{}_padding:{}_{}_version'.format(args.target, args.patch_size, args.padding_size, args.version))
	else:
		data_path = os.path.join(pre_dir, '{}_patch_{}_padding:{}_stride:{}'.format(args.target, args.patch_size, args.padding_size, args.stride_size))
	fold_list = split_subject_for_cv(data_path, n_fold=args.n_fold)

	bdclstm_device_ids = []
	for id in args.bdclstm_gpu.split(','):
		bdclstm_device_ids.append(int(id))
	# torch.cuda.set_device(unet_device_ids[0])

	# Unets_model_dir = make_dir(cur_model_dir, 'Unets_{}'.format(args.optim))
	Unets_model_dir = make_dir(cur_model_dir, 'Unets_{}'.format('Adam'))
	BDClstm_model_dir = make_dir(cur_model_dir, 'BDClstm_{}'.format(args.optim))

	if args.stride_size == 'half':
		Unet_hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_input:{}_version:{}'.format('Unet', 
						args.unet_type, args.network_size, args.patch_size, args.unet_epochs, 'stride', args.input_channels, args.version)
	else:
		Unet_hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_stride:{}_input:{}'.format('Unet', 
						args.unet_type, args.network_size, args.patch_size, args.unet_epochs, args.padding_size, args.stride_size, args.input_channels)
	if args.stride_size == 'half':
		BDClstm_hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_input:{}_hidden:{}_lastfc{}:UnetPretrain:{}_UnetParameterFix:{}_version:{}_withoutlstm:{}'.format('BDClstm', 
			args.unet_type, args.network_size, args.patch_size, args.bdclstm_epochs, args.padding_size, args.input_channels, args.hidden_channels, args.last_fc,
			args.unet_pretrain, args.unet_parameter_fix, args.version, args.without_lstm)
	else:
		BDClstm_hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_stride:{}_input:{}_hidden:{}_lastfc{}:UnetPretrain:{}_UnetParameterFix:{}_185'.format('BDClstm', 
			args.unet_type, args.network_size, args.patch_size, args.bdclstm_epochs, args.padding_size, args.stride_size, args.input_channels, args.hidden_channels, args.last_fc,
			args.unet_pretrain, args.unet_parameter_fix)

	if args.unet_type == 'transition':
		unetType_model_dir = os.path.join(Unets_model_dir, 'transition')
		unetType_bdclstm_dir = make_dir(BDClstm_model_dir, 'transition')
	else:
		unetType_model_dir = os.path.join(Unets_model_dir, 'unet')
		unetType_bdclstm_dir = make_dir(BDClstm_model_dir, 'unet')
	model_root = make_dir(unetType_bdclstm_dir, BDClstm_hyper_name)
	version_dir = make_dir(model_root, 'lstm_version:{}'.format(args.lstm_version))
	print('unet hyper:', Unet_hyper_name)
	print('lstm hyper:', BDClstm_hyper_name)

	epoch = args.bdclstm_epochs
	for i in range(args.n_fold):
		subject_paths = split_fold_to_train(fold_list, i)
		fold_name = 'fold:{}.ckpt'.format(i)
		
		print('[{}] fold training start!'.format(i))
		
		final_model_path = os.path.join(version_dir, fold_name)

		if args.unet_pretrain == 'True' and args.retrain == 'False':
			unet_path = os.path.join(unetType_model_dir, Unet_hyper_name, fold_name)
		else:
			unet_path = None
		
		if args.without_lstm == 'False':
			bdclstm = BDCLSTM_unet(args.unet_type, args.network_size, unet_path, input_channels=args.input_channels, 
									hidden_channels=[args.hidden_channels], last_fc=args.last_fc, num_classes=1, 
									pretrained=args.unet_pretrain, parameter_fix=args.unet_parameter_fix, version=args.lstm_version).to(device)
		else:
			bdclstm = no_lstm(args.unet_type, args.network_size, bdclstm_device_ids, unet_path=unet_path, input_channels=args.input_channels, 
									hidden_channels=args.hidden_channels, last_fc=args.last_fc, num_classes=1, 
									pretrained=args.unet_pretrain, parameter_fix=args.unet_parameter_fix, version=args.lstm_version).to(device)
		bdclstm = nn.DataParallel(bdclstm, device_ids=bdclstm_device_ids)
		if args.retrain == 'True':
			print("retrain BDCLSTM!")
			bdclstm.load_state_dict(torch.load(final_model_path))
			epoch = 20

		datasets = {x: CV_LSTMDataset(subject_paths[x], patch_size=args.patch_size, stride_size=args.stride_size) for x in ['train', 'val']}
		dataloaders = {x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=False)
					for x in ['train', 'val']}
		if args.optim == 'Adam':
			optimizer_ft = optim.Adam(bdclstm.parameters(), lr=1e-3)
		else:
			optimizer_ft = optim.SGD(bdclstm.parameters(), lr=1e-3, momentum=0.9)
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

		model = train_bdclstm(bdclstm, dataloaders, optimizer_ft, exp_lr_scheduler, device, args.bce_weight, final_model_path, num_epochs=epoch)
		
		torch.save(model.state_dict(), final_model_path)
		sys.exit()

if __name__ == '__main__':
	main()

