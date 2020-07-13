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

# my code
from loss import calc_loss
from utils import *
from DSC_pred import DSC_pred

import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from tqdm import tqdm
import argparse
import SimpleITK as sitk
from dataloader import LSTMDataset, UnetDataset, CV_Unet_Dataset, CV_LSTMDataset

# unet_device = torch.device('cuda:{}'.format(3) if torch.cuda.is_available() else 'cpu')

def set_subject_result(GT_dir, test_subjects):
	subject_seg = {}
	for i, subject in enumerate(test_subjects):
		subject = os.path.basename(subject)
		print(subject)
		label_path = os.path.join(GT_dir, subject + '.npy')
		# print(label_path)
		label = np.load(label_path)
		label_arr = np.transpose(label, (2, 0, 1))

		subject_seg[subject] = np.zeros(label_arr.shape, dtype=np.float16)

	return subject_seg

def test_unet(model, dataloaders, root_dir, test_subjects, save_dir, unet_device):
	model.eval()
	epoch_samples = 0
	metrics = defaultdict(float)

	result_path = os.path.join(save_dir, 'testing_result.txt')
	file = open(result_path, 'w')

	# load original shape
	GT_dir = os.path.join(root_dir, 'npy_labels')
	subject_seg = set_subject_result(GT_dir, test_subjects)
	with torch.no_grad():
		for inputs, labels, subject, name in tqdm(dataloaders):
			subject = subject[0]
			name = name[0].replace('.npy', '')

			inputs = inputs.to(unet_device)
			mean = inputs.mean()
			std = inputs.std()
			labels = labels.to(unet_device)
			
			outputs = model(inputs)
			thresh_outputs = F.sigmoid(outputs)
			thresh_outputs[thresh_outputs >= 0.5] = 1.0
			thresh_outputs[thresh_outputs < 0.5] = 0.0

			loss, dice = calc_loss(outputs, labels, metrics)
			pred = np.squeeze(thresh_outputs.cpu().data[0].numpy()).astype(np.float16)
			
			p_z, p_x, p_y = pred.shape
			_, z,x,y = name.split('_')
			z, x, y = int(z), int(x), int(y)
			subject_seg[subject][z:z+p_z,x:x+p_x,y:y+p_y] = (subject_seg[subject][z:z+p_z,x:x+p_x,y:y+p_y] + pred)/2

			epoch_samples += inputs.size(0)
	
	source_dir = '/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA/Labels'
	for key in subject_seg:
		original_path = os.path.join(source_dir, key)

		origin_label = os.path.join(source_dir, 'label{}.nii.gz'.format(key))
		origin_3D = sitk.ReadImage(origin_label, sitk.sitkInt16)

		subject_dir = os.path.join(save_dir, key)
		if not os.path.exists(subject_dir):
			os.mkdir(subject_dir)
		save_path = os.path.join(subject_dir, 'pred.nii')

		subject_seg[key][subject_seg[key] >= 0.4] = 1
		subject_seg[key][subject_seg[key] < 0.4] = 0

		result_3D = sitk.GetImageFromArray(subject_seg[key].astype(np.uint8))

		result_3D.SetDirection(origin_3D.GetDirection())
		result_3D.SetOrigin(origin_3D.GetOrigin())
		result_3D.SetSpacing(origin_3D.GetSpacing())

		sitk.WriteImage(result_3D, save_path)
		del result_3D
	
	for k in metrics.keys():
		file.write('{}: {:.4f}\n'.format(k, metrics[k] / epoch_samples))
	file.close()
	print_metrics(metrics, epoch_samples, 'test')


def Arg():
	parser = argparse.ArgumentParser(description='CT image train')
	parser.add_argument('-d', '--root_dir', dest='root_dir', default='/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA',
						help='set root_dir, default is "/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA"')
	parser.add_argument('-t', '--target', dest='target', default='pancreas',
						help='choose target, liver or pancreas, default is pancreas')
	parser.add_argument('-nf', '--n_fold', dest='n_fold', default=4, type=int,
						help='set number of folds, default is 4')
	parser.add_argument('-g', '--unet_gpu', dest='unet_gpu', default='0',
						help='choose gpu id, 0~7, default=0, if you want use multi-gpu write "0,1,2,3"')
	parser.add_argument('-e', '--unet_epochs', dest='unet_epochs', default=30, type=int,
						help="choose num epoch, default is 30")
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
						help='choose what Unet network training(for using bdclstm), Unet or transition, defulat is transition')
	
	return parser.parse_args()


def main():
	args = Arg()
	root_dir = args.root_dir
	data_root = '../datasets'
	unet_device = torch.device('cuda:{}'.format(args.unet_gpu) if torch.cuda.is_available() else 'cpu') # 'cuda:4,5'

	# split dataset to N_fold
	pre_dir = os.path.join(data_root, 'preprocessed')
	if args.stride_size == 'half':
		data_path = os.path.join(pre_dir, '{}_patch_{}_padding:{}_{}_version'.format(args.target, args.patch_size, args.padding_size, args.version))
		
	else:
		data_path = os.path.join(pre_dir, '{}_patch_{}_padding:{}_stride:{}'.format(args.target, args.patch_size, args.padding_size, args.stride_size))
	fold_list = split_subject_for_cv(data_path, n_fold=args.n_fold)

	model_dir = os.path.join(root_dir, 'models')
	cur_model_dir = make_dir(model_dir, '3D_lstm')	
	Unets_model_dir = make_dir(cur_model_dir, 'Unets_{}'.format(args.optim))

	result_dir = os.path.join(root_dir, 'results')
	cur_result_dir = make_dir(result_dir, '3D_lstm')
	Unets_result_dir = make_dir(cur_result_dir, 'Unets_{}'.format(args.optim))

	unet_device_ids = []
	for id in args.unet_gpu.split(','):
		unet_device_ids.append(int(id))
	torch.cuda.set_device(unet_device_ids[0])

	if args.stride_size == 'half':
		hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_input:{}_version:{}_185'.format('Unet', 
						args.unet_type, args.network_size, args.patch_size, args.unet_epochs, args.padding_size, args.input_channels, args.version)
	else:
		hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_stride:{}_input:{}'.format('Unet', 
						args.unet_type, args.network_size, args.patch_size, args.unet_epochs, args.padding_size, args.stride_size, args.input_channels)
	print(hyper_name)
	for i in range(3, args.n_fold):
		subject_paths = split_fold_to_train(fold_list, i)
		
		print('[{}] fold testing start!'.format(i))

		if args.unet_type == 'transition':
			if args.network_size == 'large':
				unet = transition_UNet_large(1, args.input_channels).to(unet_device)
			else:
				unet = transition_UNet(1, args.input_channels)

			unetType_model_dir = make_dir(Unets_model_dir, 'transition')
			unetType_result_dir = make_dir(Unets_result_dir, 'transition')
		elif args.unet_type == 'unet':
			if args.network_size == 'large':
				unet = UNet_large(1, args.input_channels).to(unet_device)
			else:
				unet = UNet(1, args.input_channels).to(unet_device)
			unetType_model_dir = make_dir(Unets_model_dir, 'unet')
			unetType_result_dir = make_dir(Unets_result_dir, 'unet')
		else:
			unet = encoder_decoer(1, args.input_channels).to(unet_device)
			unetType_model_dir = make_dir(Unets_model_dir, 'encoder_decoder')
			unetType_result_dir = make_dir(Unets_result_dir, 'encoder_decoder')
			
		hyper_dir = os.path.join(unetType_model_dir, hyper_name)
		Unet_name = 'fold:{}.ckpt'.format(i)
		
		unet = nn.DataParallel(unet, device_ids=unet_device_ids)
		model_path = os.path.join(hyper_dir, Unet_name)
		print(model_path)
		unet.load_state_dict(torch.load(model_path))

		datasets = CV_Unet_Dataset(subject_paths['val'])
		dataloader = DataLoader(datasets, batch_size=1, shuffle=True)

		patch_epoch_dir = make_dir(unetType_result_dir, hyper_name)
		fold_result_dir = make_dir(patch_epoch_dir, '{}_fold'.format(i), remove=True)
		test_unet(unet, dataloader, root_dir, subject_paths['val'], fold_result_dir, unet_device)

		# calculate 3D result
		DSC_pred(root_dir, fold_result_dir)
if __name__ == '__main__':
	main()
