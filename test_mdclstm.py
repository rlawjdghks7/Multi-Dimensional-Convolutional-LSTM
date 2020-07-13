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

import torch.nn as nn

from collections import defaultdict
import torch.nn.functional as F

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
from dataloader import CV_LSTMDataset


def set_subject_result(GT_dir, test_subjects):
	subject_seg = {}
	for i, subject in enumerate(test_subjects):
		subject = os.path.basename(subject)
		print(subject)
		label_path = os.path.join(GT_dir, subject + '.npy')
		label = np.load(label_path)
		label_arr = np.transpose(label, (2, 0, 1))

		subject_seg[subject] = np.zeros(label_arr.shape, dtype=np.float16)

	return subject_seg

def test_bdclstm(bdclstm, dataloaders, root_dir, test_subjects, device, save_dir):
	bdclstm.eval()

	GT_dir = os.path.join(root_dir, 'npy_labels')
	subject_seg = set_subject_result(GT_dir, test_subjects)

	metrics = defaultdict(float)
	epoch_samples = 0
	# cnt = 0
	with torch.no_grad():
		for inputs, labels, subject, name in tqdm(dataloaders):
			subject = subject[0]
			name = name[0].replace('.npy', '')
			
			labels = labels.to(device)
			inputs = inputs.to(device)
			
			outputs = bdclstm(inputs)

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
		print_metrics(metrics, epoch_samples, 'test')
	source_dir = os.path.join(root_dir, 'Labels')
	for key in subject_seg:
		original_path = os.path.join(source_dir, key)

		origin_label = os.path.join(source_dir, 'label{}.nii.gz'.format(key))
		origin_3D = sitk.ReadImage(origin_label, sitk.sitkInt16)

		subject_dir = make_dir(save_dir, key)
		save_path = os.path.join(subject_dir, 'pred.nii')

		subject_seg[key][subject_seg[key] >= 0.4] = 1
		subject_seg[key][subject_seg[key] < 0.4] = 0

		result_3D = sitk.GetImageFromArray(subject_seg[key].astype(np.uint8))

		result_3D.SetDirection(origin_3D.GetDirection())
		result_3D.SetOrigin(origin_3D.GetOrigin())
		result_3D.SetSpacing(origin_3D.GetSpacing())

		sitk.WriteImage(result_3D, save_path)
		del result_3D

	return bdclstm


def Arg():
	parser = argparse.ArgumentParser(description='CT image train')
	parser.add_argument('-d', '--root_dir', dest='root_dir', default='/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA',
						help='set root_dir, default is "/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA"')
	parser.add_argument('-t', '--target', dest='target', default='pancreas',
						help='choose target, liver or pancreas, default is liver')
	parser.add_argument('-nf', '--n_fold', dest='n_fold', default=4, type=int,
						help='set number of folds, default is 4')
	parser.add_argument('-bg', '--bdclstm_gpu', dest='bdclstm_gpu', default='0',
						help='choose gpu id, 0~7, default=1, if you want use multi-gpu write "4,5,6,7"')

	parser.add_argument('-be', '--bdclstm_epochs', dest='bdclstm_epochs', default=30, type=int,
						help="choose num epoch, default is 5")
	
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
	torch.cuda.set_device(bdclstm_device_ids[0])

	model_dir = os.path.join(root_dir, 'models')
	cur_model_dir = make_dir(model_dir, '3D_lstm')
	BDClstm_model_dir = make_dir(cur_model_dir, 'BDClstm_{}'.format(args.optim))

	result_dir = os.path.join(root_dir, 'results')
	cur_result_dir = make_dir(result_dir, '3D_lstm')
	BDClstm_result_dir = make_dir(cur_result_dir, 'BDClstm_{}'.format(args.optim))

	if args.stride_size == 'half':
		BDClstm_hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_input:{}_hidden:{}_lastfc{}:UnetPretrain:{}_UnetParameterFix:{}_version:{}_withoutlstm:{}'.format('BDClstm', 
			args.unet_type, args.network_size, args.patch_size, args.bdclstm_epochs, args.padding_size, args.input_channels, args.hidden_channels, args.last_fc,
			args.unet_pretrain, args.unet_parameter_fix, args.version, args.without_lstm)
	else:
		BDClstm_hyper_name = 'network:{}_unettype:{}_unetsize:{}_patch_size:{}_epochs:{}_padding:{}_stride:{}_input:{}_hidden:{}_lastfc{}:UnetPretrain:{}_UnetParameterFix:{}'.format('BDClstm', 
			args.unet_type, args.network_size, args.patch_size, args.bdclstm_epochs, args.padding_size, args.stride_size, args.input_channels, args.hidden_channels, args.last_fc,
			args.unet_pretrain, args.unet_parameter_fix)

	if args.unet_type == 'transition':
		unetType_bdclstm_dir = make_dir(BDClstm_model_dir, 'transition')
		unetType_result_dir = make_dir(BDClstm_result_dir, 'transition')
	else:
		unetType_bdclstm_dir = make_dir(BDClstm_model_dir, 'unet')
		unetType_result_dir = make_dir(BDClstm_result_dir, 'unet')

	for i in range(args.n_fold):
		subject_paths = split_fold_to_train(fold_list, i)
		fold_name = 'fold:{}.ckpt'.format(i)
		print('[{}] fold testing start!'.format(i))

		# load bdclstm model
		bdclstm = BDCLSTM_unet(args.unet_type, args.network_size, None, input_channels=args.input_channels, 
								hidden_channels=[args.hidden_channels], last_fc=args.last_fc, num_classes=1, 
								pretrained=args.unet_pretrain, parameter_fix=args.unet_parameter_fix, version=args.lstm_version).to(device)
		bdclstm = nn.DataParallel(bdclstm, device_ids=bdclstm_device_ids)
		bdclstm_model_path = os.path.join(unetType_bdclstm_dir, BDClstm_hyper_name, 'lstm_version:{}'.format(args.lstm_version) ,fold_name)
		bdclstm.load_state_dict(torch.load(bdclstm_model_path))

		datasets = CV_LSTMDataset(subject_paths['val'], patch_size=args.patch_size, stride_size=args.stride_size)
		dataloaders = DataLoader(datasets, batch_size=1, shuffle=True)
		patch_epoch_dir = make_dir(unetType_result_dir, BDClstm_hyper_name)
		version_dir = make_dir(patch_epoch_dir, 'lstm_version:{}'.format(args.lstm_version))
		fold_result_dir = make_dir(version_dir, fold_name.replace('.ckpt', ''), remove=True)

		test_bdclstm(bdclstm, dataloaders, root_dir, subject_paths['val'], device, fold_result_dir)
		DSC_pred(root_dir, fold_result_dir)


if __name__ == '__main__':
	main()

