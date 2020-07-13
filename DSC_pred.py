import os, sys
import matplotlib.pyplot as plt
import numpy as np

import argparse

import shutil

import SimpleITK as sitk
import glob

source_dir = '/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA'
# predict_dir = os.path.join(source_dir, 'results/3D_lstm/Unets/unet')
# predict_dir = os.path.join(source_dir, 'results/3D_lstm/Unets/transition/patch_size:32_epochs:30/0_fold')
# predict_dir = os.path.join(source_dir, 'results/3D_lstm/Unets/transition/network:Unet_unettype:transition_patch_size:48_epochs:30_padding:stride_input:64_BESTRESULT/0_fold')
# predict_dir = os.path.join(source_dir, 'results/3D_lstm/Unets/transition/patch_size:48_epochs:30/0_fold')

def precision(pred, target, smooth = 0.01):
	tp = (pred*target).sum(-1).sum(-1).sum(-1)
	tp_plus_fp = pred.sum(-1).sum(-1).sum(-1)

	prec = (tp + smooth) / (tp_plus_fp + smooth)

	return prec

def recall(pred, target, smooth = 0.01):
	tp = (pred*target).sum(-1).sum(-1).sum(-1)
	tp_plus_fn = target.sum(-1).sum(-1).sum(-1)

	rec = (tp + smooth) / (tp_plus_fn + smooth)

	return rec

def dice_score(pred, target, smooth = 1.):
	intersection = (pred * target).sum(-1).sum(-1).sum(-1)
	union = (pred + target).sum(-1).sum(-1).sum(-1)

	dice = (2. * intersection + smooth) / (union + smooth)
	print(dice.mean())
	return dice.mean()

def DSC_pred(label_dir, pred_dir):
	# pred_dir = os.path.join(pred_dir, '{}_fold'.format(n_fold))
	subject_list = sorted(os.listdir(pred_dir))
	
	result_path = os.path.join(pred_dir, '3D_result2.txt')
	file = open(result_path, 'w')

	# total_dsc, cnt = 0.0, 0
	dsc_list = []
	precision_list = []
	recall_list = []
	for subject in subject_list:
		if os.path.isdir(os.path.join(pred_dir, subject)):
			print(subject)
			predict_path = os.path.join(pred_dir, subject, 'pred.nii')
			source_path = os.path.join(source_dir, 'Labels', 'label{}.nii.gz'.format(subject))

			pred = sitk.ReadImage(predict_path)
			label = sitk.ReadImage(source_path)

			pred_arr = sitk.GetArrayFromImage(pred)
			label_arr = sitk.GetArrayFromImage(label)

			dsc = dice_score(pred_arr, label_arr)
			prec = precision(pred_arr, label_arr)
			rec = recall(pred_arr, label_arr)

			print('dsc :', dsc)
			print('prec:', prec)
			print('rec :', rec)
			# sys.exit()
			dsc_list.append(dsc)
			precision_list.append(prec)
			recall_list.append(rec)
			file.write('{}: {:.4f}, {:.4f}, {:.4f}\n'.format(subject, dsc, prec, rec))
	mean_dsc = np.array(dsc_list).mean()
	mean_prec = np.array(precision_list).mean()
	mean_rec = np.array(recall_list).mean()
	file.write('total: {:.4f}, {:.4f}, {:.4f}\n'.format(mean_dsc, mean_prec, mean_rec))
	print('total_dsc:', mean_dsc)
	print('total_precision:', mean_prec)
	print('total_recall:', mean_rec)
	std_dsc = np.array(dsc_list).std()
	std_prec = np.array(precision_list).std()
	std_rec = np.array(recall_list).std()
	file.write('std: {:.4f}, {:.4f}, {:.4f}'.format(std_dsc, std_prec, std_rec))
	print('std_dsc:', std_dsc)
	print('std_precision:', std_prec)
	print('std_recall:', std_rec)
	

def main():
	fold_mean_dsc = 0.0
	fold_mean_prec = 0.0
	fold_mean_rec = 0.0

	fold_std_dsc = 0.0
	fold_std_prec = 0.0
	fold_std_rec = 0.0

	for i in range(4):
		print('fold : {}'.format(i))
		# predict_dir = os.path.join(source_dir, 'results/3D_lstm/BDClstm/transition/network:BDClstm_unettype:transition_unetsize:normal_patch_size:32_epochs:30_padding:stride_input:64_hidden:32_previous/{}_fold'.format(i))
		# predict_dir = os.path.join(source_dir, 'results/3D_lstm/Unets/transition/network:Unet_unettype:transition_patch_size:48_epochs:30_padding:stride_input:64_BESTRESULT/{}_fold'.format(i))
		predict_dir = os.path.join(source_dir, 'results/3D_lstm/Unets/transition/network:Unet_unettype:transition_patch_size:32_epochs:30_padding:stride_input:64/{}_fold'.format(i))
		DSC_pred(source_dir, predict_dir)

		fold_mean_dsc += mean_dsc
		fold_mean_prec +=  mean_prec
		fold_mean_rec +=  mean_rec

		fold_std_dsc +=  std_dsc
		fold_std_prec +=  std_prec
		fold_std_rec +=  std_rec

if __name__ == "__main__":
	main()