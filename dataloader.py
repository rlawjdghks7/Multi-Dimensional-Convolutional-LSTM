import torch
from torch.utils.data import Dataset
import torchvision
import pydicom
import os
import sys
import numpy as np
import glob

import SimpleITK as sitk
from PIL import Image
import matplotlib.pyplot as plt

class CV_Unet_Dataset(Dataset):
	def __init__(self, subject_list):
		img_dir = 'images'
		label_dir = 'labels'

		self.img_paths = []
		self.label_paths = []
		# self.file = open('check_result.txt', 'w')
		
		for sub_path in subject_list:
			img_dir_path = os.path.join(sub_path, img_dir)
			label_dir_path = os.path.join(sub_path, label_dir)

			imgs = sorted(os.listdir(img_dir_path))

			for img in imgs:
				self.img_paths.append(os.path.join(img_dir_path, img))
				self.label_paths.append(os.path.join(label_dir_path, img))
		print("total images:", len(self.img_paths))
	def __getitem__(self, idx):
		subject_name = self.img_paths[idx].split('/')[-3]
		name = os.path.basename(self.img_paths[idx])

		img_np = np.load(self.img_paths[idx])
		label_np = np.load(self.label_paths[idx]).astype(np.uint8)
		# self.file.write('{} mean:{:.4f}, std:{:.4f}\n'.format(name, img_np.mean(), img_np.std()))
		img = torch.FloatTensor(img_np).unsqueeze(0)
		label = torch.FloatTensor(label_np).unsqueeze(0)

		return img, label, subject_name, name

	def __len__(self):
		return len(self.img_paths)


class CV_LSTMDataset(Dataset):
	def __init__(self, subject_list, patch_size=64, stride_size='half'):
		self.patch_size = patch_size
		if stride_size == 'half':
			self.stride = patch_size // 2
		elif stride_size == 'triple':
			self.stride = patch_size // 3

		img_dir = 'images'
		label_dir = 'labels'

		self.img_paths = []
		self.label_paths = []
		for sub_path in subject_list:
			# load subject shape or find min, max
			img_dir_path = os.path.join(sub_path, img_dir)
			label_dir_path = os.path.join(sub_path, label_dir)

			imgs = sorted(os.listdir(img_dir_path))
			z, x, y = self.find_min_max(imgs)
			for img in imgs:
				_, img_z, img_x, img_y = img.replace('.npy', '').split('_')
				img_z, img_x, img_y = int(img_z), int(img_x), int(img_y)
				if img_z > z['min'] and img_z < z['max'] and img_x > x['min'] and img_x < x['max'] and img_y > y['min'] and img_y < y['max']:
					self.img_paths.append(os.path.join(img_dir_path, img))
					self.label_paths.append(os.path.join(label_dir_path, img))

	def __getitem__(self, idx):
		subject_name = self.img_paths[idx].split('/')[-3]

		img_path = self.img_paths[idx]
		label_path = self.label_paths[idx]

		img_name = os.path.basename(img_path)

		_, img_z, img_x, img_y = img_name.replace('.npy', '').split('_')
		img_z, img_x, img_y = int(img_z), int(img_x), int(img_y)
		imgs = []
		labels = []
		# 0 : img_center
		# 1 : img_center x+1, 2 : img_center x-1
		# 3 : img_center y+1, 4 : img_center y-1
		# 5 : img_center z+1, 6 : img_center z-1
		imgs.append(np.load(img_path))
		label = np.load(label_path).astype(np.uint8)
		labels.append(np.load(label_path).astype(np.uint8))

		for x in [1, -1]:
			try:
				new_img_name = 'patch_{}_{}_{}.npy'.format(
					img_z, img_x+x*self.stride, img_y)
				new_img = np.load(img_path.replace(img_name, new_img_name)).astype(np.float64)
				new_label = np.load(label_path.replace(img_name, new_img_name)).astype(np.uint8)
				imgs.append(new_img)
				labels.append(new_label)
			except:
				imgs.append(np.zeros_like(imgs[0]).astype(np.float64))
				label.append(np.zeros_like(labels[0]).astype(np.float64))

		for y in [1, -1]:
			try:
				new_img_name = 'patch_{}_{}_{}.npy'.format(
					img_z, img_x, img_y+y*self.stride)
				new_img = np.load(img_path.replace(img_name, new_img_name)).astype(np.float64)
				new_label = np.load(label_path.replace(img_name, new_img_name)).astype(np.uint8)
				imgs.append(new_img)
				labels.append(new_label)
			except:
				imgs.append(np.zeros_like(imgs[0]).astype(np.float64))
				label.append(np.zeros_like(labels[0]).astype(np.float64))

		for z in [1, -1]:
			try:
				new_img_name = 'patch_{}_{}_{}.npy'.format(
					img_z+z*self.stride, img_x, img_y)
				new_img = np.load(img_path.replace(img_name, new_img_name)).astype(np.float64)
				new_label = np.load(label_path.replace(img_name, new_img_name)).astype(np.uint8)
				imgs.append(new_img)
				labels.append(new_label)
			except:
				imgs.append(np.zeros_like(imgs[0]).astype(np.float64))
				label.append(np.zeros_like(labels[0]).astype(np.float64))
		# print(subject_name)

		imgs_np = np.array(imgs).astype(np.float64)
		imgs_torch = torch.FloatTensor(imgs_np).unsqueeze(1) # 7 x 1 x patch_size x patch_size x patch_size
		# labels_torch = torch.FloatTensor(labels_np).unsqueeze(1)
		label_torch = torch.FloatTensor(label).unsqueeze(0) # 1 x 7 x patch_size x patch_size x patch_size ????

		return imgs_torch, label_torch, subject_name, img_name
		# return imgs_torch, labels_torch, subject_name, img_name

	def __len__(self):
		return len(self.img_paths)

	def find_min_max(self, imgs):
		z = {'min':9999, 'max':0}
		x = {'min':9999, 'max':0}
		y = {'min':9999, 'max':0}

		# name : patch_z_x_y.npy
		for img in imgs:
			_, z_temp, x_temp, y_temp = img.replace('.npy', '').split('_')

			z = self.cmp_val(int(z_temp), z)
			x = self.cmp_val(int(x_temp), x)
			y = self.cmp_val(int(y_temp), y)
		return z, x, y

	def cmp_val(self, temp, dict):
		# print(dict['min'])
		if temp < dict['min']:
			dict['min'] = temp
		if temp > dict['max']:
			dict['max'] = temp

		return dict


class LSTMDataset(Dataset):
	def __init__(self, root_dir='/media/NAS/nas_187/datasets/junghwan/experience/CT/TCIA/preprocessed',
				mode='train', target_label='pancreas', patch_size=64):
		self.target_label = target_label
		self.patch_size = patch_size
		self.stride = patch_size // 2
		root_dir = os.path.join(root_dir, '{}_patch_{}_half'.format(target_label, patch_size))
		mode_dir = os.path.join(root_dir, mode)

		subject_list = os.listdir(mode_dir)

		img_dir = 'images'
		label_dir = 'labels'

		# 0 : -1, 1 : 0, 2 : 1, shift
		# original : 1_1_1

		self.img_paths = []
		self.label_paths = []
		for subject in subject_list:
			# load subject shape or find min, max

			sub_path = os.path.join(mode_dir, subject)
			img_dir_path = os.path.join(sub_path, img_dir)
			label_dir_path = os.path.join(sub_path, label_dir)

			imgs = sorted(os.listdir(img_dir_path))
			# labels = sorted(os.listdir(label_dir_path))
			# print(subject)
			z, x, y = self.find_min_max(imgs)
			# sys.exit()
			# print(len(imgs))
			for img in imgs:
				_, img_z, img_x, img_y = img.replace('.npy', '').split('_')
				img_z, img_x, img_y = int(img_z), int(img_x), int(img_y)
				# print(img_z, img_x, img_y)
				if img_z > z['min'] and img_z < z['max'] and img_x > x['min'] and img_x < x['max'] and img_y > y['min'] and img_y < y['max']:
					self.img_paths.append(os.path.join(img_dir_path, img))
					self.label_paths.append(os.path.join(label_dir_path, img))
			# print(len(self.img_paths))
			# sys.exit()
			# for label in labels:
			# 	self.label_paths.append(os.path.join(label_dir_path, label))

	def __getitem__(self, idx):
		subject_name = self.img_paths[idx].split('/')[-3]
		# print(subject_name)
		img_path = self.img_paths[idx]
		label_path = self.label_paths[idx]

		img_name = os.path.basename(img_path)

		_, img_z, img_x, img_y = img_name.replace('.npy', '').split('_')
		img_z, img_x, img_y = int(img_z), int(img_x), int(img_y)
		imgs = []
		labels = []
		# 0 : img_center
		# 1 : img_center x+1, 2 : img_center x-1
		# 3 : img_center y+1, 4 : img_center y-1
		# 5 : img_center z+1, 6 : img_center z-1
		imgs.append(np.load(img_path))
		label = np.load(label_path).astype(np.uint8)
		labels.append(np.load(label_path).astype(np.uint8))

		for x in [1, -1]:
			new_img_name = 'patch_{}_{}_{}.npy'.format(
				img_z, img_x+x*self.stride, img_y)
			new_img = np.load(img_path.replace(img_name, new_img_name))
			new_label = np.load(label_path.replace(img_name, new_img_name)).astype(np.uint8)
			imgs.append(new_img)
			labels.append(new_label)

		for y in [1, -1]:
			new_img_name = 'patch_{}_{}_{}.npy'.format(
				img_z, img_x, img_y+y*self.stride)
			new_img = np.load(img_path.replace(img_name, new_img_name))
			new_label = np.load(label_path.replace(img_name, new_img_name)).astype(np.uint8)
			imgs.append(new_img)
			labels.append(new_label)

		for z in [1, -1]:
			new_img_name = 'patch_{}_{}_{}.npy'.format(
				img_z+z*self.stride, img_x, img_y)
			new_img = np.load(img_path.replace(img_name, new_img_name))
			new_label = np.load(label_path.replace(img_name, new_img_name)).astype(np.uint8)
			imgs.append(new_img)
			labels.append(new_label)

		imgs_np = np.array(imgs)
		labels_np = np.array(labels).astype(np.uint8)

		imgs_torch = torch.FloatTensor(imgs_np).unsqueeze(1) # 7 x 1 x patch_size x patch_size x patch_size
		# labels_torch = torch.FloatTensor(labels_np).unsqueeze(1)
		label_torch = torch.FloatTensor(label).unsqueeze(0)

		return imgs_torch, label_torch, subject_name, img_name
		# return imgs_torch, labels_torch, subject_name, img_name

	def __len__(self):
		return len(self.img_paths)

	def find_min_max(self, imgs):
		z = {'min':9999, 'max':0}
		x = {'min':9999, 'max':0}
		y = {'min':9999, 'max':0}

		# name : patch_z_x_y.npy
		for img in imgs:
			_, z_temp, x_temp, y_temp = img.replace('.npy', '').split('_')

			z = self.cmp_val(int(z_temp), z)
			x = self.cmp_val(int(x_temp), x)
			y = self.cmp_val(int(y_temp), y)
		# print(z)
		# print(x)
		# print(y)
		return z, x, y

	def cmp_val(self, temp, dict):
		# print(dict['min'])
		if temp < dict['min']:
			dict['min'] = temp
		if temp > dict['max']:
			dict['max'] = temp

		return dict


class UnetDataset(Dataset):
	def __init__(self, root_dir, pre_dir='preprocessed',
				mode='train', target_label='pancreas', patch_size=64):
		self.target_label = target_label
		data_dir = os.path.join(root_dir, pre_dir, '{}_patch_{}'.format(target_label, patch_size))
		mode_dir = os.path.join(data_dir, mode)

		subject_list = os.listdir(mode_dir)

		img_dir = 'images'
		label_dir = 'labels'

		self.img_paths = []
		self.label_paths = []
		for subject in subject_list:
			sub_path = os.path.join(mode_dir, subject)
			img_dir_path = os.path.join(sub_path, img_dir)
			label_dir_path = os.path.join(sub_path, label_dir)

			imgs = sorted(os.listdir(img_dir_path))

			for img in imgs:
				self.img_paths.append(os.path.join(img_dir_path, img))
				self.label_paths.append(os.path.join(label_dir_path, img))

	def __getitem__(self, idx):
		subject_name = self.img_paths[idx].split('/')[-3]
		name = os.path.basename(self.img_paths[idx])

		img_np = np.load(self.img_paths[idx])
		label_np = np.load(self.label_paths[idx]).astype(np.uint8)

		img = torch.FloatTensor(img_np).unsqueeze(0)
		label = torch.FloatTensor(label_np).unsqueeze(0)

		return img, label, subject_name, name

	def __len__(self):
		return len(self.img_paths)

if __name__ == "__main__":
	dataset = LSTMDataset()
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

	for imgs, labels, _, name in dataloader:
		print(imgs.size())
		sys.exit()