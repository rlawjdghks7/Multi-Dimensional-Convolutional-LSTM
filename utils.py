import numpy as np
import os
import random
import shutil
import glob

def reverse_transform(inp):
	inp = inp.numpy().transpose((1, 2, 0))
	inp = np.clip(inp, 0, 1)
	inp = (inp * 255).astype(np.uint8)

	return inp

def print_metrics(metrics, epoch_samples, phase):
	outputs = []
	for k in metrics.keys():
		outputs.append("{}: {:.4f}".format(k, metrics[k] / epoch_samples))
	print("{}: {}".format(phase, ", ".join(outputs)))

def make_dir(root_dir, target_dir, remove=False):
	result_dir = os.path.join(root_dir, target_dir)
	if not os.path.exists(result_dir):
		os.mkdir(result_dir)
	elif remove == True:
		shutil.rmtree(result_dir)
		os.mkdir(result_dir)

	return result_dir

def split_subject_for_cv(target_dir, n_fold=4): # return train, val subject path
	subject_split = []

	for i in range(n_fold):
		fold = []
		fold_dir = os.path.join(target_dir, 'fold_{}'.format(i))

		subject_list = os.listdir(fold_dir)
		for subject in subject_list:
			subject_path = os.path.join(fold_dir, subject)
			fold.append(subject_path)
		subject_split.append(fold)

	return subject_split


def split_subject_for_cv_old_version(target_dir, n_fold=4): # return train, val subject path
	random.seed(1)
	subject_split = []

	all_subject_path = []
	for mode in ['train', 'val', 'test']:
		mode_dir = os.path.join(target_dir, mode)

		subject_list = os.listdir(mode_dir)
		for subject in subject_list:
			subject_path = os.path.join(mode_dir, subject)
			all_subject_path.append(subject_path)

	subject_len = len(all_subject_path)
	
	# print(subject_len)
	left_one = subject_len % n_fold
	# print(left_one)
	if left_one:
		fold_size = int(subject_len / n_fold) + 1
	else:
		fold_size = int(subject_len / n_fold)

	# sys.exit()
	for i in range(n_fold):
		fold = []
		while len(fold) < fold_size:
			index = random.randrange(len(all_subject_path))
			fold.append(all_subject_path.pop(index))
		subject_split.append(fold)
		left_one -= 1
		if left_one == 0:
			fold_size -= 1

	return subject_split

def split_fold_to_train(fold_list, flag): # flag means validation set
	result_set = {'train':[], 'val':[]}
	for i, fold in enumerate(fold_list):
		if i == flag:
			result_set['val'] += fold
		else:
			result_set['train'] += fold
	return result_set