import numpy as np 
import matplotlib.pyplot as plt 
import re
import sys
import os


mac38 = 0.08349 * 1e9
mac74 = 0.17036 * 1e9
step = 120

def read_record(fname):
	with open(fname, 'r') as log:
		content = log.readlines()
		train_loss = []
		test_acc = []
		for line in content:
			loss = float(line.split(',')[0][:-2])
			acc = float(line.split(',')[2][:-2])
			train_loss.append(loss)
			test_acc.append(acc)

		train_loss[-1] = train_loss[-2]

	return train_loss, test_acc


def get_baseline_info(fname1, fname2, fname3):
	train_loss_ibm, test_acc_ibm = read_record(fname1)
	train_loss_dorefa, test_acc_dorefa = read_record(fname2)
	train_loss_wageubn, test_acc_wageubn = read_record(fname3)

	epoch_list = np.arange(1,min(len(train_loss_ibm),len(train_loss_dorefa),len(train_loss_wageubn))+1)

	cc_list_ibm = 1/16 * 50000 * mac38 * 3 * np.arange(1, len(train_loss_ibm)+1)
	cc_list_dorefa = 5.34/100 * 50000 * mac38 * 3 * np.arange(1, len(train_loss_dorefa)+1)
	cc_list_wageubn = 1/16 * 50000 * mac38 * 3 * np.arange(1, len(train_loss_wageubn)+1)

	return train_loss_ibm, train_loss_dorefa, train_loss_wageubn, test_acc_ibm, test_acc_dorefa, test_acc_wageubn, cc_list_ibm, cc_list_dorefa, cc_list_wageubn


def get_acc_list(fname):
	acc_list = []
	with open(fname, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if '*' in line and 'Full' not in line:
				acc = re.findall('\d+\.\d+', line)
				acc_list.append(float(acc[0]))
	return acc_list


def get_acc_cc_list(fname1, fname2, fname3, fname4):

	acc_list1 = get_acc_list(fname1)
	acc_list2 = get_acc_list(fname2)
	acc_list3 = get_acc_list(fname3)
	acc_list4 = get_acc_list(fname4)

	min_epoch = min(len(acc_list1),len(acc_list2), len(acc_list3), len(acc_list4))
	epoch_list = np.arange(1, min_epoch+1)

	acc_list1 = acc_list1[:min_epoch]
	acc_list2 = acc_list2[:min_epoch]
	acc_list3 = acc_list3[:min_epoch]
	acc_list4 = acc_list4[:min_epoch]

	cc_list1 = []
	for i in epoch_list:
		if i<20:
			cp = 1
		elif i < 40:
			cp = 1.5
		elif i < 60:
			cp = 2.5
		else:
			cp = 3
		if i == 1:
			cc_list1.append(cp/100 * 50000 * mac38 * 3)
		else:
			cc_list1.append(cp/100 * 50000 * mac38 * 3 + cc_list1[i-2])


	cc_list2 = []
	for i in epoch_list:
		if i<20:
			cp = 2
		elif i < 40:
			cp = 2.5
		elif i < 60:
			cp = 3
		else:
			cp = 3.5
		if i == 1:
			cc_list2.append(cp/100 * 50000 * mac38 * 3)
		else:
			cc_list2.append(cp/100 * 50000 * mac38 * 3 + cc_list2[i-2])


	cc_list3 = 3/100 * 50000 * mac38 * 3 * epoch_list 
	cc_list4 = 4.3/100 * 50000 * mac38 * 3 * epoch_list

	return cc_list1, cc_list2, cc_list3, cc_list4, acc_list1, acc_list2, acc_list3, acc_list4


def get_loss_list(fname):
	loss_list = []
	with open(fname, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if 'Iter' in line and 'Loss' in line:
				loss = re.findall('\d+\.\d+', line)[4]
				loss_list.append(float(loss))
	return loss_list

def get_loss_cc_list(fname1, fname2, fname3, fname4):
	loss_list1 = get_loss_list(fname1)[::step]
	loss_list2 = get_loss_list(fname2)[::step]
	loss_list3 = get_loss_list(fname3)[::step]
	loss_list4 = get_loss_list(fname4)[::step]

	min_epoch = min(len(loss_list1),len(loss_list2), len(loss_list3), len(loss_list4))
	iter_list = np.arange(1, min_epoch+1) * step

	loss_list1 = loss_list1[:min_epoch]
	loss_list2 = loss_list2[:min_epoch]
	loss_list3 = loss_list3[:min_epoch]
	loss_list4 = loss_list4[:min_epoch]

	cc_list1 = []
	for i in range(len(iter_list)):
		if iter_list[i]<8000:
			cp = 1
		elif iter_list[i] < 16000:
			cp = 1.5
		elif iter_list[i] < 24000:
			cp = 2.5
		else:
			cp = 3
		if i == 0:
			cc_list1.append(cp/100 * 128 * mac38 * 3 * step * 10)
		else:
			cc_list1.append(cp/100 * 128 * mac38 * 3 * step * 10 + cc_list1[i-1])


	cc_list2 = []
	for i in range(len(iter_list)):
		if iter_list[i]<8000:
			cp = 2
		elif iter_list[i] < 16000:
			cp = 2.5
		elif iter_list[i] < 24000:
			cp = 3
		else:
			cp = 3.5
		if i == 0:
			cc_list2.append(cp/100 * 128 * mac38 * 3 * step * 10)
		else:
			cc_list2.append(cp/100 * 128 * mac38 * 3 * step * 10 + cc_list2[i-1])


	cc_list3 = 3/100 * 128 * mac38 * 3 * 10 * iter_list 
	cc_list4 = 4.3/100 * 128 * mac38 * 3 * 10 * iter_list

	return cc_list1, cc_list2, cc_list3, cc_list4, loss_list1, loss_list2, loss_list3, loss_list4


fig, ax = plt.subplots(2, 2, figsize=(10,8))
plt.subplots_adjust(wspace=0.2, hspace=0.3)

font_big = 13
font_mid = 12
font_small = 10

fname1 = "learning_curve/train_logs_resnet38_cifar10_11.52.53.txt"
fname2 = "learning_curve/train_logs_resnet38_cifar10_22.533.5.txt"
fname3 = "learning_curve/train_logs_resnet38_cifar10_cr3_w4.txt"
fname4 = "learning_curve/train_logs_resnet38_cifar10_6_8.txt"

baseline1 = "baseline/IBM8/cifar10_resnet38/record.txt"
baseline2 = "baseline/DoReFa/cifar10_resnet38/record.txt"
baseline3 = "baseline/WAGEUBN/cifar10_resnet38/record.txt"


cc_list1, cc_list2, cc_list3, cc_list4, loss_list1, loss_list2, loss_list3, loss_list4 = get_loss_cc_list(fname1, fname2, fname3, fname4)

train_loss_ibm, train_loss_dorefa, train_loss_wageubn, test_acc_ibm, test_acc_dorefa, test_acc_wageubn, cc_list_ibm, cc_list_dorefa, cc_list_wageubn = get_baseline_info(baseline1, baseline2, baseline3)

ax[0,0].plot(cc_list1, loss_list1)
ax[0,0].plot(cc_list2, loss_list2)
ax[0,0].plot(cc_list3, loss_list3)
ax[0,0].plot(cc_list4, loss_list4)
# saax[0,0].plot(cc_list_ibm, train_loss_ibm)
ax[0,0].plot(cc_list_dorefa, train_loss_dorefa)
ax[0,0].plot(cc_list_wageubn, train_loss_wageubn)
ax[0,0].grid()

ax[0,0].set_title('ResNet-38@CIFAR10: Training Loss - MAC', fontsize=font_big)
ax[0,0].set_xlabel('MACs', fontsize=font_mid)
ax[0,0].set_ylabel('Training Loss', fontsize=font_mid)
ax[0,0].legend(['FracTrain-A','FracTrain-B','DFQ-CP=3','SBM[5]-FW6/BW8','DoReFa[49]-FW4/BW32','WAGEUBN[44]-FW8/BW8'], fontsize=font_small)

ax[0,0].xaxis.set_tick_params(labelsize=font_small)
ax[0,0].yaxis.set_tick_params(labelsize=font_small)

cc_list1, cc_list2, cc_list3, cc_list4, acc_list1, acc_list2, acc_list3, acc_list4 = get_acc_cc_list(fname1, fname2, fname3, fname4)

ax[0,1].plot(cc_list1, acc_list1)
ax[0,1].plot(cc_list2, acc_list2)
ax[0,1].plot(cc_list3, acc_list3)
ax[0,1].plot(cc_list4, acc_list4)
# ax[0,1].plot(cc_list_ibm, test_acc_ibm)
ax[0,1].plot(cc_list_dorefa, test_acc_dorefa)
ax[0,1].plot(cc_list_wageubn, test_acc_wageubn)
ax[0,1].grid()

ax[0,1].set_title('ResNet-38@CIFAR10: Test Accuracy - MAC', fontsize=font_big)
ax[0,1].set_xlabel('MACs', fontsize=font_mid)
ax[0,1].set_ylabel('Test Accuracy(%)', fontsize=font_mid)
ax[0,1].legend(['FracTrain-A','FracTrain-B','DFQ-CP=3','SBM[5]-FW6/BW8','DoReFa[49]-FW4/BW32','WAGEUBN[44]-FW8/BW8'], fontsize=font_small)

ax[0,1].xaxis.set_tick_params(labelsize=font_small)
ax[0,1].yaxis.set_tick_params(labelsize=font_small)


fname1 = "learning_curve/train_logs_resnet38_cifar100_11.52.53.txt"
fname2 = "learning_curve/train_logs_resnet38_cifar100_22.533.5.txt"
fname3 = "learning_curve/train_logs_resnet38_cifar100_cr3_w4.txt"
fname4 = "learning_curve/train_logs_resnet38_cifar100_6_8.txt"

baseline1 = "baseline/IBM8/cifar100_resnet38/record.txt"
baseline2 = "baseline/DoReFa/cifar100_resnet38/record.txt"
baseline3 = "baseline/WAGEUBN/cifar100_resnet38/record.txt"

cc_list1, cc_list2, cc_list3, cc_list4, loss_list1, loss_list2, loss_list3, loss_list4 = get_loss_cc_list(fname1, fname2, fname3, fname4)

train_loss_ibm, train_loss_dorefa, train_loss_wageubn, test_acc_ibm, test_acc_dorefa, test_acc_wageubn, cc_list_ibm, cc_list_dorefa, cc_list_wageubn = get_baseline_info(baseline1, baseline2, baseline3)

ax[1,0].plot(cc_list1, loss_list1)
ax[1,0].plot(cc_list2, loss_list2)
ax[1,0].plot(cc_list3, loss_list3)
ax[1,0].plot(cc_list4, loss_list4)
# ax[1,0].plot(cc_list_ibm, train_loss_ibm)
ax[1,0].plot(cc_list_dorefa, train_loss_dorefa)
ax[1,0].plot(cc_list_wageubn, train_loss_wageubn)
ax[1,0].grid()

ax[1,0].set_title('ResNet-38@CIFAR100: Training Loss - MAC', fontsize=font_big)
ax[1,0].set_xlabel('MACs', fontsize=font_mid)
ax[1,0].set_ylabel('Training Loss', fontsize=font_mid)
ax[1,0].legend(['FracTrain-A','FracTrain-B','DFQ-CP=3','SBM[5]-FW6/BW8','DoReFa[49]-FW4/BW32','WAGEUBN[44]-FW8/BW8'], fontsize=font_small)
ax[1,0].xaxis.set_tick_params(labelsize=font_small)
ax[1,0].yaxis.set_tick_params(labelsize=font_small)

cc_list1, cc_list2, cc_list3, cc_list4, acc_list1, acc_list2, acc_list3, acc_list4 = get_acc_cc_list(fname1, fname2, fname3, fname4)

ax[1,1].plot(cc_list1, acc_list1)
ax[1,1].plot(cc_list2, acc_list2)
ax[1,1].plot(cc_list3, acc_list3)
ax[1,1].plot(cc_list4, acc_list4)
# ax[1,1].plot(cc_list_ibm, test_acc_ibm)
ax[1,1].plot(cc_list_dorefa, test_acc_dorefa)
ax[1,1].plot(cc_list_wageubn, test_acc_wageubn)
ax[1,1].grid()

ax[1,1].set_title('ResNet-38@CIFAR100: Test Accuracy - MAC', fontsize=font_big)
ax[1,1].set_xlabel('MACs', fontsize=font_mid)
ax[1,1].set_ylabel('Test Accuracy(%)', fontsize=font_mid)
ax[1,1].legend(['FracTrain-A','FracTrain-B','DFQ-CP=3','SBM[5]-FW6/BW8','DoReFa[49]-FW4/BW32','WAGEUBN[44]-FW8/BW8'], fontsize=font_small)

ax[1,1].xaxis.set_tick_params(labelsize=font_small)
ax[1,1].yaxis.set_tick_params(labelsize=font_small)

plt.savefig('fractrain.pdf', bbox_inches='tight')
# plt.show()

