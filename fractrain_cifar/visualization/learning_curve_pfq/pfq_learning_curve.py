import numpy as np 
import matplotlib.pyplot as plt 
import re

def extract_loss(exp):
	step = 100
	loss_list_static = []
	with open(exp[0], 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if 'Iter' in line and 'Loss' in line:
				loss = re.findall('\d+\.\d+', line)[4]
				loss_list_static.append(float(loss))

	loss_list_static = np.array(loss_list_static)[::step]

	loss_list_prog = []
	with open(exp[1], 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if 'Iter' in line and 'Loss' in line:
				loss = re.findall('\d+\.\d+', line)[4]
				loss_list_prog.append(float(loss))

	loss_list_prog = np.array(loss_list_prog)[::step]

	epoch_list = np.arange(1,min(len(loss_list_static)+1,len(loss_list_prog)+1)) * step * 10 / 390

	return epoch_list, loss_list_static, loss_list_prog


def extract_acc(exp):
	acc_list_static = []
	with open(exp[0], 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if '*' in line and 'Full' not in line:
				acc = re.findall('\d+\.\d+', line)
				acc_list_static.append(float(acc[0]))

	acc_list_prog = []
	with open(exp[1], 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if '*' in line and 'Full' not in line:
				acc = re.findall('\d+\.\d+', line)
				acc_list_prog.append(float(acc[0]))

	epoch_list = np.arange(1,min(len(acc_list_static)+1,len(acc_list_prog)+1))

	return epoch_list, acc_list_static, acc_list_prog 


exp1 = ["train_logs_resnet74_cifar10_8_12.txt", "train_logs_resnet74_cifar10_3468_681012.txt"]
exp2 = ["train_logs_resnet38_cifar100_8_8.txt", "train_logs_resnet38_cifar100_3468_6688.txt"]

font_big = 20
font_mid = 15
font_small = 12

#fig, ax = plt.subplots(3, 2, figsize=(10,15))
fig, ax = plt.subplots(2, 2, figsize=(10,8))
plt.subplots_adjust(wspace=0.2, hspace=0.35)

epoch_list, loss_list_static, loss_list_prog = extract_loss(exp1)

ax[0,0].plot(epoch_list, loss_list_static)
ax[0,0].plot(epoch_list, loss_list_prog)
ax[0,0].grid()
ax[0,0].set_title('ResNet-74@CIFAR-10', fontsize=font_big)
ax[0,0].set_ylabel('Training Loss', fontsize=font_mid)
ax[0,0].set_xlabel('Epochs', fontsize=font_mid)
ax[0,0].legend(['FW8/BW12', 'FW3-4-6-8/BW6-8-10-12'])
ax[0,0].xaxis.set_tick_params(labelsize=font_small)
ax[0,0].yaxis.set_tick_params(labelsize=font_small)

epoch_list, acc_list_static, acc_list_prog = extract_acc(exp1)

ax[0,1].plot(epoch_list, acc_list_static)
ax[0,1].plot(epoch_list, acc_list_prog)
ax[0,1].grid()
ax[0,1].set_title('ResNet-74@CIFAR-10', fontsize=font_big)
ax[0,1].set_ylabel('Test Accuracy(%)', fontsize=font_mid)
ax[0,1].set_xlabel('Epochs', fontsize=font_mid)
ax[0,1].legend(['FW8/BW12', 'FW3-4-6-8/BW6-8-10-12'])
ax[0,1].xaxis.set_tick_params(labelsize=font_small)
ax[0,1].yaxis.set_tick_params(labelsize=font_small)

epoch_list, loss_list_static, loss_list_prog = extract_loss(exp2)

ax[1,0].plot(epoch_list, loss_list_static)
ax[1,0].plot(epoch_list, loss_list_prog)
ax[1,0].grid()
ax[1,0].set_title('ResNet-38@CIFAR-100', fontsize=font_big)
ax[1,0].set_ylabel('Training Loss', fontsize=font_mid)
ax[1,0].set_xlabel('Epochs', fontsize=font_mid)
ax[1,0].legend(['FW8/BW8', 'FW3-4-6-8/BW6-6-8-8'])
ax[1,0].xaxis.set_tick_params(labelsize=font_small)
ax[1,0].yaxis.set_tick_params(labelsize=font_small)

epoch_list, acc_list_static, acc_list_prog = extract_acc(exp2)

ax[1,1].plot(epoch_list, acc_list_static)
ax[1,1].plot(epoch_list, acc_list_prog)
ax[1,1].grid()
ax[1,1].set_title('ResNet-38@CIFAR-100', fontsize=font_big)
ax[1,1].set_ylabel('Test Accuracy(%)', fontsize=font_mid)
ax[1,1].set_xlabel('Epochs', fontsize=font_mid)
ax[1,1].legend(['FW8/BW8', 'FW3-4-6-8/BW6-6-8-8'])
ax[1,1].xaxis.set_tick_params(labelsize=font_small)
ax[1,1].yaxis.set_tick_params(labelsize=font_small)

plt.savefig('pfq_learning_curve.pdf', bbox_inches='tight')


