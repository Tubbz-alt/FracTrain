import numpy as np 
import matplotlib.pyplot as plt 
import re
import sys
import os


mac38 = 0.08349 * 1e9
mac74 = 0.17036 * 1e9


fname1 = "learning_curve/train_logs_resnet38_cifar10_11.52.53.txt"
fname2 = "learning_curve/train_logs_resnet38_cifar10_22.533.5.txt"
fname3 = "learning_curve/train_logs_resnet38_cifar10_cr3_w4.txt"
fname4 = "learning_curve/train_logs_resnet38_cifar10_6_8.txt"

step = 120

def get_loss_list(fname):
	loss_list = []
	with open(fname, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if 'Iter' in line and 'Loss' in line:
				loss = re.findall('\d+\.\d+', line)[4]
				loss_list.append(float(loss))
	return loss_list

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


plt.plot(cc_list1, loss_list1)
plt.plot(cc_list2, loss_list2)
plt.plot(cc_list3, loss_list3)
plt.plot(cc_list4, loss_list4)
plt.grid()

plt.title('Training Loss - Computation Cost')
plt.xlabel('MACs')
plt.ylabel('Training Loss / %')
plt.legend(['TSDPT1', 'TSDPT2', 'Dynamic', 'Static'])

plt.savefig('ResNet38-Cifar10-Loss-Mac.jpg')
plt.show()


