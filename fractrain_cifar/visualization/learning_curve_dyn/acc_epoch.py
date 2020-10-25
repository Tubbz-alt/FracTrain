import numpy as np 
import matplotlib.pyplot as plt 
import re
import sys
import os


mac38 = 0.08349 * 1e9
mac74 = 0.17036 * 1e9


fname1 = "learning_curve/train_logs_resnet74_cifar10_11.52.53.txt"
fname2 = "learning_curve/train_logs_resnet74_cifar10_22.533.5.txt"
fname3 = "learning_curve/train_logs_resnet74_cifar10_cr3_w4.txt"
fname4 = "learning_curve/train_logs_resnet74_cifar10_6_8.txt"


def get_acc_list(fname):
	acc_list = []
	with open(fname, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if '*' in line and 'Full' not in line:
				acc = re.findall('\d+\.\d+', line)
				acc_list.append(float(acc[0]))
	return acc_list

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


plt.plot(cc_list1, acc_list1)
plt.plot(cc_list2, acc_list2)
plt.plot(cc_list3, acc_list3)
plt.plot(cc_list4, acc_list4)
plt.grid()

plt.title('Test Accuracy - Computation Cost')
plt.xlabel('MACs')
plt.ylabel('Test Accuracy / %')
plt.legend(['TSDPT1', 'TSDPT2', 'Dynamic', 'Static'])

plt.savefig('ResNet74-Cifar10-Acc-Mac.jpg')
plt.show()


