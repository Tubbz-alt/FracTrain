import numpy as np 
import matplotlib.pyplot as plt 
import re
import sys

# fname1 = sys.argv[1]
# fname2 = sys.argv[2]

fname1 = "train_logs_resnet74_cifar10_8_12.txt"
fname2 = "train_logs_resnet74_cifar10_3468_681012.txt"

acc_list_static = []
with open(fname1, 'r') as f:
	for line in f.readlines():
		line = line.strip()
		if '*' in line and 'Full' not in line:
			acc = re.findall('\d+\.\d+', line)
			acc_list_static.append(float(acc[0]))

acc_list_prog = []
with open(fname2, 'r') as f:
	for line in f.readlines():
		line = line.strip()
		if '*' in line and 'Full' not in line:
			acc = re.findall('\d+\.\d+', line)
			acc_list_prog.append(float(acc[0]))

epoch_list = np.arange(1,min(len(acc_list_static)+1,len(acc_list_prog)+1))

plt.plot(epoch_list, acc_list_static)
plt.plot(epoch_list, acc_list_prog[:len(acc_list_static)])
plt.grid()

plt.title('Test Accuracy - Epoch')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy / %')
plt.legend(['FP8/BP12', 'FP3-4-6-8/BP6-8-10-12'])

plt.savefig('Acc-Epoch.jpg')
plt.show()