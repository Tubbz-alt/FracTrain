import numpy as np 
import matplotlib.pyplot as plt 
import re

fname1 = 'train_logs_6_8.txt'
fname2 = 'train_logs_3456_5678.txt'

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

epoch_list = np.arange(1,len(acc_list_static)+1)

plt.plot(epoch_list, acc_list_static)
plt.plot(epoch_list, acc_list_prog)
plt.grid()

plt.title('Test Accuracy - Epoch')
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy / %')
plt.legend(['static 6/8', 'progressive 3456_5678'])

plt.savefig('acc-epoch.jpg')
plt.show()