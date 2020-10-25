import numpy as np 
import matplotlib.pyplot as plt 
import re
import sys

fname = 'train_logs.txt'

layer = 'layer' + sys.argv[1]

choice_num = 6

choice_list = [[] for _ in range(choice_num)]

with open(fname, 'r') as f:
	flag = False
	cnt = 0
	for line in f.readlines():
		line = line.strip()

		if flag and cnt < choice_num:
			ratio = re.findall('\d+\.\d+e?-?\d*', line)[0]
			ratio =float(ratio)
			# ratio  =float('{:.2f}'.format(float(ratio)))
			choice_list[cnt].append(ratio)
			cnt += 1
		else:
			flag = False
			cnt = 0

		if layer+'_decision' in line:
			flag = True

epoch_list = np.arange(1,len(choice_list[0])+1)

for i in range(choice_num):
	plt.plot(epoch_list, choice_list[i])

plt.grid()

plt.title('Layer{} Ratio - Epoch'.format(sys.argv[1]) )
plt.xlabel('Epoch')
plt.ylabel('Ratio / %')

plt.legend(['choice_'+str(i) for i in range(choice_num)])

plt.savefig(layer+'-ratio-epoch.jpg')
plt.show()