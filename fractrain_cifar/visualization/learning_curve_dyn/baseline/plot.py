# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import matplotlib.pyplot as plt
import os, sys
import numpy as np
# import itertools

path = './cifar100_resnet_38/'

baseline = path + 'record.txt'

line_type = [['r', 'g'], ['b', 'y'], ['violet', 'pink']]

plt.figure(figsize=(10, 8))

def read_record(fname):
	with open(fname, 'r') as log:
	    content = log.readlines()
	    train_loss = []
	    test_acc = []
	    for line in content:
	    	loss = float(line.split(',')[0][:-2])
	        acc = float(line.split(',')[2][:-2])
	        test_acc.append(acc)
	        train_loss.append(loss)
	    train_loss[-1] = train_loss[-2]
	return train_loss, test_acc


axis = [i for i in range(1, lengh)]

# plot test acc
plt.plot(axis, train_acc, line_type[0][0], label="train acc", lw=5)
plt.plot(axis, test_acc, line_type[0][1], label="test acc", lw=5)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Epochs', fontsize=25)
plt.ylabel('Accuracy', fontsize=25)
plt.title('Convergence for ResNet38@CIFAR100 (WAGEUBN)', fontsize=25)
plt.xlim(xmin=0, xmax=lengh-1)
plt.grid(axis='both', color='gray', linestyle='-', linewidth=0.3)

plt.savefig(path + 'convergence.pdf')