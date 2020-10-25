import numpy as np 
import matplotlib.pyplot as plt 

weight_prec_4 = [4,6,8]
acc_4 = [41.12, 54.04, 58.55]

weight_prec_6 = [4,6,8]
acc_6 = [64.54, 66.59, 64.44]

weight_prec_8 = [4,6,8]
acc_8 = [66.28, 67.17, 68.57]

weight_prec_12 = [4,6,8]
acc_12 = [66.35, 67.06, 67.1]

weight_prec_16 = [4,6,8,10,12,14,16]
acc_16 = [65.48, 68.05, 68.44, 68.95, 68.88, 68.22, 68.21]

#plt.plot(weight_prec_4, acc_4, '-^')
plt.plot(weight_prec_6, acc_6, '-^')
plt.plot(weight_prec_8, acc_8, '-^')
plt.plot(weight_prec_12, acc_12, '-^')
plt.plot(weight_prec_16, acc_16, '-^')
plt.grid()

plt.title('Accuracy - Forward Precision Curve')
plt.xlabel('Forward Precision')
plt.ylabel('Accuracy')
plt.legend(['Backward Precision=6', 'Backward Precision=8', 
            'Backward Precision=12', 'Backward Precision=16'])

plt.savefig('acc-fp.jpg')
plt.show()