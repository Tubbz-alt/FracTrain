import numpy as np 
import matplotlib.pyplot as plt 

cr_dyn = [1.118, 1.602, 2.002, 3.977]
acc_dyn = [65.85, 66.71, 67.83, 68.98]

cr_prog = [1.26, 1.88, 2.22, 2.56]
acc_prog = [61.63, 69.33, 69.58, 69.09]

cr_static = [1.07, 1.46, 2.08, 2.6, 3.65, 4.3, 5.86, 7.42]
acc_static = [58.01, 67.46, 67.78, 67.93, 68.05, 68.74, 68.79, 69.99]


plt.plot(cr_static, acc_static, '-^')
plt.plot(cr_prog, acc_prog, '-^')
plt.plot(cr_dyn, acc_dyn, '-^')
plt.grid()

plt.title('Accuracy - Compression Ratio')
plt.xlabel('Compression Ratio / %')
plt.ylabel('Accuracy / %')
plt.legend(['static', 'progressive', 'dynamic'])

plt.savefig('acc-cr.jpg')
plt.show()