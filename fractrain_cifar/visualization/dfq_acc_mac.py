import numpy as np 
import matplotlib.pyplot as plt 

mac38 = 0.0849 * 1e9
mac74 = 0.17036 * 1e9

cp_dyn = np.array([1, 1.5, 2, 3, 4, 5, 6])
cp_static = np.array([0.91, 1.46, 2.08, 3.6, 4.3, 5.86])

# 2/6 3/6 4/6 4/12 6/8 6/12

acc_resnet38_static_cifar10 = np.array([74.53, 88.33, 90.89, 91.79, 92.75, 92.71])
iter_resnet38_static_cifar10 = np.array([59280, 57330, 53802, 49530, 58500, 54990])
cc_resnet38_static_cifar10 = cp_static/100 * iter_resnet38_static_cifar10 * mac38 * 128 * 3


acc_resnet38_dp_cifar10 = np.array([89.22, 90.45, 91.96, 92.49, 92.29, 92.65, 92.63])
iter_resnet38_dp_cifar10 = np.array([42900, 42120, 47190, 50700, 36270, 40170, 37830])
cc_resnet38_dp_cifar10 = cp_dyn/100 * iter_resnet38_dp_cifar10 * mac38 * 128 * 3


acc_resnet74_static_cifar10 = np.array([76.44, 88.92, 91.73, 92.36, 93.22, 93.55])
iter_resnet74_static_cifar10 = np.array([63180, 49140, 59670, 39780, 53430, 62400])
cc_resnet74_static_cifar10 = cp_static/100 * iter_resnet74_static_cifar10 * mac74 * 128 * 3


acc_resnet74_dp_cifar10 = np.array([91.11, 91.91, 92.32, 93.09, 93.11, 93.53, 93.42])
iter_resnet74_dp_cifar10 = np.array([33930, 51480, 35490, 37440, 37050, 59280, 36660])
cc_resnet74_dp_cifar10 = cp_dyn/100 * iter_resnet74_dp_cifar10 * mac74 * 128 * 3


acc_resnet38_static_cifar100 = np.array([43.41, 61.74, 66.67, 67.83, 69.43, 69.98])
iter_resnet38_static_cifar100 = np.array([51480, 63570, 54210, 33540, 33930, 35800])
cc_resnet38_static_cifar100 = cp_static/100 * iter_resnet38_static_cifar100 * mac38 * 128 * 3


acc_resnet38_dp_cifar100 = np.array([66.1, 67.79, 68.73, 69.69, 69.81, 69.99, 69.71])
iter_resnet38_dp_cifar100 = np.array([33150, 34170, 38220, 34320, 36660, 34320, 33540])
cc_resnet38_dp_cifar100 = cp_dyn/100 * iter_resnet38_dp_cifar100 * mac38 * 128 * 3


# acc_resnet74_static_cifar100 = np.array([46.36, 64.52, 68.82, 69.13, 71.13, 71.11])
# iter_resnet74_static_cifar100 = np.array([57300, 50700, 62400, 50310, 39000, 35100])
acc_resnet74_static_cifar100 = np.array([46.36, 64.52, 68.82, 71.13, 71.11])
iter_resnet74_static_cifar100 = np.array([57300, 50700, 62400, 39000, 35100])
cp_static_temp = np.array([0.91, 1.46, 2.08, 4.3, 5.86])
cc_resnet74_static_cifar100 = cp_static_temp/100 * iter_resnet74_static_cifar100 * mac74 * 128 * 3


acc_resnet74_dp_cifar100 = np.array([67.26, 69.68, 71, 70.58, 71.11, 71.4, 71.16])
iter_resnet74_dp_cifar100 = np.array([33150, 34170, 38220, 34320, 36660, 34320, 33540])
cc_resnet74_dp_cifar100 = cp_dyn/100 * iter_resnet74_dp_cifar100 * mac74 * 128 * 3


# acc_vgg_static_cifar10 = [58.02, 87.69, 89.74, 92.71, 92.91, 93.53]
# acc_vgg_dp_cifar10 = [0, 0, 90.91, 92.26, 92.45, 93.15, 93.73]

# acc_vgg_static_cifar100 = [31.98, 60.59, 67.53, 70.27, 71.79, 72.23]
# acc_vgg_dp_cifar100 = [0, 0, 68.79, 71.08, 71.74, 71.32, 71.89]

# acc_vgg_static_cifar10 = [58.02, 87.69, 89.74, 92.71, 92.91, 93.53]
# acc_vgg_dp_cifar10 = [0, 0, 90.91, 92.26, 92.45, 93.15, 93.73]


font_big = 20
font_mid = 14
font_small = 12

#fig, ax = plt.subplots(3, 2, figsize=(10,15))
fig, ax = plt.subplots(2, 2, figsize=(10,8))
plt.subplots_adjust(wspace=0.2, hspace=0.35)

ax[0,0].plot(cc_resnet38_static_cifar10, acc_resnet38_static_cifar10, '-^')
ax[0,0].plot(cc_resnet38_dp_cifar10, acc_resnet38_dp_cifar10, '-^')
ax[0,0].set_title('ResNet-38@CIFAR-10', fontsize=font_big)
ax[0,0].set_ylabel('Accuracy(%)', fontsize=font_mid)
ax[0,0].set_xlabel('MACs', fontsize=font_mid)
ax[0,0].legend(['static','dynamic'], fontsize=font_mid)
ax[0,0].grid()
ax[0,0].xaxis.set_tick_params(labelsize=font_small)
ax[0,0].yaxis.set_tick_params(labelsize=font_small)

ax[0,1].plot(cc_resnet38_static_cifar100, acc_resnet38_static_cifar100, '-^')
ax[0,1].plot(cc_resnet38_dp_cifar100, acc_resnet38_dp_cifar100, '-^')
ax[0,1].set_title('ResNet-38@CIFAR-100', fontsize=font_big)
ax[0,1].set_ylabel('Accuracy(%)', fontsize=font_mid)
ax[0,1].set_xlabel('MACs', fontsize=font_mid)
ax[0,1].legend(['static','dynamic'], fontsize=font_mid)
ax[0,1].grid()
ax[0,1].xaxis.set_tick_params(labelsize=font_small)
ax[0,1].yaxis.set_tick_params(labelsize=font_small)

ax[1,0].plot(cc_resnet74_static_cifar10, acc_resnet74_static_cifar10, '-^')
ax[1,0].plot(cc_resnet74_dp_cifar10, acc_resnet74_dp_cifar10, '-^')
ax[1,0].set_title('ResNet-74@CIFAR-10', fontsize=font_big)
ax[1,0].set_ylabel('Accuracy(%)', fontsize=font_mid)
ax[1,0].set_xlabel('MACs', fontsize=font_mid)
ax[1,0].legend(['static','dynamic'], fontsize=font_mid)
ax[1,0].grid()
ax[1,0].xaxis.set_tick_params(labelsize=font_small)
ax[1,0].yaxis.set_tick_params(labelsize=font_small)

ax[1,1].plot(cc_resnet74_static_cifar100, acc_resnet74_static_cifar100, '-^')
ax[1,1].plot(cc_resnet74_dp_cifar100, acc_resnet74_dp_cifar100, '-^')
ax[1,1].set_title('ResNet-74@CIFAR-100', fontsize=font_big)
ax[1,1].set_ylabel('Accuracy(%)', fontsize=font_mid)
ax[1,1].set_xlabel('MACs', fontsize=font_mid)
ax[1,1].legend(['static','dynamic'], fontsize=font_mid)
ax[1,1].grid()
ax[1,1].xaxis.set_tick_params(labelsize=font_small)
ax[1,1].yaxis.set_tick_params(labelsize=font_small)

# ax[2,0].plot(cp_static[2:], acc_vgg_static_cifar10[2:], '-^')
# ax[2,0].plot(cp_dyn[2:], acc_vgg_dp_cifar10[2:], '-^')
# ax[2,0].set_title('VGG8@Cifar10', fontsize=font_big)
# ax[2,0].set_ylabel('Accuracy / %', fontsize=font_mid)
# ax[2,0].set_xlabel('CP / %', fontsize=font_mid)
# ax[2,0].legend(['static','dynamic'], fontsize=font_mid)
# ax[2,0].grid()
# ax[2,0].xaxis.set_tick_params(labelsize=font_small)
# ax[2,0].yaxis.set_tick_params(labelsize=font_small)

# ax[2,1].plot(cp_static[2:], acc_vgg_static_cifar100[2:], '-^')
# ax[2,1].plot(cp_dyn[2:], acc_vgg_dp_cifar100[2:], '-^')
# ax[2,1].set_title('VGG8@Cifar100', fontsize=font_big)
# ax[2,1].set_ylabel('Accuracy / %', fontsize=font_mid)
# ax[2,1].set_xlabel('CP / %', fontsize=font_mid)
# ax[2,1].legend(['static','dynamic'], fontsize=font_mid)
# ax[2,1].grid()
# ax[2,1].xaxis.set_tick_params(labelsize=font_small)
# ax[2,1].yaxis.set_tick_params(labelsize=font_small)

plt.savefig('exp_dfq.pdf', bbox_inches='tight')
# plt.show()

