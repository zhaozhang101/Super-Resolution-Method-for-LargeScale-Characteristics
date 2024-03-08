import argparse
import os
import matplotlib.pyplot as plt
import mlt_dataprocess
import mlt_model
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import seaborn as sns

parser = argparse.ArgumentParser(description='Mltask prediction & classification')
parser.add_argument('--period', type=int, default=1)
args = parser.parse_args()
scale = 2
index = 0
characteristic_index = np.array([0, 1, 2, 3, 4, 5, 6])

# index 是随机的一个区域 顺序是 K phi theta p t los

channel_data = np.load("data/all_all_train.npy")
channel_data_test = torch.from_numpy(channel_data).type(torch.float32)
channel_data_test = channel_data_test[[index], :, :, :]

model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels=64)
model = mlt_dataprocess.resume(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'test.pth'))
Origin = channel_data_test.squeeze().numpy()
LR_Data = F.interpolate(channel_data_test, scale_factor=1 / scale, mode='nearest').squeeze().numpy()

Output = mlt_dataprocess.INterplate(channel_data_test, scale=scale, MODE='bilinear')
with torch.no_grad():
    thephi, poweratio, power, delay, los = model(Output, args)
# los = torch.argmin(lostemp, dim=1, keepdim=True)
# los = los - 1
# a = los.numpy()
# HR_Data = torch.cat((thephi, poweratio, power, delay, los), dim=1).squeeze().numpy()
HR_Data = power.squeeze().numpy()

plt.imshow(HR_Data, cmap='viridis')
plt.colorbar()
plt.show()

scipy.io.savemat('result/result_T.mat', mdict={'HR_Data': HR_Data, 'LR_Data': LR_Data, 'Origin': Origin})

