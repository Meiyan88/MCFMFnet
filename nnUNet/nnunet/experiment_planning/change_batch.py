# 注意新生成的pkl的存储路径一定是在原pkl同一个文件夹下
import pickle
from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

#源plans文件的路径，基于该文件进行修改,这个保存batchsize的文件在processed文件夹对应的任务id下，请根据实际情况修改下面路径
path = '/home/liuhu/nnUNetFrame/DATASET/nnUNet_preprocessed/Task079_TotalLiverSet/nnUNetPlansv2.1_plans_3D.pkl'
f = open(path, 'rb')
plans = pickle.load(f)
#可以通过print命令查看整个文件的内容，格式为类似json的结构，可以用一些json格式化工具查看，具体度娘
# print(plans)

# print("--------------分割线--------------")
# 查看原来的patchsize
# print(plans['plans_per_stage'][0])
# print(plans['plans_per_stage'][1]['patch_size'])


plans = load_pickle(path)

# 例如，plans 更改patchsize 将batchsize改为6 patchsize改为48*192*192
# 这个是stage0 的patch和batch 大小int(self.img_dim / 16),
# plans['plans_per_stage'][0]['batch_size'] = 20
# plans['plans_per_stage'][0]['patch_size'] = np.array((48, 96, 96))

# 这个是stage1 的patch和batch 大小
plans['plans_per_stage'][0]['batch_size'] = 10
plans['plans_per_stage'][0]['patch_size'] = np.array((128, 128, 128))
# plans['plans_per_stage'][0]['pool_op_kernel_sizes'] = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
# plans['plans_per_stage'][0]['conv_kernel_sizes'] = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
# save the plans under a new plans name. Note that the new plans file must end with _plans_2D.pkl!
#保存到默认的路径下，这样才能被识别，必须以_plans_2D.pkl或者_plans_3D.pkl结尾；可以按照以下方式命名方便通过文件名识别batchsize的大小
save_pickle(plans, join("/home/liuhu/nnUNetFrame/DATASET/nnUNet_preprocessed/Task073_EXCETLiver/",
                        'nnUNetPlansv2.1_ps128128128_bs10_plans_3D.pkl'))
#nnUNetPlansv2.1_trgSp_1x1x1_plans_3D.pkl
print(plans['plans_per_stage'][0])