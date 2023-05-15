import os
import random

"""
对图片数据集进行随机分类
以8: 1: 1的比例分为训练数据集，验证数据集和测试数据集
运行后在ImageSets文件夹中会出现四个文件
"""
ROOT = "D:/缺陷检测项目/项目二/yolov5-6.1/NEU-DET/"
trainval_percent = 0.9
train_percent = 0.9
xmlfilepath = ROOT + "Annotations"
txtsavepath = ROOT + "ImageSets"
if not os.path.exists(txtsavepath):
    os.makedirs(txtsavepath)
# 获取该路径下所有文件的名称，存放在list中
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(ROOT + "ImageSets/trainval.txt", "w")
ftest = open(ROOT + "ImageSets/test.txt", "w")
ftrain = open(ROOT + "ImageSets/train.txt", "w")
fval = open(ROOT + "ImageSets/val.txt", "w")

for i in list:
    # 获取文件名称中.xml之前的序号
    name = total_xml[i][:-4] + "\n"
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
