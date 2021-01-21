import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")#E:\gitHubProjects\Mask_RCNN-master\Mask_RCNN-master
print(ROOT_DIR)

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco

# %matplotlib inline

# Directory to save logs and trained model 已经训练好的模型和相关日志的目录
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
COCO_MODEL_PATH = r"E:\gitHubProjects\Mask_RCNN-master\Mask_RCNN-master\logs\mask_rcnn_balloon_0030.h5"
# Download COCO trained weights from Releases if needed 如果预训练模型不存在则从网上下载
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on# 用于检测的图像集
# IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = r'E:\gitHubProjects\Mask_RCNN-master\Mask_RCNN-master\samples\balloon\balloonDataset\val'
##########################################################
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()



#########################################################
# Create model object in inference mode.model是已经训练好的模型
# model_dir:Directory to save training logs and trained weights
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

###########################################################
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')coco数据集所有的类名称，列表索引号就是类别号
class_names = ['BG', 'polygon']

###########################################################################
# Load a random image from the images folder任意加载一张图片进行目标检测
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

# Run detection # 开始检测
results = model.detect([image], verbose=1) # 返回的是一个列表，一张图片一个字典，字典含有roi、mask等信息

# Visualize results
r = results[0]#取第一张图的字典，并且进行显示结果
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])