# -*- coding: utf-8 -*-
'''
主要分为以下几个文件：
labelme生成的json文件的文件夹；
用于生成json文件的图像集文件夹；
利用labelme_json_to_dataset.py生成的json文件夹，每个json生成五个文件
'''
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from mrcnn.config import Config
# import utils
from mrcnn import model as modellib, utils
from mrcnn import visualize
import yaml
from mrcnn.model import log
from PIL import Image

# Root directory of the project
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

iter_num = 0

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon_0030.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
	"""Configuration for training on the box shapes dataset.
	Derives from the base Config class and overrides values specific
	to the box shapes dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "shapes"

	# Train on 1 GPU and 8 images per GPU. We can put multiple images on each
	# GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # background + boxs

	# Use small images for faster training. Set the limits of the small side
	# the large side, and that determines the image shape.
	IMAGE_MIN_DIM = 256
	IMAGE_MAX_DIM = 512

	# Use smaller anchors because our image and objects are small
	RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

	# Reduce training ROIs per image because the images are small and have
	# few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
	TRAIN_ROIS_PER_IMAGE = 50

	# Use a small epoch since the data is simple
	STEPS_PER_EPOCH = 50
	# use small validation steps since the epoch is small
	VALIDATION_STEPS = 20


config = ShapesConfig()
config.display()


class DrugDataset(utils.Dataset):
	# 得到该图中有多少个实例（物体）
	def get_obj_index(self, image):
		n = np.max(image)
		return n

	# 解析labelme中得到的yaml文件，从而得到mask每一层对应的实例标签
	def from_yaml_get_class(self, image_id):
		info = self.image_info[image_id]
		with open(info['yaml_path']) as f:
			temp = yaml.load(f.read())
			labels = temp['label_names']
			del labels[0]
		return labels

	# 重新写draw_mask
	def draw_mask(self, num_obj, mask, image, image_id):
		# print("draw_mask-->",image_id)
		# print("self.image_info",self.image_info)
		info = self.image_info[image_id]
		# print("info-->",info)
		# print("info[width]----->",info['width'],"-info[height]--->",info['height'])
		for index in range(num_obj):
			for i in range(info['width']):
				for j in range(info['height']):
					# print("image_id-->",image_id,"-i--->",i,"-j--->",j)
					# print("info[width]----->",info['width'],"-info[height]--->",info['height'])
					at_pixel = image.getpixel((i, j))
					if at_pixel == index + 1:
						mask[j, i, index] = 1
		return mask

	def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
		"""Generate the requested number of synthetic images.
		count: number of images to generate.
		height, width: the size of the generated images.

		count:测试集图片数
		img_floder:测试集图片文件夹
		mask_folder:
		imglist:测试集图片文件夹文件列表
		dataset_root_path:根目录
		"""
		# Add classes，添加类别
		# self.add_class("shapes", 1, "defect")  # 脆性区域
		self.add_class("shapes",1,"box")

		for i in range(count):
			# 获取图片宽和高

			filestr = imglist[i].split(".")[0] # 获取图片名称前缀

			# mask_path = mask_floder + "/" + filestr + ".png"
			mask_path = dataset_root_path + "/" + filestr + "_json/label.png" # mask文件
			yaml_path = dataset_root_path + "/" + filestr + "_json/info.yaml" # 五个文件之yaml文件
			print(dataset_root_path + "/" + filestr + "_json/img.png") # 五个文件之图像文件
			cv_img = cv2.imread(dataset_root_path + "/" + filestr + "_json/img.png") # 读取图像

			# 添加每个图像的图像信息
			self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
			               width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)

	# 重写load_mask
	def load_mask(self, image_id):
		"""Generate instance masks for shapes of the given image ID.
		"""
		global iter_num
		print("image_id", image_id)
		info = self.image_info[image_id]
		count = 1  # number of object
		img = Image.open(info['mask_path']) # 利用PIL.Image打开掩模图像
		num_obj = self.get_obj_index(img) # mask个人理解是不同的物体用不同的像素值来表示，首先背景为0，其他的不同物体依次从1递增，因此利用max可以知道图中物体个数
		mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
		mask = self.draw_mask(num_obj, mask, img, image_id)
		occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
		for i in range(count - 2, -1, -1):
			mask[:, :, i] = mask[:, :, i] * occlusion

			occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
		labels = []
		labels = self.from_yaml_get_class(image_id)
		labels_form = []
		for i in range(len(labels)):
			if labels[i].find("box") != -1:
				# print "box"
				labels_form.append("box")
		# elif labels[i].find("PJ")!=-1:
		#     #print "column"
		#     labels_form.append("PJ")

		class_ids = np.array([self.class_names.index(s) for s in labels_form])
		return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
	"""Return a Matplotlib Axes array to be used in
	all visualizations in the notebook. Provide a
	central point to control graph sizes.
	Change the default size attribute to control the size
	of rendered images
	"""
	_, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
	return ax


def list2array(list):
	b = np.array(list[0])
	for i in range(1, len(list)):
		b = np.append(b, list[i], axis=0)
	return b


def text_save(filename, data):  # filename为写入CSV文件的路径，data为要写入数据列表.
	file = open(filename, 'a')
	for i in range(len(data)):
		s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
		s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
		file.write(s)
	file.close()
	print("保存txt文件成功")

# 测试集设置
dataset_root_path = r"E:\GitHub_Projects\Mask_RCNN_master\tuyang_json_img\\" # 数据集根目录
img_floder = dataset_root_path + "pic" # 存放测试图片的文件夹
mask_floder = dataset_root_path + "cv2_mask" #
imglist = os.listdir(img_floder) # 图片文件夹下文件名和文件夹名称组成的列表
count = len(imglist) # 长度

# 准备test数据集
dataset_test = DrugDataset() # 实例化
dataset_test.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
dataset_test.prepare()


# mAP
# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
class InferenceConfig(ShapesConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

model_path = os.path.join(MODEL_DIR, "mask_rcnn_balloon_0030.h5")  # 修改成自己训练好的模型

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

img_list = np.random.choice(dataset_test.image_ids, 85)
APs = []
count1 = 0

# 遍历测试集
for image_id in img_list:
	# 加载测试集的ground truth
	image, image_meta, gt_class_id, gt_bbox, gt_mask = \
		modellib.load_image_gt(dataset_test, inference_config,
		                       image_id, use_mini_mask=False)
	# 将所有ground truth载入并保存
	if count1 == 0:
		save_box, save_class, save_mask = gt_bbox, gt_class_id, gt_mask
	else:
		save_box = np.concatenate((save_box, gt_bbox), axis=0)
		save_class = np.concatenate((save_class, gt_class_id), axis=0)
		save_mask = np.concatenate((save_mask, gt_mask), axis=2)

	molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)

	# 启动检测
	results = model.detect([image], verbose=0)
	r = results[0]

	# 将所有检测结果保存
	if count1 == 0:
		save_roi, save_id, save_score, save_m = r["rois"], r["class_ids"], r["scores"], r['masks']
	else:
		save_roi = np.concatenate((save_roi, r["rois"]), axis=0)
		save_id = np.concatenate((save_id, r["class_ids"]), axis=0)
		save_score = np.concatenate((save_score, r["scores"]), axis=0)
		save_m = np.concatenate((save_m, r['masks']), axis=2)

	count1 += 1

# 计算AP, precision, recall
AP, precisions, recalls, overlaps = \
	utils.compute_ap(save_box, save_class, save_mask,
	                 save_roi, save_id, save_score, save_m)

print("AP: ", AP)
print("mAP: ", np.mean(AP))

# 绘制PR曲线
plt.plot(recalls, precisions, 'b', label='PR')
plt.title('precision-recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.show()

# 保存precision, recall信息用于后续绘制图像
text_save('Kpreci.txt', precisions)
text_save('Krecall.txt', recalls)