# 基于E:\GitHub_Projects\Mask_RCNN_master\samples\shapes进行的修改
# %%
from shapes import *
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# %matplotlib inline

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_balloon_0030.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# %%

class ShapesConfig(Config):
	"""Configuration for training on the toy shapes dataset.
	Derives from the base Config class and overrides values specific
	to the toy shapes dataset.
	"""
	# Give the configuration a recognizable name
	NAME = "shapes"

	# Train on 1 GPU and 8 images per GPU. We can put multiple images on each
	# GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
	GPU_COUNT = 1
	IMAGES_PER_GPU = 8

	# Number of classes (including background)
	NUM_CLASSES = 1 + 3  # background + 3 shapes

	# Use small images for faster training. Set the limits of the small side
	# the large side, and that determines the image shape.
	IMAGE_MIN_DIM = 128
	IMAGE_MAX_DIM = 128

	# Use smaller anchors because our image and objects are small
	RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

	# Reduce training ROIs per image because the images are small and have
	# few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
	TRAIN_ROIS_PER_IMAGE = 32

	# Use a small epoch since the data is simple
	STEPS_PER_EPOCH = 100

	# use small validation steps since the epoch is small
	VALIDATION_STEPS = 5


config = ShapesConfig()
config.display()


# %%

def get_ax(rows=1, cols=1, size=8):
	"""Return a Matplotlib Axes array to be used in
	all visualizations in the notebook. Provide a
	central point to control graph sizes.

	Change the default size attribute to control the size
	of rendered images
	"""
	_, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
	return ax

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

## Detection

# %%

class InferenceConfig(ShapesConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(ROOT_DIR, "logs\mask_rcnn_shapes.h5")
# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# %%

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask = \
	modellib.load_image_gt(dataset_val, inference_config,
	                       image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                            dataset_train.class_names, figsize=(8, 8))

# %%

results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

# %% md

## Evaluation

# %%

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 10)
APs = []
for image_id in image_ids:
	# Load image and ground truth data
	image, image_meta, gt_class_id, gt_bbox, gt_mask = \
		modellib.load_image_gt(dataset_val, inference_config,
		                       image_id, use_mini_mask=False)
	molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
	# Run object detection
	results = model.detect([image], verbose=0)
	r = results[0]
	# Compute AP
	AP, precisions, recalls, overlaps = \
		utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
		                 r["rois"], r["class_ids"], r["scores"], r['masks'])
	APs.append(AP)

print("mAP: ", np.mean(APs))