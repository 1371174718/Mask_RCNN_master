# 基于E:\GitHub_Projects\Mask_RCNN_master\samples\shapes进行的修改
# %%
from shapes import *
import numpy as np
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

# %%

class BalloonConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "version"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1

    VALIDATION_STEPS = 1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640


config = BalloonConfig()
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


def data_split(full_list, ratio, shuffle=False):
	n_total = len(full_list)
	offset = int(n_total * ratio)
	if n_total == 0 or offset < 1:
		return [], full_list
	if shuffle:
		random.shuffle(full_list)
	sublist_1 = full_list[:offset]
	sublist_2 = full_list[offset:]
	return sublist_1, sublist_2

from pathlib import Path
config = BalloonConfig()
dataset_root_path = r'E:\GitHub_Projects\Mask_RCNN_master\samples\compute_map\saveImgFiles'
# dataset_root_path = r'D:\宜昌\zhiwei_3D\tuyang_json_img'
# os.path.join(ROOT_DIR, "train_data\\train_1")
path = Path(dataset_root_path)
all_json_file = list(path.glob('**/*.json'))

val_json_file, train_json_file = data_split(all_json_file, ratio=0.2, shuffle=True)
train_count = len(train_json_file)
val_count = len(val_json_file)


# Training dataset.
dataset_train = BalloonDataset()
dataset_train.load_balloon(dataset_root_path, train_json_file)
dataset_train.prepare()

# Validation dataset
dataset_val = BalloonDataset()
dataset_val.load_balloon(dataset_root_path, val_json_file)
dataset_val.prepare()

## Detection

# %%

class InferenceConfig(BalloonConfig):
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = r"E:\GitHub_Projects\Mask_RCNN_master\logs\mask_rcnn_balloon_0030.h5"
# model_path = r"D:\宜昌\zhiwei_3D\tuyang_json_img\mask_rcnn_balloon_0030.h5"
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