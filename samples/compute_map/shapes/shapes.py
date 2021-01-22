"""
Mask R-CNN
Configurations and data loading code for the synthetic Shapes dataset.
This is a duplicate of the code in the noteobook train_shapes.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../../yichangzidong/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


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
import json
import skimage

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, imglist):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("shapes", 1, "box")
        for json_file in imglist:
            with json_file.open() as f:
                json_result = json.load(f)
            if type(json_result['shapes']) is dict:
                polygons = [r['points'] for r in json_result['shapes'].values()]
                shapes = [r['label'] for r in json_result['shapes']]
            else:
                polygons = [r['points'] for r in json_result['shapes']]
                shapes = [r['label'] for r in json_result['shapes']]
                # shapes=

            image_path = os.path.join(dataset_dir, json_result['imagePath'])
            height = json_result['imageHeight']
            width = json_result['imageWidth']

            self.add_image(
                "shapes",
                image_id=json_result['imagePath'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                shapes=shapes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)  # number of object

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        mask = np.zeros([info["height"], info["width"], count],
                        dtype=np.uint8)
        for i, p in enumerate(info['polygons']):
            p_y = []
            p_x = []
            for point in p:
                p_y.append(point[1])
                p_x.append(point[0])
            rr, cc = skimage.draw.polygon(p_y, p_x)
            mask[rr, cc, i:i + 1] = 1

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))

        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "balloon":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)



    def random_shape(self, height, width):
        """Generates specifications of a random shape that lies within
        the given height and width boundaries.
        Returns a tuple of three valus:
        * The shape name (square, circle, ...)
        * Shape color: a tuple of 3 values, RGB.
        * Shape dimensions: A tuple of values that define the shape size
                            and location. Differs per shape type.
        """
        # Shape
        shape = "box"
        # Color
        color = tuple([random.randint(0, 255) for _ in range(3)])
        # Center x, y
        buffer = 20
        y = random.randint(buffer, height - buffer - 1)
        x = random.randint(buffer, width - buffer - 1)
        # Size
        s = random.randint(buffer, height // 4)
        return shape, color, (x, y, s)

