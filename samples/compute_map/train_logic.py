# -*- coding: utf-8 -*-
# Created by WIN10 on 2020/10/9
# Copyright (c) 2020 WIN10. All rights reserved.
from resource.train_ui import Ui_Train
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QFileDialog
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import os
from pathlib import Path
import json
import numpy as np
import skimage.draw
import random
from PyQt5 import QtWidgets, QtCore,QtGui
import _thread
from PyQt5.QtCore import pyqtSignal
############################################################
#  Configurations
############################################################


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


############################################################
#  Dataset
############################################################

class BalloonDataset(utils.Dataset):

    def load_balloon(self, dataset_dir, imglist):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("shapes", 1, "sack")
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


# 继承QThread
class Runthread(QtCore.QThread):
    #  通过类成员对象定义信号对象
    _signal = pyqtSignal(str)

    def __init__(self, image_path, train_epoch, model_path):
        super(Runthread, self).__init__()
        self.image_path = image_path
        self.train_epoch = train_epoch
        self.model_path = model_path
        self.step=0

    def __del__(self):
        self.wait()

    def run(self):
        config = BalloonConfig()
        dataset_root_path = self.image_path
        # os.path.join(ROOT_DIR, "train_data\\train_1")
        path = Path(dataset_root_path)
        all_json_file = list(path.glob('**/*.json'))

        val_json_file, train_json_file = self.data_split(all_json_file, ratio=0.2, shuffle=True)
        train_count = len(train_json_file)
        val_count = len(val_json_file)
        config.STEPS_PER_EPOCH = train_count
        """Train the model."""
        # Training dataset.
        dataset_train = BalloonDataset()
        dataset_train.load_balloon(dataset_root_path, train_json_file)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = BalloonDataset()
        dataset_val.load_balloon(dataset_root_path, val_json_file)
        dataset_val.prepare()

        model = modellib.MaskRCNN(mode="training", config=config, model_dir='')
        model.load_weights(self.model_path, by_name=True)
        _thread.start_new_thread(self.check_change,(model,train_count,))
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=self.train_epoch,
                    layers="heads")

    def check_change(self,model,train_count):
        self._signal.emit(str(model.step))
        while True:
            if model.step!=self.step:
                self._signal.emit(str(model.step*100//(self.train_epoch*train_count)))
                self.step=model.step


    def data_split(self, full_list, ratio, shuffle=False):
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]
        return sublist_1, sublist_2


class Train_logic(QMainWindow, Ui_Train):

    def __init__(self):
        super(Train_logic, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.connect()
        self.image_path = 'image'
        self.model_path = ''
        self.model_Save = ''
        self.train_epoch = 50
        self.step_epoch = 100
        self.step = 0

    def connect(self):
        self.pushButton.clicked.connect(self.model_path)
        self.pushButton_2.clicked.connect(self.image_path)
        self.pushButton_4.clicked.connect(self.start_login)

    def model_path(self):

        fname = QFileDialog.getOpenFileName(self, 'open file', '/')
        if fname[0]:
            try:
                self.model_path = fname[0]
                msg = '训练模型：' + fname[0] + '\n'
                self.write_msg(msg)
            except:
                QtWidgets.QMessageBox.critical(self, "错误", "打开文件失败，可能是文件内型错误")

    def call_backlog(self, msg):
        self.pb22.setValue(int(msg))  # 将线程的参数传入进度条
        if msg == '100':
            # self.thread.terminate()
            del self.thread
            self.pushButton_4.setEnabled(True)

    def image_path(self):
        fname = QFileDialog.getExistingDirectory(self, 'open file', '/')
        if fname:
            try:
                self.image_path = fname
                msg = '图片路径：' + fname + '\n'
                self.write_msg(msg)
            except:
                QtWidgets.QMessageBox.critical(self, "错误", "打开文件夹失败")

    def start_login(self):
        # 创建线程
        self.pushButton_4.setEnabled(False)
        self.thread = Runthread(self.image_path,int(self.lineEdit.text()),self.model_path)
        # 连接信号
        self.thread._signal.connect(self.call_backlog)  # 进程连接回传到GUI的事件
        # 开始线程
        self.thread.start()

    def write_msg(self, msg):
        self.textBrowser.insertPlainText(msg)
        # 滚动条移动到结尾
        self.textBrowser.moveCursor(QtGui.QTextCursor.End)
