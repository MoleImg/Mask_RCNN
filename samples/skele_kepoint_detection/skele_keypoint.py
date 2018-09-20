# -*- coding: utf-8 -*-
"""
Date  :  19/09/2018
Author:  Yuan
Todo :  skeleton keypoints detection using Mask R-CNN

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 skele_keypoint.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 skele_keypoint.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 skele_keypoint.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Test
    python3 skele_keypoint.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import logging
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize
from mrcnn.model import log


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


############################################################
#  Configurations
############################################################


class Skele_keypoint_config(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "skeleton keypoint"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 10  # Background + kepoints

    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class Skele_kepoint_dataset(utils.Dataset):
    """ Load the skeleton keypoints dataset
        For each X-ray image in the dataset,
        we labeled it with 10 skeletal keypoints, with the numerical model as follows
        1-左股骨头中点    2-右股骨头中点
        3-左膝关节中心上10cm中点 4-右膝关节中心上10cm中点
        5-左股骨髁间窝中点  6-右股骨髁间窝中点
        7-左胫骨平台中点   8-右胫骨平台中点
        9-左踝关节中点    10-右踝关节中点
    """

    def __init__(self):
        super(Skele_kepoint_dataset, self).__init__()
        self.keypoint_class_names = ["左股骨头中点", "右股骨头中点",
                            "左膝关节中心上10cm中点", "右膝关节中心上10cm中点",
                            "左股骨髁间窝中点", "右股骨髁间窝中点",
                            "左胫骨平台中点", "右胫骨平台中点",
                            "左踝关节中点", "右踝关节中点"]
        self.image_info = {}
        self.keypoint_image_ids = []

    def load_skele_keypoint(self, data_dir, subset):
        """
        reads the JSON file, extracts the annotations, and iteratively calls the internal
        add_class and add_image functions to build the dataset
        """
        # add classes
        for ii in range(0, len(self.keypoint_class_names)):
            self.add_class("keypoint", ii+1, self.keypoint_class_names[ii])

        # select train or validation dataset
        assert subset in ["train", "val"]
        data_dir = os.path.join(data_dir, subset)

        # load annotations as dict
        with open(os.path.join(data_dir, 'anno_keypoints.json'), 'r') as f:
            annos_dic = json.load(f)

        count = 0
        # add images
        # self.image_info = {}
        for img_name in annos_dic.keys():
            img_path = os.path.join(data_dir, img_name)
            img_path = img_path + ".jpg"
            owners = list(annos_dic[img_name])
            print("Add image " + img_name)
            if img_name == 'mjb_A00111':
                print("mjb_A00111")
            if '海那边1517' in owners and 'luyufengyangmeng' in owners:
                annos_dic[img_name] = self.__merge_annotations(annos_dic[img_name],
                                                               anno1='海那边1517', anno2='luyufengyangmeng')
            elif '海那边1517' in owners:
                annos_dic[img_name]['annotations'] = annos_dic[img_name]['海那边1517']
            else:
                annos_dic[img_name]['annotations'] = annos_dic[img_name]['luyufengyangmeng']
            self.add_image("keypoint", image_id=img_name,
                           path=img_path, height=annos_dic[img_name]['height'],
                           width=annos_dic[img_name]['width'], class_num=annos_dic[img_name]['class_count'],
                           annotations=annos_dic[img_name]['annotations'])
            count = count + 1
            self.keypoint_image_ids.append(img_name)

        print(str(count) + " images are loaded.")

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info[image_id] = image_info


    def load_image(self, image_id):
        """
        Load the image data via image id
        Returns: image data in numpy array
        """
        img_info = self.image_info[image_id]
        img_path = img_info['path']
        image = cv2.imread(img_path)

        print(image_id + " is loaded.")
        return image


    def load_mask(self, image_id):
        """
        Generate instance masks for a single image
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]  # dict of each image
        # mask generation: [height, width, instance_count=class_count]
        mask = np.zeros([image_info['height'], image_info['width'], image_info['class_num']], dtype=np.uint8)
        instance_count = image_info['class_num']
        for ii in range(0, instance_count):
            annotations = image_info['annotations'][self.keypoint_class_names[ii]]
            (x1, y1, x2, y2) = self.__point_growing(annotations, size=50)

            if not self.__isValidMask((x1, y1, x2, y2), image_info['height'], image_info['width']):
                print(image_id + " has invalid mask.")
                continue
            mask[y1:y2, x1:x2, ii] = 1
            # print("max: "+str(np.max(mask[:,:,ii])) + "min: "+str(np.min(mask[:,:,ii])))
            # print("ii: " + str(ii) + " class: " + keypoint_class_names[ii])
            # print("(x1, y1): " + str(x1)+","+str(y1))
            # print("(x2, y2): " + str(x2) + "," + str(y2))

        print(image_id + " mask is generated.")

        return mask.astype(np.bool), np.array(range(0, instance_count), dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "keypoint":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def __merge_annotations(self, anno_dic, anno1, anno2, merge_rule='average'):
        """
        Merge annotations by two different owners
        Returns: single annotation in an anno dict
        """
        dict1 = anno_dic[anno1]
        dict2 = anno_dic[anno2]
        for label, coords in dict1.items():
            if label in dict2.keys():
                if not (len(dict1[label]) == len(dict2[label])):
                    print(label)
                    continue
                annotations = np.mean([dict1[label], dict2[label]], axis=0)
                dict2[label] = annotations.tolist()
        anno_dic['annotations'] = dict2
        return anno_dic

    def __point_growing(self, point_corrds, size):
        if len(point_corrds) == 2:  # centerX, centerY, return (x-size, y-size, x+size, y+size)
            return (round(point_corrds[0]) - int(size / 2), round(point_corrds[1]) - int(size / 2),
                    round(point_corrds[0]) + int(size / 2), round(point_corrds[1]) + int(size / 2))
        else:
            return (-1, -1, -1, -1)

    def __isValidMask(self, point_corrds, height, width):
        if (len(point_corrds) is not 4): return False
        if not ((point_corrds[0] > 0) & (point_corrds[1] > 0)
                    & (point_corrds[2] < width) & point_corrds[3] < height):
            return False
        return True


if __name__ == '__main__':
    import re
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(dir_path, '../'))
    root = os.path.join(dir_path, '..')
    data_dir = os.path.join(root, '..')
    data_dir = os.path.join(data_dir, 'images/keypoint/')

    data_handle = Skele_kepoint_dataset()
    # load dataset
    data_handle.load_skele_keypoint(data_dir, "train")
    # data_handle.prepare()
    # print("Image Count: {}".format(len(data_handle.image_ids)))
    # print("Class Count: {}".format(data_handle.num_classes))
    # for i, info in enumerate(data_handle.class_info):
    #     print("{:3}. {:50}".format(i, info['name']))

    # load single image
    image_ids = np.random.choice(data_handle.keypoint_image_ids, 4)
    for image_id in image_ids:
        image = data_handle.load_image(image_id)
        # load mask
        mask, class_ids = data_handle.load_mask(image_id)
        bbox = utils.extract_bboxes(mask)

        print("image_id ", image_id, data_handle.image_reference(image_id))
        log("image", image)
        log("mask", mask)
        log("class_ids", class_ids)
        log("bbox", bbox)

        visualize.display_instances(image, bbox, mask, class_ids, data_handle.keypoint_class_names)

    # Randomly choose 4 images
    # image_path = os.path.join(data_dir, 'train/')
    # image_files = os.listdir(image_path)
    # image_id_list = []
    # for ii in range(len(image_files)):
    #     img_name = image_files[ii]
    #     # img_name = re.search('\/.+\.(JPG|JPEG|jpg|png)(?=\"\})', img_name)
    #     if os.path.splitext(img_name)[1] == '.jpg':
    #         img_name = img_name.split('.')
    #         img_name = img_name[0]
    #         image_id_list.append(img_name)
    # image_ids = np.random.choice(data_handle.keypoint_image_ids, 4)
    #
    # for image_id in image_ids:
    #     image = data_handle.load_image(image_id)
    #     mask, class_ids = data_handle.load_mask(image_id)
    #     visualize.display_top_masks(image, mask, class_ids, data_handle.keypoint_class_names)

    # configuration
    # config = Skele_keypoint_config()
    # config.display()

    print("========finished========.")

