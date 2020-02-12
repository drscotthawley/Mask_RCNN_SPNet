#! /usr/bin/env python
"""
Mask R-CNN
Train on the Steelpan Vibrations dataset and count the number of rings.
by Scott H. Hawley

Modified from Balloon, Coco and Nucleus examples,...
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 steelpan.py train --dataset=/path/to/steelpan/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 steelpan.py train --dataset=/path/to/steelpan/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 steelpan.py train --dataset=/path/to/steelpan/dataset --weights=imagenet

    # Run  evaluation on the last model you trained
    python3 steelpan.py evaluate --dataset=/path/to/steelpan/dataset --weights=last
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # turn off tf warnings

import sys
import json
import datetime
import numpy as np
import skimage.draw

# Specifics used by SPNet code
import glob
from operator import itemgetter
import pandas as pd
import cv2
import random

import imgaug.augmenters as iaa


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")





############################################################
#  Configurations
############################################################


class SteelpanConfig(Config):
    """Configuration for training on the steelpan dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "steelpan"

    # Train on _ GPU and _ images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 11  # background + 11 ring-count classes
    #NUM_CLASSES = 1 + 1  # background + is there an antinode region

    # Image size parameters
    IMAGE_MIN_DIM = 384
    IMAGE_MAX_DIM = 512
    #IMAGE_RESIZE_MODE = "crop"
    IMAGE_CHANNEL_COUNT = 3
    #MEAN_PIXEL = (128, 128, 128)

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask


    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 500
    VALIDATION_STEPS = 100

    BACKBONE = 'resnet50'

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 100
    POST_NMS_ROIS_INFERENCE = 200

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 32

    # Image mean (RGB)
    #MEAN_PIXEL = np.array([43.53, 39.56, 48.22])
    MEAN_PIXEL = np.array([24.9,24.9,24.9])


    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 10

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 20

    # default is 0.7, but noticed that reasonable proposals were often around 0.46
    DETECTION_MIN_CONFIDENCE = 0.5


class SteelpanInferenceConfig(SteelpanConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2
    # Don't resize images for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7



############################################################
#  Dataset
############################################################

class SteelpanDataset(utils.Dataset):

    def parse_csv_file(self, csv_filename):
        col_names = ['cx', 'cy', 'a', 'b', 'angle', 'rings']
        df = pd.read_csv(csv_filename,header=None,names=col_names)  # read metadata file
        df.drop_duplicates(inplace=True)  # sometimes the data from Zooniverse has duplicate rows

        arrs = []    # this is a list of lists, containing all the ellipse info & ring counts for an image
        for index, row in df.iterrows() :
            cx, cy = row['cx'], row['cy']
            a, b = row['a'], row['b']
            angle, num_rings = float(row['angle']), row['rings']
            # Input format (from file) is [cx, cy,  a, b, angle, num_rings]
            if (num_rings > 0.0):    # Actually, check for existence
                #tmp_arr = [cx, cy, a, b, np.cos(2*np.deg2rad(angle)), np.sin(2*np.deg2rad(angle)), 0, num_rings]
                tmp_arr = [cx, cy, a, b, angle, num_rings]
                arrs.append(tmp_arr)
            else:
                pass  # do nothing.  default is no ellipses in image

        arrs = sorted(arrs,key=itemgetter(0,1))     # sort by y first, then by x

        return np.array(arrs).tolist()  #np.tolist() merely used so numbers are convered


    def load_steelpan(self, dataset_dir, subset, fraction=1.0):
        """Load a subset of the steelpan dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We will use one class for every count of interference rings/fringes
        max_rings = 11
        for r in range(max_rings):
           self.add_class("steelpan", r+1, str(r+1))
        #self.add_class("steelpan", 1, str(1))

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        """
        The steelpan dataset format is a set of .png images and matching .csv files
        The .csv files list (on one line per antinode):
          x, y, a, b, angle, rings

        We will use the .csv file info to generate the masks for the mask-rnn routine
        """

        # get a list of files
        img_filenames = sorted(glob.glob(dataset_dir+'/*.png'))
        csv_filenames = sorted(glob.glob(dataset_dir+'/*.csv'))
        assert len(img_filenames) == len(csv_filenames)  # simple check to make sure # files matches

        if fraction < 1.0:  # grab a random group from the larger dataset
            c = list(zip(img_filenames, csv_filenames))
            random.shuffle(c)
            img_filenames, csv_filenames = zip(*c)
            total_img_filenames = int(fraction * len(img_filenames))
            img_filenames = img_filenames[0:total_img_filenames]
            csv_filenames = csv_filenames[0:total_img_filenames]

        print("subset",subset,": Loading",len(img_filenames),"images.")

        # add images
        for i, img_file in enumerate(img_filenames):
            csv_file = csv_filenames[i]
            image = skimage.io.imread(img_file)
            height, width = image.shape[:2]

            ellipses = self.parse_csv_file(csv_file)

            self.add_image(
                "steelpan",
                image_id=img_file,
                path=img_file,
                width=width, height=height,
                ellipses=ellipses)

    def get_ellipse_bb(self, x, y, major, minor, angle_deg):
        """
        NOTE: in mask-rcnn bounding boxes are computed directly from masks
        Thus this routine never actually gets called (except maybe as a check)
        Computes tight ellipse bounding box.
           from https://gist.github.com/smidm/b398312a13f60c24449a2c7533877dc0
        """
        rad = np.radians(angle_deg)
        cr, sr, tr = np.cos(rad), np.sin(rad), np.tan(rad)
        mino2, majo2 = minor/2, major/2
        return [y-majo2, x-majo2, y+majo2, x+majo2]

        t = np.arctan(-mino2 * tr / majo2)
        [max_x, min_x] = [x + majo2 * np.cos(t) * cr -
                          mino2 * np.sin(t) * sr for t in (t, t + np.pi)]
        t = np.arctan(mino2 / tr / majo2)
        [max_y, min_y] = [y + mino2 * np.sin(t) * cr +
                          majo2 * np.cos(t) * sr for t in (t, t + np.pi)]
        return [min_y, min_x, max_y, max_x]


    def load_mask(self, image_id):
        """Generate instance masks for an image.  ("Instance" means an antinode region)
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a steelpan dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "steelpan":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        num_ellipses = len(info["ellipses"])
        class_ids = []
        #print("num_ellipses = ",num_ellipses)
        if num_ellipses > 0:
            mask = np.zeros([info["height"], info["width"], num_ellipses],
                            dtype=np.uint8)
            #print("load_mask: info =",info)
            for i, ellipse in enumerate(info["ellipses"]):
                [x, y, a, b, angle, rings ] = ellipse
                x, y, a, b  = round(x), round(y), round(a), round(b)
                angle = round(angle)
                #class_id = int(rings > 0.5)   # for just "is there any antinode or not"?
                class_id = round(rings)
                color = 1
                # draw filled ellipse of 1's
                #  For some reason, drawing directly onto mask[:,:,i] wasn't working. Hence...
                thismask = np.zeros([info["height"], info["width"]])
                cv2.ellipse(thismask, (x,y), (a,b), -angle, 0, 360, color, -1, cv2.LINE_AA, 0)
                mask[:,:,i] = thismask
                class_ids.append(class_id)

            class_ids = np.array(class_ids).astype(np.int32)
            print("     class_ids = ",class_ids)
            # Return mask, and array of class IDs of each instance.
            return mask.astype(np.bool), class_ids
        else:
            # Return our own empty mask to avoid warning messages from super()
            mask = np.empty([0, 0, 0])
            class_ids = np.empty([0], np.int32)
            return mask, class_ids


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "steelpan":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model, fraction=1.0, start_head=True):
    """Train the model."""
    # Training dataset.
    dataset_train = SteelpanDataset()
    dataset_train.load_steelpan(args.dataset, "train", fraction=fraction)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SteelpanDataset()
    dataset_val.load_steelpan(args.dataset, "val")
    dataset_val.prepare()

    #augmentation = iaa.Sometimes(0.5, [iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.GaussianBlur(sigma=(0.0, 5.0))])
    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])



    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    if start_head:
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=10, augmentation=augmentation,
                    layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    print("Training all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 100,
                epochs=200, augmentation=augmentation,
                layers="all")


############################################################
#   Evaluation
############################################################

def evaluate(model, dataset, config):
    # Compute VOC-Style mAP @ IoU=0.5
    num_test = 10# len(dataset.image_ids)
    image_ids = np.random.choice(dataset.image_ids, num_test)
    APs = []

    for image_id in image_ids:
        path = dataset.image_reference(image_id)

        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, config,
                                   image_id, use_mini_mask=False)
        print("path, image_id, gt_class_id, gt_bbox =",path, image_id, gt_class_id, gt_bbox)

        if gt_class_id.size != 0:    # skip 'empty' images
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
            # Run object detection
            results = model.detect([image], verbose=1)
            r = results[0]
            print("        model predictions = ",r)
            # Compute AP
            AP, precisions, recalls, overlaps =\
                utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                 r["rois"], r["class_ids"], r["scores"], r['masks'])
            print("        AP, precisions, recalls, overlaps =",AP, precisions, recalls, overlaps,"\n")
            APs.append(AP)
        else:
            print("      skipping")

    print("mAP: ", np.mean(APs))

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect steelpans.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        default="../../datasets/steelpan",
                        metavar="/path/to/steelpan/dataset/",
                        help='Directory of the steelpan dataset')
    parser.add_argument('--fraction', required=False, type=float,
                        default=1.0, metavar=1.0,
                        help='Fraction of training dataset to load')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        default="imagenet",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "video":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SteelpanConfig()
    else:
        config = SteelpanInferenceConfig()
    config.display()


    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    model.keras_model.metrics_tensors = []


    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights


    # Load weights
    if weights_path.lower() != 'random':
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)


    # train or evaluate
    if args.command == "train":
        start_head = (weights_path != 'random') and (args.weights.lower() != "last")
        train(model, args.fraction, start_head=start_head)
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = SteelpanDataset()
        dataset_val.load_steelpan(args.dataset, "val")
        dataset_val.prepare()
        print("Running evaluation")
        evaluate(model, dataset_val, config)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
