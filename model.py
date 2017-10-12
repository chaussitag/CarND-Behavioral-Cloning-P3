#!/usr/bin/env python
# coding=utf8

import argparse
import csv
import math
import numpy as np
import os

from PIL import Image

from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import Cropping2D, Conv2D, Dense, Dropout, Flatten

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt


class TrainConfig(object):
    """

    a class defines all training configure items as properties
    """

    def __init__(self):
        # the default values for all the training config
        self._csv_path = "./train_data/driving_log.csv"
        self._image_dir = "./train_data/IMG"
        self._init_model_path = None
        self._validation_portion = 0.2
        self._use_side_cameras = False
        self._left_correction = -0.2
        self._right_correction = 0.2
        self._batch_size = 512
        self._epochs = 6
        self._dropout_prob = 0.2
        self._top_crop = 70
        self._bottom_crop = 25
        self._image_channel = 3
        self._image_height = 160
        self._image_width = 320
        self._model_output_path = "./model.h5"

    @property
    def csv_path(self):
        return self._csv_path

    @csv_path.setter
    def csv_path(self, p):
        self._csv_path = p

    @property
    def image_dir(self):
        return self._image_dir

    @image_dir.setter
    def image_dir(self, d):
        self._image_dir = d

    @property
    def init_model_path(self):
        return self._init_model_path

    @init_model_path.setter
    def init_model_path(self, model_path):
        self._init_model_path = model_path

    @property
    def validation_portion(self):
        return self._validation_portion

    @validation_portion.setter
    def validation_portion(self, portion):
        self._validation_portion = portion

    @property
    def use_side_cameras(self):
        return self._use_side_cameras

    @use_side_cameras.setter
    def use_side_cameras(self, use):
        self._use_side_cameras = use

    @property
    def left_correction(self):
        return self._left_correction

    @left_correction.setter
    def left_correction(self, lc):
        self._left_correction = lc

    @property
    def right_correction(self):
        return self._right_correction

    @right_correction.setter
    def right_correction(self, rc):
        self._right_correction = rc

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, bs):
        self._batch_size = bs

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, e):
        self._epochs = e

    @property
    def dropout_prob(self):
        return self._dropout_prob

    @dropout_prob.setter
    def dropout_prob(self, p):
        self._dropout_prob = p

    @property
    def top_crop(self):
        return self._top_crop

    @top_crop.setter
    def top_crop(self, tc):
        self._top_crop = tc

    @property
    def bottom_crop(self):
        return self._bottom_crop

    @bottom_crop.setter
    def bottom_crop(self, bc):
        self._bottom_crop = bc

    @property
    def image_channel(self):
        return self._image_channel

    @image_channel.setter
    def image_channel(self, c):
        self._image_channel = c

    @property
    def image_width(self):
        return self._image_width

    @image_width.setter
    def image_width(self, w):
        self._image_width = w

    @property
    def image_height(self):
        return self._image_height

    @image_height.setter
    def image_height(self, h):
        self._image_height = h

    @property
    def model_output_path(self):
        return self._model_output_path

    @model_output_path.setter
    def model_output_path(self, o):
        self._model_output_path = o


def createDefaultTrainConfig():
    default_config = TrainConfig()
    return default_config


default_train_config = createDefaultTrainConfig()


def load_training_info(train_config_):
    sample_list = []
    with open(train_config_.csv_path) as csvfile:
        reader = csv.reader(csvfile)
        # ignore the header
        _ = next(reader)
        for line in reader:
            center_image_path = os.path.join(train_config_.image_dir) + line[0].split('/')[-1]
            assert os.path.isfile(center_image_path), "center_image %s dose not exist" % center_image_path
            steering_angle = float(line[3])
            sample_list.append({
                "image_path": center_image_path,
                "steering_angle": steering_angle,
                "lr_flip": False
            })
            sample_list.append({
                "image_path": center_image_path,
                "steering_angle": -steering_angle,
                "lr_flip": True
            })
            if train_config_.use_side_cameras:
                left_image_path = os.path.join(train_config_.image_dir) + line[1].split('/')[-1]
                assert os.path.isfile(left_image_path), "left image %s does not exist" % left_image_path
                right_image_path = os.path.join(train_config_.image_dir) + line[2].split('/')[-1]
                assert os.path.isfile(right_image_path), "right image %s does not exist" % right_image_path
                left_angle = steering_angle + train_config_.left_correction
                right_angle = steering_angle + train_config_.right_correction

                sample_list.append({
                    "image_path": left_image_path,
                    "steering_angle": left_angle,
                    "lr_flip": False
                })
                sample_list.append({
                    "image_path": left_image_path,
                    "steering_angle": -left_angle,
                    "lr_flip": True
                })

                sample_list.append({
                    "image_path": right_image_path,
                    "steering_angle": right_angle,
                    "lr_flip": False
                })
                sample_list.append({
                    "image_path": right_image_path,
                    "steering_angle": -right_angle,
                    "lr_flip": True
                })

    train_samples_, validation_samples_ =\
        train_test_split(sample_list, test_size=train_config_.validation_portion)
    return train_samples_, validation_samples_


def batch_generator(samples, batch_size):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                # the simulator send RGB image for steering angle prediction,
                # so the model should be trained on RGB image,
                # but opencv read an image as BGR format which could not feed directly as training input.
                # center_image = cv2.imread(batch_sample["image_path"])

                # PIL.Image read image in RGB format
                pil_image = Image.open(batch_sample["image_path"])
                center_image = np.asarray(pil_image)
                # flip image horizontally if necessary
                if batch_sample["lr_flip"]:
                    center_image = np.fliplr(center_image)
                images.append(center_image)
                angles.append(batch_sample["steering_angle"])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def trainModel(train_config_):
    train_samples, validation_samples = load_training_info(train_config_)
    print("total number of training samples %d" % len(train_samples))
    print("total number of validataion samples %d" % len(validation_samples))

    # compile and train the model using the generator function
    train_generator = batch_generator(train_samples, batch_size=train_config_.batch_size)
    validation_generator = batch_generator(validation_samples, batch_size=train_config_.batch_size)

    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation
    model.add(Lambda(lambda x: x / 127.5 - 1.0,
                     input_shape=(train_config_.image_height, train_config_.image_width, train_config_.image_channel)))
    # crop top and bottom portion of the image to only see section with road
    model.add(Cropping2D(cropping=((train_config_.top_crop, train_config_.bottom_crop), (0, 0))))

    # the following model is based on nvidia-architecher from https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    # i add drop out to the model to prevent over-fitting
    model.add(Conv2D(24, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Dropout(train_config_.dropout_prob))

    model.add(Conv2D(36, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Dropout(train_config_.dropout_prob))

    model.add(Conv2D(48, (5, 5), activation="relu", strides=(2, 2)))
    model.add(Dropout(train_config_.dropout_prob))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(train_config_.dropout_prob))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(Dropout(train_config_.dropout_prob))

    model.add(Flatten())

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(train_config_.dropout_prob))

    model.add(Dense(50, activation='relu'))
    model.add(Dropout(train_config_.dropout_prob))

    model.add(Dense(10, activation='relu'))

    model.add(Dense(1))

    # load the initial weights if specified
    if train_config_.init_model_path is not None and \
            os.path.isfile(train_config_.init_model_path):
        model.load_weights(train_config_.init_model_path)
        print("load %s as the initial weights" % train_config_.init_model_path)

    # use mean squared error since this is a regression problem
    model.compile(loss='mse', optimizer='adam')
    hist_obj = model.fit_generator(train_generator,
                                   steps_per_epoch=math.ceil(len(train_samples) / train_config_.batch_size),
                                   validation_data=validation_generator,
                                   validation_steps=math.ceil(len(validation_samples) / train_config_.batch_size),
                                   nb_epoch=train_config_.epochs, verbose=1)

    if os.path.isfile(train_config_.model_output_path):
        os.rename(train_config_.model_output_path, train_config_.model_output_path + ".old")
        print("model save path conflict with an existing file, rename the existing one to %s" \
              % train_config_.model_output_path + ".old")
    # save the trained model to specified path
    model.save(train_config_.model_output_path)
    print("training finished, model saved to %s" % train_config_.model_output_path)
    return hist_obj


def parse_args():
    parser = argparse.ArgumentParser("training a model to auto steer a car in the simulator")
    parser.add_argument("--init_model", "-m", default=None, type=str,
                        help="a previous saved model used as initial weights for the training model")
    parser.add_argument("--not_show_loss", "-s", action="store_true",
                        help="do not visualize the training and validation loss")
    parser.add_argument("--output", "-o", default="model.h5",
                        help="the path for saving the trained model, default to ./model.h5")
    parser.add_argument("--batch_size", "-b", type=int, default=default_train_config.batch_size,
                        help="batch size, default to %d" % default_train_config.batch_size)
    parser.add_argument("--epochs", "-e", type=int, default=default_train_config.epochs,
                        help="number of epochs, default to %d" % default_train_config.epochs)
    parser.add_argument("--use_side_cameras", action="store_true",
                        help="besides center camera, also use images from left and right camera for training")
    parser.add_argument("--left_correction", type=float,
                        default=default_train_config.left_correction,
                        help="valid only if --use_side_cameras was specified, \
                              use the steering angle for the center camera plus this correction \
                              as the angle for left camera, default to %f" % default_train_config.left_correction)
    parser.add_argument("--right_correction", type=float,
                        default=default_train_config.right_correction,
                        help="valid only if --use_side_cameras was specified, \
                              use the steering angle for the center camera plus this correction \
                              as the angle for right camera, default to %f" % default_train_config.right_correction)
    parser.add_argument("--csv_path", default=default_train_config.csv_path,
                        help="path to the csv file containing information for the training dataset, \
                              default to %s" % default_train_config.csv_path)
    parser.add_argument("--image_dir", default=default_train_config.image_dir,
                        help="directory containing the training images, \
                              default to %s" % default_train_config.image_dir)
    parser.add_argument("--valid_portion", default=default_train_config.validation_portion,
                        help="portion of validation set, float number in [0.0, 1.0), \
                              default to %f" % default_train_config.validation_portion)

    return parser.parse_args()


def createTrainConfigFromArgs(args):
    train_config_ = TrainConfig()

    if args.init_model is not None:
        if os.path.isfile(args.init_model):
            train_config_.init_model_path = args.init_model
        else:
            print("the initial model path %s does not exist %s" % args.init_model)

    train_config_.use_side_cameras = args.use_side_cameras
    if args.use_side_cameras:
        train_config_.left_correction = args.left_correction
        train_config_.right_correction = args.right_correction

    train_config_.batch_size = args.batch_size
    train_config_.epochs = args.epochs

    assert args.image_dir is not None, "the specified image directory is None"
    assert os.path.isdir(args.image_dir), "the specified image directory %s does not exist" % args.image_dir
    train_config_.image_dir = args.image_dir
    if train_config_.image_dir[-1] != os.path.sep:
        train_config_.image_dir += os.path.sep

    assert os.path.isfile(args.csv_path), "%s does not exist" % args.csv_path
    train_config_.csv_path = args.csv_path

    train_config_.validation_portion = args.valid_portion

    train_config_.model_output_path = args.output

    return train_config_


if __name__ == "__main__":
    cmd_line_args = parse_args()
    train_config = createTrainConfigFromArgs(cmd_line_args)
    history_object = trainModel(train_config)

    if not cmd_line_args.not_show_loss:
        ### print the keys contained in the history object
        print(history_object.history.keys())

        ### plot the training and validation loss for each epoch
        plt.plot(history_object.history['loss'])
        plt.plot(history_object.history['val_loss'])
        plt.title('model mean squared error loss')
        plt.ylabel('mean squared error loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validation set'], loc='upper right')
        plt.show()
