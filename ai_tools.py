from types import new_class
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras import layers
import rasterio
from rasterio.plot import reshape_as_image, reshape_as_raster
from skimage.transform import resize

import albumentations as A



from tensorflow.python.keras.backend import int_shape
from tensorflow.python.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import DepthwiseConv2D
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers.experimental.preprocessing import Resizing


import os
import random
import numpy as np
import re

from sklearn.metrics import accuracy_score, jaccard_score, f1_score
from sklearn.metrics import confusion_matrix,recall_score,precision_score
import seaborn as sns
import matplotlib.pyplot as plt


"""
from tensorflow.keras.layers.pooling import MaxPooling2D
from tensorflow.keras.layers.merge import concatenate
"""


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
               kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16,
                      kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs],)
    return model


def get_multiUnet(input_img, nclasses, n_filters=16, dropout=0.1, batchnorm=True):

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1,
                      kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters * 16,
                      kernel_size=3, batchnorm=batchnorm)

    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3),
                         strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters * 4, (3, 3),
                         strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters * 2, (3, 3),
                         strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters * 1, (3, 3),
                         strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(nclasses, (1, 1), activation='softmax')(c9)
    model = Model(inputs=[input_img], outputs=[outputs],)
    return model


class DataLoader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, num_bands, input_img_paths, target_img_paths, nclasses=2, augment=False, sparse=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths
        self.num_bands = num_bands
        self.nclasses = nclasses
        self.sparse = sparse

        self.augment = augment
        if self.augment:
            self.augmentation = A.Compose([

                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ])

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.img_size +
                     (self.num_bands,), dtype="float32")

        for j, path in enumerate(batch_input_img_paths):
            with rasterio.open(path) as src:
                rast = src.read()

            img = reshape_as_image(rast) # <<<<<<<<<<<<<<<<<<<<<<<<<<<<
            #img = tf.image.resize(images=img, size=self.img_size)
            x[j] = img

        if self.nclasses > 2 and not self.sparse:
            y = np.zeros((self.batch_size,) + self.img_size +
                         (self.nclasses,), dtype="uint8")

        else:
            y = np.zeros((self.batch_size,) +
                         self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_target_img_paths):
            with rasterio.open(path) as src:
                rast = src.read()

            img = reshape_as_image(rast) # <<<<<<<<<<<<<<<<<<<<<<<<<<
            #img = tf.image.resize(images=img, size=self.img_size)

            if self.augment:
                transformed = self.augmentation(image=x[j], mask=img)
                x[j] = transformed["image"]
                img = transformed["mask"]

            if self.nclasses > 2 and not self.sparse:
                img = keras.utils.to_categorical(
                    img, num_classes=self.nclasses)
                y[j] = img

            else:
                y[j] = img

        return x, y


def train_folders(input_dir, target_dir):
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".tif") or fname.endswith(".png")
        ])

    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".tif") and not fname.startswith(".") or fname.endswith(".png")
        ])

    return input_img_paths, target_img_paths


def split_train_val(input_dir, target_dir, num_val_samples, batch_size, img_size, nbands, nclasses=2, augment=[False, False], sparse=False):
    # Split our img paths into a training and a validation set

    input_img_paths, target_img_paths = train_folders(input_dir, target_dir)
    val_samples = num_val_samples

    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)

    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    train_gen = DataLoader(
        batch_size, img_size, nbands, train_input_img_paths, train_target_img_paths, nclasses=nclasses, augment=augment[0], sparse=sparse
    )
    val_gen = DataLoader(batch_size, img_size, nbands,
                         val_input_img_paths, val_target_img_paths, nclasses=nclasses, augment=augment[1], sparse=sparse)

    return train_gen, val_gen


def mk_input_img(in_shape):
    input_img = Input(in_shape)
    return input_img


def predict(model):
    def predictionFunc(raster):
        img = rasterio.plot.reshape_as_image(raster)

        img = np.expand_dims(img, 0)

        y = model.predict(img)[0]
        y = reshape_as_raster(y)
        return y

    return predictionFunc



def validate_model(model,val_generator):
  val_data = [i for i in val_generator]
  val_imgs = [i for (i,_) in val_data]
  val_mask = [i for (_,i) in val_data]


  val_pred = [model(x) for x in val_imgs]
  val_mask = np.concatenate(val_mask,axis=0)

  #y_pred_model_output = model.predict(val_imgs)
  y_pred_model_output = np.concatenate(val_pred, axis=0)
  y_pred = np.argmax(y_pred_model_output,axis=-1)
  y_true = np.argmax(val_mask,axis=-1)
  y_pred_cat = keras.utils.to_categorical(y_pred,7)
  y_true_cat = keras.utils.to_categorical(y_true,7)

  acc = keras.metrics.CategoricalAccuracy()
  acc.update_state(y_true_cat, y_pred_cat)
  print("Accuracy Keras:", float(acc.result()))

  y_true_flat = y_true.ravel()
  y_pred_flat = y_pred.ravel()

  print("Accuracy:",accuracy_score(y_true_flat, y_pred_flat))
  print("IoU:", jaccard_score(y_true_flat, y_pred_flat,average="micro"))
  print("F1:", f1_score(y_true_flat, y_pred_flat,average="micro"))
  print("Precision:", precision_score(y_true_flat, y_pred_flat,average="micro"))
  print("Recall:" ,recall_score(y_true_flat, y_pred_flat,average="micro"))
  print("\n\n")

  num_classes = np.max(y_true) + 1
  class_accuracy = np.zeros(num_classes)
  class_precision = np.zeros(num_classes)
  class_recall = np.zeros(num_classes)
  class_f1 = np.zeros(num_classes)
  class_iou = np.zeros(num_classes)

  # Compute for each class
  for class_label in range(num_classes):
      class_y_true = (y_true_flat == class_label)
      class_y_pred = (y_pred_flat == class_label)


      class_accuracy[class_label] = accuracy_score(class_y_true, class_y_pred)
      class_precision[class_label] = precision_score(class_y_true, class_y_pred)
      class_recall[class_label] = recall_score(class_y_true, class_y_pred)
      class_f1[class_label] = f1_score(class_y_true, class_y_pred)
      class_iou[class_label] = jaccard_score(class_y_true, class_y_pred)

  print("Accuracy for each class:", class_accuracy)
  print("Precision for each class:", class_precision)
  print("Recall for each class:", class_recall)
  print("F1 for each class:", class_f1)
  print("IoU for each class:", class_iou)

  print("Accuracy:",np.mean( class_accuracy))
  print("Precision:", np.mean(class_precision))
  print("Recall:", np.mean(class_recall))
  print("F1:", np.mean(class_f1))
  print("IoU:", np.mean(class_iou))

  conf_matrix = confusion_matrix(y_true_flat,y_pred_flat)
  plt.figure(figsize=(10, 8))
  sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  plt.title("Confusion Matrix")
  plt.show()



