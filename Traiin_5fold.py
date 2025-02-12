from re import S
import tensorflow as tf
import os
import cv2
import numpy as np
import time
import keras
from tensorflow.keras import datasets, layers, models
import keras.backend as K
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from pprint import pprint
from keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from pprint import pprint
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import History
import requests
from plyer import notification


SPLITS = 5
EPOCH = 40
BATCH_SIZE = 128
FILTERSIZE = 3
SAVE_DIR_PATH = "result"
FOLDNUM = 5
SUBJECTNUM = 10
Subjects = [f"s{s}" for s in range(SUBJECTNUM)]

def learn(fn):
    time_sta = time.time()

    if not os.path.exists(SAVE_DIR_PATH):
        os.makedirs(SAVE_DIR_PATH)

    X_train = []  # image data
    Y_train = []  # label
    nom_of_trainpic = 0 # number of images
    for i in range(len(Subjects)):
        train_img_file_name_list = os.listdir("train_data/" + str(fn) + "/" + Subjects[i]) 
        print("{}:have {} pictures for training".format(Subjects[i], len(train_img_file_name_list)))
        nom_of_trainpic = nom_of_trainpic + len(train_img_file_name_list)


        for j in range(0, len(train_img_file_name_list)):
            n = os.path.join("train_data/"+ str(fn) + "/" + Subjects[i], train_img_file_name_list[j])
            img = cv2.imread(n)
            print(img)
            if img is None:
                print('image' + str(j) + ':NoImage')
                continue
            else:
                r, g, b = cv2.split(img)
                img = cv2.merge([r, g, b])
                X_train.append(img)
                Y_train.append(i)
    del train_img_file_name_list
    print("Total number of images for training is " + str(nom_of_trainpic))

    X_val = []  # image data
    Y_val = []  # label
    nom_of_valpic = 0 #number of images
    for i in range(len(Subjects)):
        val_img_file_name_list = os.listdir("val_data/"+ str(fn) + "/" + Subjects[i])
        print("{}:have {} pictures for validation".format(Subjects[i], len(val_img_file_name_list)))
        nom_of_valpic = nom_of_valpic + len(val_img_file_name_list)
        for j in range(0, len(val_img_file_name_list)):
            n = os.path.join("val_data/"+ str(fn) + "/" +Subjects[i], val_img_file_name_list[j])
            img = cv2.imread(n)
            if img is None:
                print('image' + str(j) + ':NoImage')
                continue
            else:
                X_val.append(img)
                Y_val.append(i)
    
    del val_img_file_name_list
    print("Total number of images for validation is : " + str(nom_of_valpic))

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(Y_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(Y_val, dtype=np.float32)
    train_images = X_train
    val_images = X_val

    del X_train,X_val
    train_images, val_images = train_images / 225.0, val_images / 225.0


    model = models.Sequential()
    model.add(layers.Conv2D(64, (FILTERSIZE, FILTERSIZE), activation='relu',input_shape=(96, 96, 3), kernel_initializer="he_uniform"))
    model.add(layers.Conv2D(64, (FILTERSIZE, FILTERSIZE), activation='relu', kernel_initializer="he_uniform"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (FILTERSIZE, FILTERSIZE), activation='relu', kernel_initializer="he_uniform"))
    model.add(layers.Conv2D(128, (FILTERSIZE, FILTERSIZE), activation='relu', kernel_initializer="he_uniform"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (FILTERSIZE, FILTERSIZE), activation='relu', kernel_initializer="he_uniform"))
    model.add(layers.Conv2D(256, (FILTERSIZE, FILTERSIZE), activation='relu', kernel_initializer="he_uniform"))
    model.add(layers.Conv2D(256, (FILTERSIZE, FILTERSIZE), activation='relu', kernel_initializer="he_uniform"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))


    print("=================model.summary=================")
    model.summary()

    print("=================model.compile=================")
    model.compile(optimizer="Adam",
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("=================model.fit=================")
    stack = model.fit(train_images, y_train, epochs=EPOCH,
                      batch_size=2, validation_data=(val_images, y_val), verbose=2)

###############################################################################
#################################train curve###################################
###############################################################################

    fig, [axes1, axes2] = plt.subplots(1, 2, sharex="all", figsize=(8, 4))
    epochs = range(1, EPOCH+1)

    # Changes in loss for each epoch
    axes1.set_title('loss')
    axes1.plot(epochs, stack.history['loss'], label='loss')
    axes1.plot(epochs, stack.history['val_loss'], label='val_loss')
    axes1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes1.legend(['loss', 'val_loss'], loc='upper left')
    axes1.set_xticks(epochs)

    # Changes in accuracy for each epoch
    axes2.set_title('accuracy')
    axes2.plot(epochs, stack.history['accuracy'], label='accuracy')
    axes2.plot(epochs, stack.history['val_accuracy'], label='val_accuracy')
    axes2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes2.legend(['accuracy', 'val_accuracy'], loc='upper left')
    axes2.set_xticks(epochs)

    pic_savepath = SAVE_DIR_PATH + "/train_curve_pic"
    if not os.path.exists(pic_savepath):
        os.makedirs(pic_savepath)
    plt.xticks(np.arange(0, EPOCH+10, 10))
    plt.savefig(pic_savepath + "/train_curve_" + str(fn) + ".png")

###############################################################################
###################################save model##################################
###############################################################################

    model_path = SAVE_DIR_PATH + "/model"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    savemodelpath = model_path + "/model.h5"
    model.save(savemodelpath)

    time_end = time.time()
    tim = time_end - time_sta

    print("")
    print("【Processing time：" + str(tim), "】")
    print("Successed!!!!!")
    print("")



def main():
    for V in range(FOLDNUM):
        print(" ")
        print(" ")
        print(" ")
        print("▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼")
        print(f"Val : {V}")
        print("▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲")
        print(" ")
        learn(V)



if __name__ == "__main__":
    main()