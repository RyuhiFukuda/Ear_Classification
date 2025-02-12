from re import S
import tensorflow as tf
import os
import cv2
import numpy as np
import time
import keras
from keras.utils.np_utils import to_categorical
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
from tensorflow.python.keras.models import load_model
from tqdm import tqdm
from keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras.utils import array_to_img, img_to_array, load_img, save_img



PX = 96
IMAGE_SIZE = (PX,PX)
MODEL_DIR_PATH = "result/model"
SAVE_DIR_PATH = "result"
SUBJECTNUM = 10
FOLDNUM = 1

LastConvLayreNum = 8 #How many layers the last convolution layer is on from the top (counting from 0)
Subjects = [f"s{s}" for s in range(SUBJECTNUM)]



def test(VAL_VER):

    time_sta = time.time()
    
    if not os.path.exists(SAVE_DIR_PATH):
        os.makedirs(SAVE_DIR_PATH)

    X_test = []  # image data
    Y_test = []  # label
    pic_name_list = [] # test images file name
    Y_test_label = [] # test images label name
    nom_of_testpic = 0 # number of images
    Nom_Of_Persons_Testpic = []
    for i in range(len(Subjects)):
        test_img_file_name_list = os.listdir("test_data/" + str(VAL_VER) + "/" + Subjects[i])
        print("{} has {}pictures for test".format(Subjects[i], len(test_img_file_name_list)))
        Nom_Of_Persons_Testpic.append(len(test_img_file_name_list))
        nom_of_testpic = nom_of_testpic + len(test_img_file_name_list)
        for j in range(0, len(test_img_file_name_list)):
            n = os.path.join("test_data/" + str(VAL_VER) + "/"  + Subjects[i], test_img_file_name_list[j])
            pic_name = os.path.join(str(VAL_VER)+ "/"  + Subjects[i], test_img_file_name_list[j])
            img = cv2.imread(n)
            pic_name_list.append(pic_name)
            if img is None:
                print('image' + str(j) + ':NoImage')
                continue
            else:
                X_test.append(img)
                Y_test.append(i)
                Y_test_label.append(Y_test)


    print("testデータの合計枚数は : " + str(nom_of_testpic) + "枚です。")
   

    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(Y_test, dtype=np.float32)

    test_images = X_test

    # Nomalization
    test_images = test_images / 255.0

    modelpath = MODEL_DIR_PATH + "/model.h5"
    model = load_model(modelpath)


    print("=================model.evaluate=================")
    model.evaluate(test_images,  y_test, verbose=2)

    # =========================Result display=========================
    pred = model.predict(test_images)
    y_pred = np.argmax(pred, axis=1)


    Errata=[]
    TestPicCounter = [0]
    for l in range(len(Subjects)):
        if l > 0:
            TestPicCounter.append(TestPicCounter[l-1] + Nom_Of_Persons_Testpic[l])

        errata=[]
        if l == 0:
            for check in range(Nom_Of_Persons_Testpic[l]):
                if Y_test[check] == y_pred[check]:
                    errata.append(1)
                else:
                    errata.append(0)
        else:
            for check in range(TestPicCounter[l-1],TestPicCounter[l]):
                if Y_test[check] == y_pred[check]:
                    errata.append(1)
                else:
                    errata.append(0)
        Errata.append(errata)
        print(f"{Subjects[l]}'s Errata : {Errata[l]}")

    #▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼Grad-CAM▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    CallGradCam(model, VAL_VER, Errata)
    #▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    print("=========================confusion matrix=========================")
    val_mat = confusion_matrix(y_test, y_pred)
    print(val_mat)

    # accracy
    acc_score = accuracy_score(y_test, y_pred)
    print("")
    print("=========================accracy=========================")
    print("【", acc_score, "】")

    # recall
    rec_score = recall_score(y_test, y_pred, average=None)
    print("")
    print("=========================recall=========================")
    print(rec_score)
    sum_rec_score = 0
    for u in range(len(rec_score)):
        sum_rec_score = sum_rec_score + rec_score[u]
    ave_rec_score = sum_rec_score / (len(rec_score))
    print("【ave_rec_score : ", ave_rec_score, "】")

    # precision
    pre_score = precision_score(y_test, y_pred, average=None)
    # pre_score = precision_score(y_test, y_pred, average='micro')
    print("")
    print("=========================precision=========================")
    print(pre_score)
    sum_pre_score = 0
    for t in range(len(pre_score)):
        sum_pre_score = sum_pre_score + pre_score[t]
    ave_pre_score = sum_pre_score / (len(pre_score))
    print("【ave_pre_score : ", ave_pre_score, "】")

    # F1-Score
    print("")
    print("=========================F=========================")
    f1_scores = (2 * ave_rec_score * ave_pre_score) / (ave_rec_score + ave_pre_score)
    print("【ave_f1_score : ", f1_scores, "】")



    time_end = time.time()

    tim = time_end - time_sta

    print("")
    print("【Processing time：" + str(tim), "】")
    print("")
    print("Successed!!!!!")

    # notify()


    # return EER,CORRECT_LIST
    return acc_score,ave_rec_score,ave_pre_score,f1_scores


def grad_cam(input_model, x, layer_name):
    """
    Args: 
        input_model(object): model object
        x(ndarray): image
        layer_name(string): name of convolution layer
    Returns:
        output_image(ndarray): output image
    """

    X = np.expand_dims(x, axis=0)
    preprocessed_input = X.astype('float32') / 255.0    

    grad_model = models.Model([input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(preprocessed_input)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')

    guided_grads = gate_f * gate_r * grads

    # Weights are averaged and multiplied by the output of the layer
    weights = np.mean(guided_grads, axis=(0, 1))
    cam = np.dot(output, weights)

    # Scale the image to the same size as the original image
    cam = cv2.resize(cam, IMAGE_SIZE, cv2.INTER_LINEAR)

    cam  = np.maximum(cam, 0)
    
    # Calculate heat map
    heatmap = cam / cam.max()


    jet_cam = cv2.applyColorMap(np.uint8(255.0*heatmap), cv2.COLORMAP_JET)
    rgb_cam = cv2.cvtColor(jet_cam, cv2.COLOR_BGR2RGB)
    output_image = (np.float32(rgb_cam) + x / 2)  

    del rgb_cam,jet_cam,heatmap,cam,weights,guided_grads,gate_r,gate_f,grads,output,grad_model,preprocessed_input,X

    return output_image


def CallGradCam(Model, Val_Ver, ERatta):

    for i in range(len(Subjects)):

        # image directory path
        input_dir = "test_data/" + str(Val_Ver) + "/"  + Subjects[i]

        # read files
        files = os.listdir(input_dir)
        # k = 0
        
        # Determine the output directory
        output_dir = SAVE_DIR_PATH + "/grad_cam_data/" + Subjects[i]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # load model
        model = Model

        #▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼Process to obtain the lowest convolution layer▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        for w in model.layers[LastConvLayreNum].weights:
            lastconvlayrename = w.name
        if "/" in lastconvlayrename:
            LastConvLayreName = lastconvlayrename.split('/')
        else:
            print("The layer is not found.")
        del w
        #▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


        #▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼GradCAM▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        if i == 0:
            n=0
            for file in files:
                # load model and images
                image_path = input_dir + "/" + file
                x = img_to_array(load_img(image_path, target_size=IMAGE_SIZE))

                array_to_img(x)

                target_layer = LastConvLayreName[0]
                print(f"target_layer : {target_layer}")
                cam = grad_cam(model, x, target_layer)

                array_to_img(cam)
                if "." in file:
                    File = file.split(".")
                save_img(output_dir + "/" + File[0] + "-" + str(ERatta[i][n]) + "." + File[1],cam)
                del x,cam,target_layer,image_path
                n += 1
        else:

            n = 0
            for file in files:
                image_path = input_dir + "/" + file
                x = img_to_array(load_img(image_path, target_size=IMAGE_SIZE))

                array_to_img(x)
                target_layer = LastConvLayreName[0]
                cam = grad_cam(model, x, target_layer)


                array_to_img(cam)
                if "." in file:
                    File = file.split(".")

                save_img(output_dir + "/" + File[0] + "-" + str(ERatta[i][n]) + "." + File[1],cam)
                del x,cam,target_layer,image_path
                n += 1
        del files,model
        #▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


def main():
    Coulum = ["Train_Test","EER","ACC","REC","PRE","F1"]
    Table = []
    Last_ACC = []
    Last_REC = []
    Last_PRE = []
    Last_F1 = []
    for V in range(FOLDNUM):
        A,R,P,F = test(V)
        Last_ACC.append(A)
        Last_REC.append(R)
        Last_PRE.append(P)
        Last_F1.append(F)

    print("Last_ACC =",Last_ACC)
    print("Last_REC =",Last_REC)
    print("Last_PRE =",Last_PRE)
    print("Last_F1 =",Last_F1)
    Average_ACC = sum(Last_ACC)/len(Last_ACC)
    Average_REC = sum(Last_REC)/len(Last_REC)
    Average_PRE = sum(Last_PRE)/len(Last_PRE)
    Average_F1 = sum(Last_F1)/len(Last_F1)

    print("Average_ACC =",Average_ACC)
    print("Average_REC =",Average_REC)
    print("Average_PRE =",Average_PRE)
    print("Average_F1 =",Average_F1)
    Table.append(Average_ACC)
    Table.append(Average_REC)
    Table.append(Average_PRE)
    Table.append(Average_F1)
    Table.append(Table)
    df = pd.DataFrame(Table, columns=Coulum)
    df.to_csv("test.csv")

if __name__ == "__main__":
    main()