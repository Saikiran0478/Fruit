from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import matplotlib.pyplot as plt
import os
import pandas as pd
from skimage import color
from skimage.feature import graycomatrix, graycoprops
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout
import pickle
from sklearn.decomposition import PCA
import json

main = tkinter.Tk()
main.title("Fruit and LeafDisease Detection")
main.geometry("1000x650")

global ann_model
global ann_model1
global filename
global X, Y
global X_train, X_test, y_train, y_test, testImage

low_green = np.array([25, 52, 72])
high_green = np.array([102, 255, 255])

leaf_labels = [
'Apple Apple_scab','Apple Black_rot','Apple Cedar_apple_rust','Apple healthy',
'Corn_(maize) Cercospora_leaf_spot Gray_leaf_spot',
'Corn_(maize) Common_rust','Corn_(maize) healthy',
'Corn_(maize) Northern_Leaf_Blight','Grape Black_rot',
'Grape Esca_(Black_Measles)','Grape healthy',
'Grape Leaf_blight_(Isariopsis_Leaf_Spot)','Potato Early_blight',
'Potato healthy','Potato Late_blight','Tomato Bacterial_spot',
'Tomato Early_blight','Tomato healthy','Tomato Late_blight',
'Tomato Leaf_Mold','Tomato Septoria_leaf_spot',
'Tomato Spider_mites Two-spotted_spider_mite','Tomato Target_Spot',
'Tomato Tomato_mosaic_virus','Tomato Tomato_Yellow_Leaf_Curl_Virus'
]

fruit_labels = [
'Blotch_Apple','Mango_Alternaria','Mango_Anthracnose',
'Mango_Black Mould Rot','Mango_Healthy','Mango_Stem and Rot',
'Normal_Apple','Pomegranate_Alternaria','Pomegranate_Anthracnose',
'Pomegranate_Bacterial_Blight','Pomegranate_Cercospora',
'Pomegranate_Healthy','Rot_Apple','Scab_Apple'
]

def remove_green_pixels(image):
    channels_first = channels_first_transform(image)
    r_channel = channels_first[0]
    g_channel = channels_first[1]
    b_channel = channels_first[2]
    mask = np.multiply(g_channel > r_channel, g_channel > b_channel)
    channels_first = np.multiply(channels_first, mask)
    image = channels_first.transpose(1, 2, 0)
    return image

def rgb2lab(image):
    return color.rgb2lab(image)

def rgb2gray(image):
    return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False):
    single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
    gclm = greycomatrix(single_channel_image, offsets, angles)
    return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image):
    image = channels_first_transform(image).reshape(3, -1)
    r_channel = image[0]
    g_channel = image[1]
    b_channel = image[2]

    r_hist = np.histogram(r_channel, bins=26, range=(0,255))[0]
    g_hist = np.histogram(g_channel, bins=26, range=(0,255))[0]
    b_hist = np.histogram(b_channel, bins=26, range=(0,255))[0]

    return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
    color_histogram = np.histogram(image.flatten(), bins=255, range=(0,255))[0]

    return np.array([
        np.mean(color_histogram),
        np.std(color_histogram),
        stats.entropy(color_histogram),
        stats.kurtosis(color_histogram),
        stats.skew(color_histogram),
        np.sqrt(np.mean(np.square(color_histogram)))
    ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green=True):
    image = remove_green_pixels(full_image) if remove_green else full_image
    gray_image = rgb2gray(image)
    glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
    return glcm_features(glcmatrix)

def glcm_features(glcm):
    return np.array([
        greycoprops(glcm, 'correlation'),
        greycoprops(glcm, 'contrast'),
        greycoprops(glcm, 'energy'),
        greycoprops(glcm, 'homogeneity'),
        greycoprops(glcm, 'dissimilarity'),
    ]).flatten()

def channels_first_transform(image):
    return image.transpose((2, 0, 1))

def extract_features(image):
    offsets = [1,3,10,20]
    angles = [0, np.pi/4, np.pi/2]

    channels_first = channels_first_transform(image)

    return np.concatenate((
        texture_features(image, offsets=offsets, angles=angles),
        texture_features(image, offsets=offsets, angles=angles, remove_green=False),
        histogram_features_bucket_count(image),
        histogram_features(channels_first[0]),
        histogram_features(channels_first[1]),
        histogram_features(channels_first[2]),
    ))

def loadDataset():
    global filename
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,str(filename) + " loaded\n\n")

def preprocessDataset():
    global X, Y, testImage
    text.delete('1.0', END)

    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')

    X = X.astype('float32')
    X = X / 255

    testImage = X[0].reshape(64,64,3)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    Y = to_categorical(Y)

    text.insert(END,"Image Processing Completed\n\n")
    text.insert(END,"Total images found in dataset: " + str(X.shape[0]))

def segmentation():
    text.delete('1.0', END)
    global X, Y, testImage, pca

    text.insert(END,"Total features available before extraction: " + str(X.shape[1]) + "\n")

    if os.path.exists('model/pca.txt'):
        with open('model/pca.txt','rb') as file:
            pca = pickle.load(file)
        X = pca.fit_transform(X)
    else:
        pca = PCA(n_components=1200)
        X = pca.fit_transform(X)
        with open('model/pca.txt','wb') as file:
            pickle.dump(pca,file)

    text.insert(END,"Total features after extraction: " + str(X.shape[1]) + "\n")

    cv2.imshow("Segmented Image", cv2.resize(testImage,(300,300)))
    cv2.waitKey(0)

def trainANN():
    text.delete('1.0', END)

    global X,Y
    global ann_model, ann_model1

    X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

    text.insert(END,"Dataset Train & Test Split for ANN training\n")
    text.insert(END,"Training Size: " + str(X_train.shape[0]) + "\n")
    text.insert(END,"Testing Size: " + str(X_test.shape[0]) + "\n")

    ann_model = Sequential()
    ann_model.add(Dense(512,input_shape=(X_train.shape[1],)))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))

    ann_model.add(Dense(512))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))

    ann_model.add(Dense(256))
    ann_model.add(Activation('relu'))
    ann_model.add(Dropout(0.3))

    ann_model.add(Dense(len(leaf_labels),activation='softmax'))

    ann_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    history = ann_model.fit(X_train,y_train,batch_size=32,epochs=30,validation_split=0.2)

    score = ann_model.evaluate(X_test,y_test)

    text.insert(END,"Accuracy: %.4f\n" % (score[1]))

def predict():
    global filename, testImage
    text.delete('1.0', END)

    imagePath = askopenfilename(title="Select an Image",
    filetypes=[("Image files","*.jpg *.jpeg *.png")])

    image = cv2.imread(imagePath)
    image = cv2.resize(image,(64,64))

    text.insert(END,"Selected Image Path: " + imagePath + "\n")

    processed_image = extract_features(image).reshape(1,-1)

    prediction1 = ann_model.predict(processed_image)

    predicted_label1 = leaf_labels[np.argmax(prediction1)]

    text.insert(END,"Predicted LeafDisease: " + predicted_label1 + "\n")

def createWidgets():
    global text

    text = Text(main)
    text.pack()

    btnLoad = Button(main,text='Load Dataset',command=loadDataset)
    btnLoad.pack()

    btnProcess = Button(main,text='Process Dataset',command=preprocessDataset)
    btnProcess.pack()

    btnSegmentation = Button(main,text='Feature Extraction and Segmentation',command=segmentation)
    btnSegmentation.pack()

    btnTrain = Button(main,text='Train ANN',command=trainANN)
    btnTrain.pack()

    btnPredict = Button(main,text='Predict',command=predict)
    btnPredict.pack()

createWidgets()
main.mainloop()