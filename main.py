from tkinter import *
from tkinter import filedialog
import numpy as np
import cv2
import pickle
import os

from utils.features import extract_features
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

main = Tk()
main.title("Fruit and Leaf Disease Detection")
main.geometry("800x600")

X, Y = None, None
model = None
pca = None

leaf_labels = [
'Apple Apple_scab','Apple Black_rot','Apple Cedar_apple_rust','Apple healthy',
'Corn Cercospora','Corn Rust','Corn healthy','Corn Blight',
'Grape Black_rot','Grape Esca','Grape healthy','Grape Leaf_blight',
'Potato Early','Potato healthy','Potato Late',
'Tomato Bacterial','Tomato Early','Tomato healthy','Tomato Late'
]

text = Text(main)
text.pack()

def load_dataset():
    global X, Y
    text.delete('1.0', END)

    X = np.load('model/X.txt.npy')
    Y = np.load('model/Y.txt.npy')

    X = X.astype('float32') / 255
    Y = to_categorical(Y)

    text.insert(END, f"Dataset loaded: {X.shape[0]} samples\n")

def apply_pca():
    global X, pca
    text.insert(END, "Applying PCA...\n")

    if os.path.exists('model/pca.txt'):
        pca = pickle.load(open('model/pca.txt','rb'))
        X = pca.transform(X)
    else:
        pca = PCA(n_components=500)
        X = pca.fit_transform(X)
        pickle.dump(pca, open('model/pca.txt','wb'))

    text.insert(END, f"Features reduced to {X.shape[1]}\n")

def train_model():
    global model

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(Y.shape[1], activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32)

    loss, acc = model.evaluate(X_test, y_test)
    text.insert(END, f"Accuracy: {acc:.4f}\n")

def predict():
    global model, pca

    file = filedialog.askopenfilename()
    img = cv2.imread(file)
    img = cv2.resize(img, (64,64))

    features = extract_features(img).reshape(1,-1)

    if pca:
        features = pca.transform(features)

    pred = model.predict(features)
    label = leaf_labels[np.argmax(pred)]

    text.insert(END, f"Prediction: {label}\n")

Button(main, text="Load Dataset", command=load_dataset).pack()
Button(main, text="Apply PCA", command=apply_pca).pack()
Button(main, text="Train Model", command=train_model).pack()
Button(main, text="Predict", command=predict).pack()

main.mainloop()
