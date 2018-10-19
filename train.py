import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

df = pd.read_csv('wakatigakidone.csv', encoding = "shift-jis")

X = df['body'].values
y = df['total']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(X_train, X_test)

X_train_bow = vectorizer.transform(X_train)
X_test_bow = vectorizer.transform(X_test)

X_train_array = X_train_bow.toarray()
X_test_array = X_test_bow.toarray()

def model_train(X_train_array,  y_train):
    model = Sequential()
    model.add(Conv2D(32,(3,3), padding='same', input_shape = (2000,15756,500)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
    model.fit(X, y, batch_size=32, epochs=10)

    # モデルの保存
    model.save('./gourmet_cnn.h5')


    return model

def model_eval(model, X_test_array, y_test):
    scores = model.evaluate(X_test_array, y_test, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

model = model_train(X_train_array, y_train)
model_eval(model, X_test_array, y_test)
