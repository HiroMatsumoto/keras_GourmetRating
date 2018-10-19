import keras, sys
import numpy as np
import pandas as pd
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import keras
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Activation, Dropout, Dense

df = pd.read_csv('./wakatigakidone2.csv', encoding = "shift-jis")
X = df['body'].values

vectorizer = pickle.load(open("feature.pkl", "rb"))

X_bow = vectorizer.transform(X)
X_array = X_bow.toarray()
model = load_model('./gourmet1_cnn.h5')

predicted = model.predict(X_array, batch_size = 1)
print(predicted)
