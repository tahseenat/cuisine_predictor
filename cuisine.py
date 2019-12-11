import csv
import pandas as pd
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from gensim.models import KeyedVectors

from keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

file_name = 'ingredients.csv'

df = pd.read_csv(file_name, encoding="ISO-8859-1", usecols=range(0, 3))
print(df.head)
df['copy_ingredients'] = df['ingredients']

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['copy_ingredients'])
print(X)

Y = []
Y = df['cuisine']

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

for K in range(25):
    K_value = K + 1
    neigh = KNeighborsClassifier(n_neighbors=K_value, weights='distance', algorithm='auto',  n_jobs=-1)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100, "% for K-Value:", K_value)
neigh.predict_proba(X_test)

neigh = KNeighborsClassifier(n_neighbors=16, weights='distance', algorithm='auto',  n_jobs=-1, p=2)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
y_prob = neigh.predict_proba(X_test)
print("Accuracy is ", accuracy_score(y_test, y_pred) * 100, "% for K-Value:", K_value)

