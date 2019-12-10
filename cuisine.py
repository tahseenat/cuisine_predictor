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

ing = pd.read_csv(file_name, encoding="ISO-8859-1", usecols=range(0, 3))
print(ing.head)
id_ = [x for x in ing["id"]]
cuisine = [x for x in ing["cuisine"]]
ingredients = [x for x in ing["ingredients"]]

import pandas as pd

df = pd.DataFrame({'id': id_,
                   'cuisine': cuisine,
                   'ingredients': ingredients})
print(df.head(5))
df['ingredients']
df['ing_mod'] = df['ingredients']

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ing_mod'])

print(X)
# print(vectorizer.get_feature_names())

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(df['cuisine'])
df['cuisine'] = le.transform(df['cuisine'])

Y = []
Y = df['cuisine']

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
for K in range(25):
    K_value = K + 1
    neigh = KNeighborsClassifier(n_neighbors=K_value, weights='distance', algorithm='auto',  n_jobs=-1)
    neigh.fit(X_train, y_train)
    y_pred = neigh.predict(X_test)
    print("Accuracy is ", accuracy_score(y_test, y_pred) * 100, "% for K-Value:", K_value)

K_value = 16
neigh = KNeighborsClassifier(n_neighbors=K_value, weights='distance', algorithm='auto', n_jobs=-1)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
print("Accuracy is ", accuracy_score(y_test, y_pred) * 100, "% for K-Value:", K_value)
