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

EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
file_name = 'ingredients.csv'

ing = pd.read_csv(file_name, encoding="ISO-8859-1", usecols=range(0, 3))
print(ing.head)
clean_ing = [x for x in ing["ingredients"]]
labels = [x for x in ing["cuisine"]]

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(clean_ing, labels, test_size=0.3, random_state=100)
