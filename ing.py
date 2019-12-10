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

from clean_tweet_fun import clean_tweets
from model import brain

warnings.filterwarnings("ignore")

# Reproducibility
np.random.seed(1234)

RANDOM_NROWS = 38000
MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 3200
EMBEDDING_DIM = 300

# Splitting the arrays into test (70%), validation (20%), and train data (20%)
TRAIN_SPLIT = 0.7
TEST_SPLIT = 0.2
LEARNING_RATE = 0.1
EPOCHS = 15

EMBEDDING_FILE = 'GoogleNews-vectors-negative300.bin'

word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Applying the pre processing clean_text function to every element in the depressive tweets and random tweets data.
file_name = 'ingredients.csv'

ing = pd.read_csv(file_name, encoding="ISO-8859-1", usecols=range(0, 3))
print(ing.head)

# Applying the pre processing clean_text function to every element in the depressive tweets and random tweets data.

clean_ing = [x for x in ing["ingredients"]]

X_c = clean_tweets(clean_ing)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(X_c)

sequences_c = tokenizer.texts_to_sequences(X_c)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_c = pad_sequences(sequences_c, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of data_r tensor:', data_c.shape)

nb_words = min(MAX_NB_WORDS, len(word_index))

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))

for (word, idx) in word_index.items():
    if word in word2vec.vocab and idx < MAX_NB_WORDS:
        embedding_matrix[idx] = word2vec.word_vec(word)

# Assigning labels to the depressive tweets and random tweets data

labels = [x for x in ing["cuisine"]]

data_train = data_c[0:int(RANDOM_NROWS * TRAIN_SPLIT)]
labels_train = labels[0:int(RANDOM_NROWS * TRAIN_SPLIT)]

data_test = data_c[int(RANDOM_NROWS * TRAIN_SPLIT):int(RANDOM_NROWS * (TEST_SPLIT + TRAIN_SPLIT))]
labels_test = labels[int(RANDOM_NROWS * TRAIN_SPLIT):int(RANDOM_NROWS * (TEST_SPLIT + TRAIN_SPLIT))]

data_val = data_c[int(RANDOM_NROWS * (TEST_SPLIT + TRAIN_SPLIT)):RANDOM_NROWS]
labels_val = labels[int(RANDOM_NROWS * (TEST_SPLIT + TRAIN_SPLIT)):RANDOM_NROWS]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
# Decision tree
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(data_train, labels_train)
pred = clf.predict(data_test)
matrix = confusion_matrix(labels_test, pred)
print(matrix)
print(classification_report(labels_test, pred))
print(roc_auc_score(labels_test, pred))
# base model


model = brain(embedding_matrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH)
print(model.summary())

early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])

hist = model.fit(data_train, labels_train, validation_data=(data_val, labels_val), epochs=EPOCHS, batch_size=100,
                 shuffle=True, callbacks=[early_stop])

labels_c_pred = model.predict(data_test)

labels_pred = np.round(labels_c_pred.flatten())

accuracy = accuracy_score(labels_test, labels_pred)
print("Accuracy: %.2f%%" % (accuracy * 100))

print(classification_report(labels_test, labels_pred))

model_name = "with_5000_dataset_acc={}".format(accuracy)
model.save(model_name)

import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

for K in range(25):
    K_value = K + 1
    neigh = KNeighborsClassifier(n_neighbors=K_value, weights='uniform', algorithm='auto')
    neigh.fit(data_train, labels_train)
    y_pred = neigh.predict(data_test)
    print("Accuracy is ", accuracy_score(labels_test, y_pred) * 100, "% for K-Value:", K_value)
