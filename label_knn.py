import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_name = 'ingredients.csv'

df = pd.read_csv(file_name, encoding="ISO-8859-1", usecols=range(0, 3))
print(df.head)
df['ingredients']

grocery_file = 'groceries.csv'
dfg = pd.read_csv(grocery_file, encoding="ISO-8859-1")
dfg['item']

temp = []
for i in range(len(df['ingredients'])):
    temp.append( df['ingredients'][i])

for i in range(len(dfg['item'])):
    temp.append( dfg['item'][i])


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_total = vectorizer.fit_transform(temp)


gro_ind = 39774
X = X_total[:gro_ind]
print(X)

X_label = X_total[gro_ind:]

Y = []
Y = df['cuisine']

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

neigh = KNeighborsClassifier(n_neighbors=16, weights='distance', algorithm='auto',  n_jobs=-1, p=2)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
y_prob = neigh.predict_proba(X_test)
print("Accuracy is ", accuracy_score(y_test, y_pred) * 100, "% for K-Value:", 16)

label_pred = neigh.predict(X_label)
label_pred_prob = neigh.predict_proba(X_label)
# use label_pred_prob to find out two most probable cuisine and write them in groceries.csv file

