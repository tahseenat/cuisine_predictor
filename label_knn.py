import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# file to be used to read
file_name = 'ingredients.csv'
grocery_file = 'groceries.csv'

# no. of rows in the ingredients.csv
ing_ind = 39774
#  train 70% and Test on 30%
test_split = 0.3
# randomness for consistent result
randomness = 100

# read ingredient file
df = pd.read_csv(file_name, encoding="ISO-8859-1", usecols=range(0, 3))
print(df.head(5))

# read groceries file
dfg = pd.read_csv(grocery_file, encoding="ISO-8859-1")

temp = []
for i in range(len(df['ingredients'])):
    temp.append(df['ingredients'][i])

for i in range(len(dfg['item'])):
    temp.append(dfg['item'][i])

# tokenize bag of words
vectorizer = TfidfVectorizer()
X_total = vectorizer.fit_transform(temp)

# split the data for training and testing
X = X_total[:ing_ind]

# prepare the labels
Y = df['cuisine']

# split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_split, random_state=randomness)


# implement KNN
neigh = KNeighborsClassifier(n_neighbors=16,
                             weights='distance',
                             algorithm='auto',
                             leaf_size=30,
                             p=2,
                             metric='minkowski',
                             metric_params=None,
                             n_jobs=-1)
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)
y_prob = neigh.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("Accuracy: {}".format(acc))

# use this KNN to predict label for groceries
X_label = X_total[ing_ind:]
label_pred = neigh.predict(X_label)
label_pred_prob = neigh.predict_proba(X_label)

# use label_pred_prob to find out two most probable cuisine and write them in groceries.csv file
# Index for Cuisines
ind = ['brazilian', 'british', 'cajun_creole', 'chinese', 'filipino', 'french', 'greek', 'indian', 'italian',
       'jamaican', 'japanese', 'korean', 'mexican', 'moroccan', 'russian',
       'southern_us', 'spanish', 'thai', 'vietnamese', 'irish', 'none']
probability_array = []

# finding two best probabilities
for i in range(len(label_pred_prob)):
    tmp_list = list(label_pred_prob[i])
    x = tmp_list.index(max(tmp_list))
    ma = max(tmp_list)
    y = 0
    for j in range(len(tmp_list)):
        if y < tmp_list[j] < ma:
            y = tmp_list[j]
    if y > 0.1:
        y = tmp_list.index(y)
    else:
        y = 20
    # storing two probabilities in a list
    probability_array.append(ind[x])
    probability_array.append(ind[y])

file_name = 'groceries.csv'

# writing cuisines names to CSV
with open(file_name, 'r') as csvinput:
    with open('labeled_groceries.csv', 'w', newline='') as csvoutput:
        writer = csv.writer(csvoutput)
        count = 0
        i = 0
        for row in csv.reader(csvinput):
            if count == 0:
                writer.writerow(['Cuisine 1', "Cuisine 2"] + row)
            else:
                writer.writerow([probability_array[i], probability_array[i + 1]] + row)
                i = i + 2
            count += 1
            # print(probability_array[i],probability_array[i+1])
