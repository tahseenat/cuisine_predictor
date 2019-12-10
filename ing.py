import csv
import pandas as pd
file_name = 'ingredients.csv'
ing = pd.read_csv(file_name, encoding="ISO-8859-1", usecols=range(0, 3))
print(ing.head)
