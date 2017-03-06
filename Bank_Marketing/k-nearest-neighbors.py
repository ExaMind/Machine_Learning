import  numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, neighbors

#
df = pd.read_csv('../data/bank-additional-full.csv')
df = pd.get_dummies(df) #Converst categorical data to numerical data
print(df)

X = np.array(df.drop(['y_yes'], 1))
y = np.array(df['y_yes'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


