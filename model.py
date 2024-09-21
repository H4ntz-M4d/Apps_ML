import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle
df = pd.read_csv('Mesin_Data.csv')

column_to_transform = ['Jam Operasi']

# outliers handler (mean)
def outliers_mean(df, column_name):
  while True:
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
    if len(outliers) == 0:
      break
    mean = df[column_name].mean()
    df.loc[outliers.index, column_name] = mean
  return df

df = outliers_mean(df, 'Jam Operasi')

# set X and Y
X = df.drop(columns=['ID Mesin', 'Kegagalan'], axis=1)
y = df['Kegagalan'].map({'TIDAK': 0, 'YA': 1})

# apply into train test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=81, stratify=y)

# knn model
knn = KNeighborsClassifier()

cv_scores = cross_val_score(knn, X, y, cv=10)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")
print(f"Standard Deviation of Cross-Validation Scores: {np.std(cv_scores)}")

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
correct_predictions = cm[0][0] + cm[1][1]
incorrect_predictions = cm[0][1] + cm[1][0]

print(accuracy)

with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn, model_file)

def load():
  return accuracy, correct_predictions, incorrect_predictions