# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the data – Read the CSV file, handle encoding, and split the messages (v2) and labels (v1).
2. Split the dataset – Divide the data into training and test sets.
3. Convert text to numerical features – Use CountVectorizer to transform the messages into a format suitable for machine learning.
4. Train and evaluate model – Fit an SVM classifier on the training data and evaluate its accuracy on the test set.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: LOKESH S
RegisterNumber:  212224230143
*/

import chardet
file = "C:\\Users\\admin\\Downloads\\spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv( "C:\\Users\\admin\\Downloads\\spam.csv", encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

x = data["v2"].values  # messages
y = data["v1"].values  # labels

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```
## Output:
![SVM For Spam Mail Detection](sam.png)
![Screenshot 2025-05-14 084321](https://github.com/user-attachments/assets/37eddc82-3cce-4de8-8a2c-2a7f20f65023)
![Screenshot 2025-05-14 084424](https://github.com/user-attachments/assets/49ed39c9-13b9-4e0c-a426-4e90bc63507f)
![Screenshot 2025-05-14 084527](https://github.com/user-attachments/assets/d413518c-7f96-4b87-a3f7-125e97a2c01f)
![Screenshot 2025-05-14 084611](https://github.com/user-attachments/assets/7c0819bc-1aa6-440b-9340-32ffe022f263)
![Screenshot 2025-05-14 084645](https://github.com/user-attachments/assets/5769b89d-55a2-487f-899c-d1bc541a2841)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
