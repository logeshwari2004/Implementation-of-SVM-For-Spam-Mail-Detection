# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect a labeled dataset of emails, distinguishing between spam and non-spam.

2.Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.

3.Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.

4.Split the dataset into a training set and a test set.

5.Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.

6.Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.

7.Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.

8.Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.

9.Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed. 

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Logeshwari.P
RegisterNumber:212221230055
```
```

import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
## RESULT OUTPUT:
![image](https://github.com/Mythilidharman/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104110/0ec7d672-7118-4d39-a751-8e411ba67b27)
## data.head()
![image](https://github.com/Mythilidharman/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104110/5eadaf7c-ee33-4361-a9b8-24af5923dd40)
## data.info()
![image](https://github.com/Mythilidharman/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104110/14153aab-995a-4d45-9acc-c4816840fed3)
## data.isnull().sum()
![image](https://github.com/Mythilidharman/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104110/d43598c6-f707-4587-92e4-79ea2212eb51)
## Y_Prediction value
![image](https://github.com/Mythilidharman/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104110/254ee274-15d7-492a-89ec-48254b96624f)
## Accuracy value
![image](https://github.com/Mythilidharman/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119104110/b5a40f55-9c43-4d3e-9616-eafd1c48c42a)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
