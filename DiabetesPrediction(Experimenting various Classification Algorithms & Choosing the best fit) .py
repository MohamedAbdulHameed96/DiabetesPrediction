# Experimenting various Classification Algorithms & Choosing the best fit

  
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from scipy.stats import skew
#%matplotlib inline

  
data=pd.read_csv("diabetes-dataset.csv")

  
data.head()

  
#Checking null values
data.info()

  
data.shape

  
data.describe()


# # Visualizing the data

  
sns.pairplot(data, x_vars=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'],y_vars='Outcome',kind="scatter",height=2,aspect=0.7)

  
#The above plot shows that attributes have squiggle relationship.

  
sns.heatmap(data.corr(),annot=True)

  
#From the above correlation plot its evident that increase in age & Preganancy are associated with diabetes


# # Splitting the features and the target

  
X=data.drop("Outcome",axis=1)
Y=data["Outcome"]

  
X.head()

  
Y.head()

  
# # Splitting the training and test data

  
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=1)

  
# # LogisticRegression model

  
model=LogisticRegression(max_iter=50000)
model.fit(X_train,Y_train)

  
# # LogisticRegression Accuracy

  
#Training data accuracy
XtrainPred=model.predict(X_train)
trainingDataAccuracy=accuracy_score(XtrainPred,Y_train)
print("Accuracy of the training data :",trainingDataAccuracy)

  
#Accuracy of test data
XtestPred=model.predict(X_test)
testDataAccuracy=accuracy_score(XtestPred,Y_test)
print("Accuracy of the test data is :",testDataAccuracy)

  
# # kNeighbors Classifier model

  
model=KNeighborsClassifier()
model.fit(X_train,Y_train)

  
# # kNeighbors Classifier model Accuracy

  
#Training data accuracy
XTrainPred=model.predict(X_train)
TrainDataAccuracy=accuracy_score(XTrainPred,Y_train)
print("The accuracy of Training data is : ",TrainDataAccuracy)

  
#Accuracy of test data
XtestPred=model.predict(X_test)
testDataAccuracy=accuracy_score(XtestPred,Y_test)
print("Accuracy of the test data is :",testDataAccuracy)

  
# # DecisionTreeClassifier model

  
model=DecisionTreeClassifier()
model.fit(X_train,Y_train)

  
# # DecisionTreeClassifier model Accuracy

  
#Training data accuracy
XTrainPred=model.predict(X_train)
TrainDataAccuracy=accuracy_score(XTrainPred,Y_train)
print("The accuracy of Training data is : ",TrainDataAccuracy)

  
#Accuracy of test data
XtestPred=model.predict(X_test)
testDataAccuracy=accuracy_score(XtestPred,Y_test)
print("Accuracy of the test data is :",testDataAccuracy)

  
# # RandomForestClassifier model

  
model=RandomForestClassifier()
model.fit(X_train,Y_train)

  
# # RandomForestClassifier model Accuracy

  
#Training data accuracy
XTrainPred=model.predict(X_train)
TrainDataAccuracy=accuracy_score(XTrainPred,Y_train)
print("The accuracy of Training data is : ",TrainDataAccuracy)

  
#Accuracy of test data
XtestPred=model.predict(X_test)
testDataAccuracy=accuracy_score(XtestPred,Y_test)
print("Accuracy of the test data is :",testDataAccuracy)

  
# # Building a User Interface to predict whether the patient as Diabetes or not

  
#As the Decision Tree Classifier has high Accuracy(98.5%). Lets choose Decision Tree Model over the other models.

  
# # model=DecisionTreeClassifier()
# model.fit(X_train,Y_train)

  
user_input=(0,179,50,36,159,37.8,0.455,22)

#changing the user_input to numpy array
userInputArray=np.asarray(user_input)
#reshaping the numpy array
userInputReshaped=userInputArray.reshape(1,-1)
prediction=model.predict(userInputReshaped)
print(prediction)
if(prediction[0]==1):
  print("THIS PATIENT HAS DIABETES")
else:
    print("THIS PATIENT DOES NOT HAVE DIABETES")

  
# # Finally, We got to know that the Decision Tree Classifier gives us the best output with high accuracy of around 98.5%.
# # Or take RandomForest Classifier

  



