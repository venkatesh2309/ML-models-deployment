import pandas as pd
import numpy as np
##import matplotlib.pyplot as plt
##import seaborn as sns
##%matplotlib inline

import  streamlit as st
##st.header('# Diabetes Prediction')
st.write(""" 
         
     #    DIABETES PREDICTOR
         
   """)
st.markdown("### An Machine Learning Web App, built with Streamlit")

data_df=pd.read_csv('diabetes.csv')
st.sidebar.header('Please Provide Below Information')


##data_df

data_df.info()

## lets make some EDA to understand each feature with frespect to the dependent feature
##sns.pairplot(data_df,hue='Outcome')


## as Far we analyzed 

## 1.The persons with high glucose mostly fall in diabetes 
##2. The Age feature does not making an impact on diabetes where more or less all the age categories have been affected
##3. The persons with high BMI has making a significant impact for the diabetes.

## lets go for coorelation
##correlation=data_df.corr()

##correlation

## observation: As we can observe that we can able to segrate the classification with a line we can go for logistic/svm.
## Aslo we try with KNN with feature scaling

data_df.describe()

## feature engineering Lets us try to categorize the feature for prediction.

data_df.isnull().sum()
## let us go for feature scaling 
data_knn= data_df.copy()
x=data_knn.drop('Outcome',axis=1)
y=data_knn['Outcome']

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
xtrans=sc.fit_transform(x)

df=pd.DataFrame(data=xtrans,columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])


x1=df
y=data_knn['Outcome']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

##x_test.shape,y_test.shape

# KNN or k-Nearest Neighbors
##from sklearn.neighbors import KNeighborsClassifier
##from sklearn.metrics import accuracy_score

##knn = KNeighborsClassifier(n_neighbors=13)
##knn.fit(x_train, y_train)
##y_pred = knn.predict(x_test)
##acc_knn = round(accuracy_score(y_test, y_pred) * 100, 2)
##print(acc_knn)

## let us look for best K
##from sklearn.model_selection import cross_val_score
##overall=[]
##for i in range(1,40):
    ##knn2=KNeighborsClassifier(n_neighbors=i)
    ##score=cross_val_score(knn2,x1,y,cv=10)
    ##overall.append(score.mean())
    
    
##plt.plot(range(1,40),overall,marker='o')
  
from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier(n_estimators=100)
random.fit(x_train,y_train)
y_pred=random.predict(x_test)
acc_rn = round(accuracy_score(y_test, y_pred) * 100, 2)
print(acc_rn)
    

def input_features():
    preg = st.sidebar.slider('NO of pregencies',0,20,1)
    glu = st.sidebar.slider('Glucose',20,500,180)
    bp = st.sidebar.slider('Blood pressure (mm Hg)',0,500,50)
    skint= st.sidebar.slider('Skin Thickness (mm)',0,500,50)
    su=st.sidebar.slider(' Insulin(U/ml)',0,500,50)
    bmi= st.sidebar.slider('Body mass index (weight in kg/(height in m)^2)',0,500,50)
    dpf= st.sidebar.slider('Diabetes pedigree function',0.0,2.0,0.5)
    age  = st.sidebar.slider('Age (years)',10,80,25)
    data={'pregencies': preg,'glucose':glu,'bp':bp,'Skin thickness':skint,'Insulin':su,'BMI':bmi,'DBF':dpf,'Age':age}
    
    features=pd.DataFrame(data,index=[0])
    return features
        
        
        
    
    
        
ev=input_features()
st.info("BELOW ARE THE PROVIDED INFORMATION")
st.write(ev)
##sc=StandardScaler()
##xtranspred=sc.fit_transform(ev)
prediction=random.predict(ev)
##prediction2=random.predict_proba(ev)


st.markdown("### PREDICTION")
if prediction == 1:
    st.warning('OOPS **YOU HAVE DIABETES...**')
elif prediction == 0:
    st.success('KUDOS YOU **DONT HAVE DIABETES...**')
    st.balloons()
    
    
##print(__name__)
##if __name__ == '__main__':
   ##st.write(prediction)
   ##st.write(prediction2)
    
    
        

	
    
    
    
    




    
    
    


    







