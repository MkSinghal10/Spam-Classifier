import numpy as np
import pandas as pd
data=pd.read_csv('spam.csv',encoding='ISO-8859-1')
dt=data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis='Column')

import re
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]

for i in range(0, len(dt)):
    review=re.sub('[^a-zA-Z]',' ',dt['message'][i])
    review=review.lower()
    review=review.split()
    
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
from imblearn.over_sampling import RandomOverSampler
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500)
X=cv.fit_transform(corpus).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
cv1=TfidfVectorizer(ngram_range=(3,3),max_features=2500)
X1=cv1.fit_transform(corpus).toarray()

import pickle 
pickle.dump(cv1, open('tranform.pkl', 'wb'))

dt['class'] = dt['class'].map({'ham': 0, 'spam': 1})
Y=dt['class']
#Y=pd.get_dummies(dt['class'])
#Y=Y.iloc[:,1].values
#ros=RandomOverSampler(sampling_strategy='minority',random_state=32)
#X_resample,Y_resample=ros.fit_resample(X1,Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X1,Y,test_size=0.33,random_state=42)

from sklearn.naive_bayes import MultinomialNB
Spam_model=MultinomialNB().fit(X_train,Y_train)

Y_pred=Spam_model.predict(X_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



pickle.dump(Spam_model, open('modelPred.pkl', 'wb'))
