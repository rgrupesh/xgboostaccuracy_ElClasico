#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd
import xgboost as xgb
from IPython.display import display






# In[164]:


data=pd.read_csv(r'C:\Users\HP\Desktop\final.csv')
display(data.head(3))


# In[165]:


#exploring data
n_matches=data.shape[0]
n_features=data.shape[1] - 1
n_homewins=len(data[data.FTR == 'H'])
win_rate= (float(n_homewins) / (n_matches)) *100 
print ("Total number of matches: {}".format(n_matches))
print ("Number of features: {}".format(n_features))
print ("Total homewins :{}".format(n_homewins))
print ("The home win rate is: {:.2f} %".format(win_rate))


# In[166]:


from pandas.plotting import scatter_matrix
scatter_matrix(data[['FTHG','FTAG','HTHG','HTAG','HS']], figsize=(10,10))


# In[167]:


#seperating full time result with other features
X_all = data.drop(['FTR'],1)
y_all = data['FTR']
from sklearn.preprocessing import scale
cols = [['FTHG','FTAG','FTAG','HS','HTAG']]
for col in cols:
    X_all[col] = scale(X_all[col])


# In[168]:



#converting all features into same (integer)type
def preprocess_features(X):
    output = pd.DataFrame(index = X.index)

    for col, col_data in X.iteritems():

        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix = col)
                    
        output = output.join(col_data)
    
    return output

X_all = preprocess_features(X_all)
print ("Processed feature columns {} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))


# In[144]:



print ("\nFeature values:")
display(X_all.head())


# In[169]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, 
                                                    test_size = 0.2,
                                                    random_state =0)


# In[170]:



from time import time 

from sklearn.metrics import f1_score

def train_classifier(clf, X_train, y_train):
    
    

    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
   
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    
    
  
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, average='micro'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    
    
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    

    train_classifier(clf, X_train, y_train)
    
    f1, acc = predict_labels(clf, X_train, y_train)
    print (f1, acc)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(f1 , acc))
   
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(f1 , acc))


# In[171]:


clf_X = xgb.XGBClassifier(seed = 82)

train_predict(clf_X , X_train , y_train , X_test , y_test)
print ('')

