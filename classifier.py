#classifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[300]:

def classifier(tf,df,train_process_filename):#takes as input, training and testing dataframe containing all features

    with open(train_process_filename,'r') as f:
        str=f.read()
        train_pro=str.split('\t')
    f1gold=df[df['Process'].isin(train_pro)].reset_index()
    e=f1gold['Escore']
    c=f1gold['Cscore']
    l=f1gold['True_label']
    left=f1gold['lf_Cscore']
    right=f1gold['rf_Cscore']
    pos=f1gold['POSsim']
    postt=tf['POSsim']
    leftt=tf['lf_Cscore']
    rightt=tf['rf_Cscore']
    l=np.ravel(l)
    et=tf['Escore']
    ct=tf['Cscore']
    lt=tf['True_label']

    X = np.array([e,c,left,right,pos]).T
    Xt = np.array([et,ct,leftt,rightt,postt]).T
    lt=np.ravel(lt)
    #print Xt

    clfs = [ RandomForestClassifier(n_estimators=100, n_jobs=2)]
    clf_names = [ 'RandomForestClassifier']

    results = {}
    for (i, clf_) in enumerate(clfs):

        clf = clf_.fit(X,l)
        preds = clf.predict(Xt)
        predicted=clf.predict_proba(Xt)



        precision = metrics.precision_score(lt, preds)
        recall = metrics.recall_score(lt, preds)
        f1 = metrics.f1_score(lt, preds)


        accuracy = accuracy_score(lt, preds)
        scores = cross_val_score(clf, X, l, scoring='accuracy', cv=10)
        scoremean=scores.mean()

        # matrix = metrics.confusion_matrix(Ytst, preds, labels=list(set(labels)))

        data = {'precision':precision,
                'recall':recall,
                'f1_score':f1,
                'accuracy':accuracy,
                # 'clf_report':report,
                # 'clf_matrix':matrix,
                #'y_predicted':preds
                'Cross Validation Score':scoremean}

        results[clf_names[i]] = data
    cols = ['precision', 'recall', 'f1_score', 'accuracy','Cross Validation Score']
    results_df=pd.DataFrame(results).T[cols].T
    print results_df
    #print predicted
    probc=predicted[:,0]
    tf['Classification_result']=preds
    tf['Probability of result']=probc
    for i in list(tf.index.get_values()):
        if int(tf.loc[i,'Classification_result'])==1:

            tf.loc[i,'probability_align']=tf.loc[i,'Probability of result']
        elif int(tf.loc[i,'Classification_result'])==0:

            tf.loc[i,'probability_align']=1-tf.loc[i,'Probability of result']
    return tf



