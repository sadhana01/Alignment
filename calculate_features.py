import math
import pandas as pd
import ilp_utils
import word2vec
import numpy as np
from ilp_utils import *
from word2vec import *


def jaccard_similarity(x,y):

    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)
def cosine_similarity(x,y):
    cs = np.dot(x,y)/(np.linalg.norm(x)* np.linalg.norm(y))
    score=(cs+1)/2
    return score

def get_rf_cosine_similarity(df):
    w=Word2VecModel()

    for j in list(df.index.get_values()):

        try:
            if type(df.loc[j,'rf1'])==float:
                if math.isnan(df.loc[j,'rf1']) :
                    v1=[]
            else:
                df.loc[j,'rf1']=df.loc[j,'rf1'].replace('_',' ')
                v1=w.get_sent_vector(df.loc[j,'rf1'])
            if type(df.loc[j,'rf2'])==float:
                if math.isnan(df.loc[j,'rf2']):
                    v2=[]
            else:
    
                    df.loc[j,'rf2']=df.loc[j,'rf2'].replace('_',' ')
                    v2=w.get_sent_vector(df.loc[j,'rf2'])
            if v1!=[] and v2!=[]:
                score=cosine_similarity(v1,v2)
            else:
                score=0
        except ValueError:
            score=0
        df.loc[j,'rf_Cscore']=score
    return df
def get_lf_cosine_similarity(df):
    w=Word2VecModel()

    for j in list(df.index.get_values()):

        try:
            if type(df.loc[j,'lf1'])==float:
                if math.isnan(df.loc[j,'lf1']) :
                    v1=[]
            else:
                df.loc[j,'lf1']=df.loc[j,'lf1'].replace('_',' ')
                v1=w.get_sent_vector(df.loc[j,'lf1'])
            if type(df.loc[j,'lf2'])==float:
                if math.isnan(df.loc[j,'lf2']):
                    v2=[]
            else:

                    df.loc[j,'lf2']=df.loc[j,'lf2'].replace('_',' ')
                    v2=w.get_sent_vector(df.loc[j,'lf2'])
            if v1!=[] and v2!=[]:
                score=cosine_similarity(v1,v2)
            else:
                score=0
        except ValueError:
            score=0
        df.loc[j,'lf_Cscore']=score
    return df

def get_arg_cosine_simialrity(df):
    w=Word2VecModel()
    for j in list(df.index.get_values()):
        try:
                v1=w.get_sent_vector(df.loc[j,'Arg1'])
                v2=w.get_sent_vector(df.loc[j,'Arg2'])
                score=cosine_similarity(v1,v2)
        except ValueError:
                score=np.nan



        df.loc[j,'Cscore']=score
    return df
def get_entailment_score(df):
    for j in list(df.index.get_values()):

        df.loc[j,'Escore']=get_similarity_score(df.loc[j,'Arg1'],df.loc[j,'Arg2'])
    return df

def get_pos_similarity(df):
    for i in list(df.index.get_values()):
        df.loc[i,'POSsim']=jaccard_similarity(df.loc[i,'POStag1'],df.loc[i,'POStag2'])

