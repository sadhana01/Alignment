import ilp_utils

from ilp_utils import *
import json
import pandas as pd
import itertools
import numpy as np
#program to compute the sentences dataframe from the data.also includes ov1- index where argument and sentence overlap.This helps for finding neighbouring frames

#returns sentences dataframe
def extract_testsen(json_filename):
    data=load_srl_data(json_filename)
    sen_t1=pd.DataFrame(columns=['sen_id','Sentence','Process','arg_id','Arg','role'])
    c=0

    for p in data:

            for i in data[p][1]:

                if data[p][1][i]==[]:
                    #print "Breaking",i
                    continue
                #print data[p]
                for n in data[p][1][i]:
                        #print n
                        sen_t1.loc[c,'Process']=p
                        sen_t1.loc[c,'Arg']=n[1]
                        sen_t1.loc[c,'sen_id']=i
                        sen_t1.loc[c,'arg_id']=int(n[0])
                        id1=(i,int(n[0]))
                        sen_t1.loc[c,'role']=data[p][3][id1][0]
                        for d in data[p][0]:
                            if data[p][0][d]==i:
                                sen_t1.loc[c,'Sentence']=d

                        c=c+1

    #optional:delete spaces near special character
    for i in list(sen_t1.index.get_values()):

            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace('-LRB- ','(')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace(' -RRB-',')')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace(' ,',',')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace('-LSB- ','[')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace(' -RSB-',']')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace(' \'' , '\'')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace(' ;',';')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace(' .','.')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace(' - ','- ')
            sen_t1.loc[i,'Arg']=sen_t1.loc[i,'Arg'].replace(' :',':')

            sen_t1.loc[i,'o1']=sen_t1.loc[i,'Sentence'].find(sen_t1.loc[i,'Arg'])
    #sen_t1.to_csv('NEWSENDATAQA.csv',sep='\t')
    return sen_t1

def extract_goldsen(json_filename):
    d_gold = json.load(open(json_filename, "r"))
    gold_data =get_gold_data(d_gold)
    sentences=pd.DataFrame(columns=['sen_id','Sentence','Process','Arg','role'])
    n=0
    #getting sentence data
    data=load_srl_data(json_filename)
    s={}
    for p in data:
        s[p]=data[p][4]



    for i in gold_data :

        sentences.loc[n,'sen_id']=i[0]
        sentences.loc[n,'Arg']=i[1]
        sentences.loc[n,'Process']=i[4]

        sentences.loc[n,'role']=gold_data[i]
        ind=sentences.loc[n,'sen_id']
        p=sentences.loc[n,'Process']
        sentences.loc[n,'Sentence']=s[p][ind]
        sentences.loc[n,'o1']=sentences.loc[n,'Sentence'].find(sentences.loc[n,'Arg'])
        n=n+1


    #optional:delete spaces near special characters
    for i in list(sentences.index.get_values()):
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace('-LRB- ','(')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace(' -RRB-',')')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace(' ,',',')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace('-LSB- ','[')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace(' -RSB-',']')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace(' \'' , '\'')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace(' ;',';')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace(' .','.')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace(' - ','- ')
        sentences.loc[i,'Arg']=sentences.loc[i,'Arg'].replace(' :',':')


    return sentences




