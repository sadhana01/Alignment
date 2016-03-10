
import pandas as pd
import itertools
import numpy as np
def get_arg_pairs(sentences):
    df=pd.DataFrame(columns=['Process','Arg1','Arg2','Sentence1','Sentence2','Role1','Role2','True_label'])
    m=0
    processes=list(set(sentences['Process']))
    for p in processes:
        for ac in itertools.combinations(sentences[sentences['Process']==p].index.tolist(),2):


                        df.loc[m,'Process']=p
                        df.loc[m,'Arg1']=sentences.loc[ac[0],'Arg']
                        df.loc[m,'Arg2']=sentences.loc[ac[1],'Arg']
                        df.loc[m,'Sentence1']=sentences.loc[ac[0],'Sentence']
                        df.loc[m,'Sentence2']=sentences.loc[ac[1],'Sentence']
                        df.loc[m,'Role1']=sentences.loc[ac[0],'Role']
                        df.loc[m,'Role2']=sentences.loc[ac[1],'Role']
                        if df.loc[m,'Role1']==df.loc[m,'Role2']:
                            df.loc[m,'True_label']=1
                        else:
                            df.loc[m,'True_label']=0

                        m=m+1

    return df
def merge_sen_df(sentences,df):
    for i in list(df.index.get_values()):
        print i
        arg=df.loc[i,'Arg1']
        y=sentences[(sentences['Arg']==arg) & (sentences['Process']==df.loc[i,'Process']) ]
        z=sentences[(sentences['Arg']==df.loc[i,'Arg2']) & (sentences['Process']==df.loc[i,'Process']) ]
        df.loc[i,'Sentence1']=y['Sentence'].values[0]
        df.loc[i,'Sentence2']=z['Sentence'].values[0]
        df.loc[i,'lf1']=y['lf1'].values[0]
        df.loc[i,'rf1']=y['rf1'].values[0]
        df.loc[i,'lf2']=z['lf1'].values[0]
        df.loc[i,'rf2']=z['rf1'].values[0]
        df.loc[i,'o1']=y['o1'].values[0]
        df.loc[i,'o2']=z['o1'].values[0]
        df.set_value(i,'POStag1',y['POStag'].values[0])

        df.set_value(i,'POStag2',z['POStag'].values[0])
    return df