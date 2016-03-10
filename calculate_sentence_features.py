import ilp_utils
import word2vec
from ilp_utils import *
from word2vec import *
import pandas as pd
from frame_extraction import *
from operator import itemgetter
import os
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger


os.environ['CLASSPATH']='STANFORDTOOLSDIR/stanford-postagger-full-2015-12-09/stanford-postagger.jar' #set classpath to pos tagger
os.environ['STANFORD_MODELS']='STANFORDTOOLSDIR/stanford-postagger-full-2015-12-09/models'
#file for calculating neighbouring frame ,sentence wise and inserting the frames into the dataset containing argument pairs
#by matching the sentence and process in the dataset
#returns sentence dataframe
def get_frame_feature(sen_t1):
	frame_test=get_frames('NEWTESTSEN_FOLD5.txt.out')#Insert appropriate semafor output file
	print "Getting sen features "
	for i in list(sen_t1.index.get_values()):
		fs=frame_test[i]
		#print fs
		#print sen_t1.loc[i,'Sentence']
		fs.sort(key=itemgetter(2),reverse=False)
		maxi1=0
		mini1=1000



		le=int(sen_t1.loc[i,'o1'])


		for j in range(0,len(fs)):
			e=int(fs[j][2])
			s=int(fs[j][1])
			if(e > maxi1) and ( e < le):


					lf=fs[j][0]
					maxi1=int(fs[j][2])

			else:
				lf=''

			if(s < mini1) and (s > (le +len(sen_t1.loc[i,'Arg']))):


					mini1= int(fs[j][1])

					rf=fs[j][0]
			else:
				rf=''



		sen_t1.loc[i,'lf1']=lf

		sen_t1.loc[i,'rf1']=rf
	return sen_t1


def get_pos_tag(sen):#pass sentence dataframe
    st = StanfordPOSTagger('/home/sadhana/stanford-postagger-full-2015-12-09/models/english-left3words-distsim.tagger',path_to_jar=
                           '/home/sadhana/stanford-postagger-full-2015-12-09/stanford-postagger.jar')#,path_to_models_jar='/home/sadhana/stanford-postagger-full-2015-12-09/models')

    stanford_dir = st._stanford_jar.rpartition('/')[0]
    stanford_jars = find_jars_within_path(stanford_dir)
    st._stanford_jar = ':'.join(stanford_jars)
    for i in list(sen.index.get_values()):
        t=st.tag(sen.loc[i,'Arg'].split())
        tags=[]
        for j in range(0,len(t)):
            tags.append(t[j][1])
        #print i
        sen.set_value(i,'POStag',tags)
    return sen