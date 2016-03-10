import pandas as pd

#Construct training data
from extract_sentences import *
from calculate_sentence_features import *
from get_main_df import *
from calculate_features import *
from classifier import *
#insert json file
gold_sentence_df=extract_goldsen('test.srlout_new5.json')
gold_sentence_df=get_frame_feature(gold_sentence_df)
gold_sentence_df=get_pos_tag(gold_sentence_df)

gold_df=get_arg_pairs(gold_sentence_df)

gold_df=merge_sen_df(gold_sentence_df,gold_df)
gold_df=get_arg_cosine_simialrity(gold_df)
gold_df=get_lf_cosine_similarity(gold_df)
gold_df=get_rf_cosine_similarity(gold_df)

gold_df=get_entailment_score(gold_df)
gold_df=get_pos_similarity(gold_df)






#Construct test dataframe

test_sentence_df=extract_goldsen('test.srlpredict_new5.json')
test_sentence_df=get_frame_feature(test_sentence_df)
test_sentence_df=get_pos_tag(test_sentence_df)

test_df=get_arg_pairs(test_sentence_df)

test_df=merge_sen_df(test_sentence_df,test_df)
test_df=get_arg_cosine_simialrity(test_df)
test_df=get_lf_cosine_similarity(test_df)
test_df=get_rf_cosine_similarity(test_df)

test_df=get_entailment_score(test_df)
test_df=get_pos_similarity(test_df)

#classification,precision,f1 score is printed and final dataframe with results is  returned

result_df=classifier(test_df,gold_df,train_process_filename)

