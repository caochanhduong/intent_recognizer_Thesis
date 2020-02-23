#!/usr/bin/env python
# coding: utf-8

from sklearn.externals import joblib
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from fastai.text import *
import torch
import pickle
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import preprocessing
from constants import *


def sentence_to_index_vector(input_sentence):
  input_sentence = re.sub('[\,\_=\+\-\#\@\$\%\$\\.\?\:\(\)\~\!\@\;\'\|\<\>\]\[\"\–“”…*]',' ',input_sentence).lower()
  list_token=input_sentence.split(' ')
  while '' in list_token:
    list_token.remove('')
  return vocab.numericalize(list_token)



def forward_dropout(input_sentence):
  t = torch.tensor([sentence_to_index_vector(input_sentence)])
  lm.reset()
  raw_output, dropout_output = lm[0](t)
  dropout_output_last_lst=dropout_output[2].detach().numpy().tolist()
  dropout_output_last_lst=dropout_output_last_lst[0]
  max_pooling_lst=[]
  avg_pooling_lst=[]
  for i in range(emb_sz):
    lst_one_emb=[]
    for j in range(len(dropout_output_last_lst)-1):
      lst_one_emb.append(dropout_output_last_lst[j][i])
    max_pooling_lst.append(max(lst_one_emb))
    avg_pooling_lst.append(sum(lst_one_emb) / len(lst_one_emb) )
  return max_pooling_lst+avg_pooling_lst+dropout_output_last_lst[-1]


#message -> intent want information
def extract_and_get_intent(message):
    # 5 intent machine learning 

    max_proba=np.amax(clf.predict_proba([forward_dropout(message.lower())])[0])

    # print(clf.predict_proba([forward_dropout(message)])[0])
    # print(max_proba)
    if max_proba>CONST_THRESHOLD:
        return le.inverse_transform(clf.predict([forward_dropout(message.lower())]))[0],max_proba

    # 10 intent pattern matching
    for notification in list_name_place_notification:
        if message.lower().find(notification)!=-1:
            return 'name_place',1.0


    for notification in list_type_activity_notification:
        if message.lower().find(notification)!=-1:
            return 'type_activity',1.0

    for notification in list_name_activity_notification:
        if message.lower().find(notification)!=-1:
            return 'name_activity',1.0

    for notification in list_address_notification:
        if message.lower().find(notification)!=-1:
            return 'address',1.0

    for notification in list_time_notification:
        if message.lower().find(notification)!=-1:
            return 'time',1.0

    for notification in list_holder_notification:
        if message.lower().find(notification)!=-1:
            return 'holder',1.0

    for notification in list_reward_notification:
        if message.lower().find(notification)!=-1:
            return 'reward',1.0

    return 'other',1.0


#message -> check intent want information 
def check_intent(message):
#     ... subject muốn hỏi/biết/xin… *
# ... subject muốn được hỏi/tư vấn .... *
# ... cho subject hỏi/biết/xin *
# ... subject cần hỏi/biết/xin... *
# .... subject cần ... thông tin … *
# .... gửi subject ... *
# ... cho hỏi/xin/biết… *
# ... có/được (...)? không (...) *
    #bắt WH question 
    for signal in list_question_signal:
        if signal in message.lower():
            # print(signal)
            return True

    for verb in list_verb_have:
        if (message.lower().find(verb)!=-1 and message.lower().find("không")!=-1 and message.lower().find(verb)<message.lower().find("không")):
            print("1")
            return True

    for subject in list_subject:
        for verb in list_verb_want:
            if (message.lower().find(subject+" muốn "+verb)!=-1 or message.lower().find("cho "+subject+" "+verb)!=-1 or message.lower().find(subject+" cần "+verb)!=-1):
                print("2")
                return True

    for subject in list_subject:
        if (message.lower().find(subject+" muốn được hỏi")!=-1 or message.lower().find(subject+" muốn được tư vấn")!=-1):
            print("3")
            return True
        if (message.lower().find(subject+" cần")!=-1 and message.lower().find("thông tin")!=-1 and message.lower().find(subject+" cần")<message.lower().find("thông tin")):
            print("4")
            return True
        if (message.lower().find(subject+" muốn")!=-1 and message.lower().find("thông tin")!=-1 and message.lower().find(subject+" muốn")<message.lower().find("thông tin")):
            print("4")
            return True
        if (message.lower().find("gửi "+subject)!=-1):
            print("5")
            return True

    for verb in list_verb_want:
        if (message.lower().find("cho "+verb)!=-1):
            print("6")
            return True

    #bắt YES-NO/WH question mà signal cuối câu 
    if len(message.split(" "))>3 and (message.split(" ")[-1].lower()=="chưa" or message.split(" ")[-1].lower()=="không" or message.split(" ")[-1].lower()=="ta" or message.split(" ")[-1].lower()=="sao" or message.split(" ")[-1].lower()=="nhỉ" or message.split(" ")[-1].lower()=="nào"):
        print("7")
        return True

    #bắt YES-NO question cuối câu có chủ ngữ

    for subject in list_object:
        for question_signal_last in list_question_signal_last:
            if message.split(" ")[-1].lower()==subject and message.split(" ")[-2].lower()==question_signal_last:
                print("8")
                return True 
    

    return False



#message -> final output 
def process_message(message):
    message_preprocessed = re.sub('[\,\_=\+\-\#\@\$\%\$\\.\?\:\(\)\~\!\@\;\'\|\<\>\]\[\"\–“”…*]',' ',message)
    list_token=input_sentence.split(' ')
    while '' in list_token:
        list_token.remove('')
    if check_intent(message_preprocessed):
        return extract_and_get_intent(message_preprocessed)

    for notification in list_bye_notification:   
        if message_preprocessed.lower().find(notification)!=-1:
            return 'bye',1.0

    for notification in list_hello_notification:   
        if message_preprocessed.lower().find(notification)!=-1:
            return 'hello',1.0

    return 'not intent',1.0



#TEST
if __name__ == '__main__':
    testcases=testcase_file.readlines()
    num_success_testcases=0
    num_testcases=len(testcases)
    for testcase in testcases:
        success=False
        testcase_without_enter=testcase.replace('\n', '').split('|')
        message=testcase_without_enter[0]
        intent_label=testcase_without_enter[1]
        intent_predict,proba=extract_and_get_intent(message)
        if intent_label==intent_predict:
            success=True
            num_success_testcases+=1
        output_test_file.write("Message: {0} \t Intent_label: {1} \t Intent_predict: {2} \t Probability: {3} \t Success: {4} \n".format(message,intent_label,intent_predict,proba,success))
    output_test_file.write("Success rate: {0} %".format(100*float(num_success_testcases)/num_testcases))
    output_test_file.close()
    testcase_file.close()