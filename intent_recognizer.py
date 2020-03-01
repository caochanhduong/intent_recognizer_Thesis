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
  list_token=input_sentence.split(' ')
  return vocab.numericalize(list_token)



def forward_dropout(input_sentence):
  t = torch.tensor([sentence_to_index_vector(input_sentence)])
  lm.reset()
  raw_output, dropout_output = lm[0](t)
  
  dropout_output_last_lst=dropout_output[2].detach().numpy().tolist()
  dropout_output_last_lst=dropout_output_last_lst[0]
#   #print(dropout_output_last_lst)
  max_pooling_lst=[]
  avg_pooling_lst=[]
#   print(len(dropout_output_last_lst))
  for i in range(emb_sz):
    lst_one_emb=[]
    for j in range(len(dropout_output_last_lst)-1):
        lst_one_emb.append(dropout_output_last_lst[j][i])
    if len(dropout_output_last_lst)==1:
        max_pooling_lst.append(dropout_output_last_lst[0][i])
        avg_pooling_lst.append(dropout_output_last_lst[0][i])
    else:    
        max_pooling_lst.append(max(lst_one_emb))
        avg_pooling_lst.append(sum(lst_one_emb) / len(lst_one_emb) )
  return max_pooling_lst+avg_pooling_lst+dropout_output_last_lst[-1]


#message -> intent want information
def extract_and_get_intent(message):
        # 10 intent pattern matching

    for notification in list_address_notification:
        if message.lower().find(notification)!=-1:
            return 'address',1.0

    for notification in list_name_place_notification:
        if message.lower().find(notification)!=-1 and "liên lạc" not in message.lower() and "liên hệ" not in message.lower() and "đăng kí" not in message.lower() and "đăng ký" not in message.lower():
            return 'name_place',1.0


    for notification in list_type_activity_notification:
        if message.lower().find(notification)!=-1:
            return 'type_activity',1.0

    for notification in list_name_activity_notification:
        if message.lower().find(notification)!=-1:
            return 'name_activity',1.0

    

    for notification in list_time_notification:
        if message.lower().find(notification)!=-1 and "đăng ký" not in message.lower() and "đăng kí" not in message.lower() and "có khi nào" not in message.lower():
            return 'time',1.0

    for notification in list_holder_notification:
        if message.lower().find(notification)!=-1 and "liên lạc" not in message.lower() and "liên hệ" not in message.lower() and "đăng kí" not in message.lower() and "đăng ký" not in message.lower() and "sdt" not in message.lower() and "số điện thoại" not in message.lower() and "email" not in message.lower() and "sđt" not in message.lower() and "facebook" not in message.lower() and "fb" not in message.lower():
            return 'holder',1.0

    for notification in list_reward_notification:
        if message.lower().find(notification)!=-1:
            return 'reward',1.0
        
    
    # 5 intent machine learning 
    #remove ? with blank in the last
    message=re.sub('[?]','',message.lower())
    list_token=message.split(' ')
    while '' in list_token:
        list_token.remove('')
    message=' '.join(list_token)

    max_proba=np.amax(clf.predict_proba([forward_dropout(message)])[0])

    #print(clf.predict_proba([forward_dropout(message)])[0])
    #print(max_proba)
    if max_proba>CONST_THRESHOLD:
        return le.inverse_transform(clf.predict([forward_dropout(message.lower())]))[0],max_proba


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
            # #print(signal)
            return True

    for verb in list_verb_have:
        if (message.lower().find(verb)!=-1 and message.lower().find("không")!=-1 and message.lower().find(verb)<message.lower().find("không")):
            #print("1")
            return True

    #....sao (liên hệ/đăng ký sao)
    if "sao"==message.split(' ')[len(message.split(' '))-1]:
        return True


    #.... sao bạn
    for object in list_object:
        if message.lower().find("sao")!=-1 and message.lower().find(object)!=-1 and message.lower().find("sao")<message.lower().find(object):
            return True

    if (message.lower().find("còn")!=-1 and message.lower().find("không")!=-1 and message.lower().find("còn")<message.lower().find("không")):
        #print("1")
        return True
    
    if message.lower().find("xin")!=-1 and (message.lower().find("chào")< message.lower().find("xin")):
        #print("1")
        return True

    #cách liên hệ/đăng ký
    if message.lower().find("cách")==0:
        #print("1")
        return True

    #ai .... (ai được tham gia)
    if message.lower().find("ai")==0:
        #print("1")
        return True


    # #thông tin về xxx/thông tin xxx
    # if message.lower().find("thông tin")!=-1:
    #     #print("1")
    #     return True

    for subject in list_subject:
        for verb in list_verb_want:
            if (message.lower().find(subject+" muốn "+verb)!=-1 or message.lower().find("cho "+subject+" "+verb)!=-1 or message.lower().find(subject+" cần "+verb)!=-1):
                #print("2")
                return True

    for subject in list_subject:
        #chứ bạn
        if message.lower().find("chứ "+subject)!=-1:
            #print("3")
            return True
        if (message.lower().find(subject+" muốn được hỏi")!=-1 or message.lower().find(subject+" muốn được tư vấn")!=-1):
            #print("3")
            return True
        if (message.lower().find(subject+" cần")!=-1 and message.lower().find("thông tin")!=-1 and message.lower().find(subject+" cần")<message.lower().find("thông tin")):
            #print("4")
            return True
        if (message.lower().find(subject+" muốn")!=-1 and message.lower().find("thông tin")!=-1 and message.lower().find(subject+" muốn")<message.lower().find("thông tin")):
            #print("4")
            return True
        if (message.lower().find("gửi "+subject)!=-1):
            #print("5")
            return True
        if (message.lower().find("chỉ "+subject)!=-1):
            #print("5")
            return True
        if (message.lower().find("chỉ giúp "+subject)!=-1):
            #print("5")
            return True

    #cho xin
    for verb in list_verb_want:
        if (message.lower().find("cho "+verb)!=-1):
            #print("6")
            return True

    # nào ... nhỉ
    for signal in list_question_signal_last:
        if (message.lower().find("nào")!=-1) and (message.lower().find(signal)!=-1) and (message.lower().find("nào")<message.lower().find(signal)):
            #print("6")
            return True

    #...... hả
    if message.lower().split(' ')[len(message.lower().split(' '))-1]=="hả":
        return True
    

    
    #gửi cho mình/cho mình/gửi mình/mình cần/mình muốn
    for subject in list_subject:
        if (message.lower().find("cho "+subject)!=-1) or (message.lower().find("gửi "+subject)!=-1) or (message.lower().find(subject+" cần")!=-1) or (message.lower().find(subject+" muốn")!=-1):
            return True


    #mình (có việc) muốn/cần
    #mình định....
    for subject in list_subject:
        if ((message.lower().find(subject)!=-1) and (message.lower().find("định")!=-1) and (message.lower().find(subject)<message.lower().find("định"))) or ((message.lower().find(subject)!=-1) and (message.lower().find("cần")!=-1) and (message.lower().find(subject)<message.lower().find("cần"))) or ((message.lower().find(subject)!=-1) and (message.lower().find("muốn")!=-1) and (message.lower().find(subject)<message.lower().find("muốn"))):
            return True

    #bắt YES-NO/WH question mà signal cuối câu 
    if len(message.split(" "))>3 and (message.split(" ")[-1].lower()=="chưa" or message.split(" ")[-1].lower()=="không" or message.split(" ")[-1].lower()=="ta" or message.split(" ")[-1].lower()=="sao" or message.split(" ")[-1].lower()=="nhỉ" or message.split(" ")[-1].lower()=="nào"):
        #print("7")
        return True

    #bắt YES-NO question cuối câu có chủ ngữ

    for subject in list_object:
        for question_signal_last in list_question_signal_last:
            if message.split(" ")[-1].lower()==subject and message.split(" ")[-2].lower()==question_signal_last:
                #print("8")
                return True 
    

    return False



#message -> final output 
def process_message(message):
    message_preprocessed = re.sub('[\:\_=\+\-\#\@\$\%\$\\(\)\~\@\;\'\|\<\>\]\[\"\–“”…*]',' ',message)
    message_preprocessed=message_preprocessed.replace(',', ' , ')
    message_preprocessed=message_preprocessed.replace('.', ' . ')
    message_preprocessed=message_preprocessed.replace('!', ' ! ')
    message_preprocessed=message_preprocessed.replace('&', ' & ')
    message_preprocessed=message_preprocessed.replace('?', ' ? ')
    message_preprocessed = compound2unicode(message_preprocessed)
    list_token=message_preprocessed.split(' ')
    while '' in list_token:
        list_token.remove('')
    message_preprocessed=' '.join(list_token)

    if check_intent(message_preprocessed):
        return extract_and_get_intent(message_preprocessed)

    for notification in list_bye_notification:   
        if message_preprocessed.lower().find(notification)!=-1:
            return 'bye',1.0

    for notification in list_hello_notification:   
        if message_preprocessed.lower().find(notification)!=-1:
            return 'hello',1.0
        
    for object in list_object:
        if message_preprocessed.lower().find("hi " +object)!=-1 and len(message_preprocessed.split(' '))<=5:
            return 'hello',1.0
    if message_preprocessed.lower()=="hi":
        return "hello",1.0

    return 'not intent',1.0

def compound2unicode(text):
  #https://gist.github.com/redphx/9320735`
  text = text.replace("\u0065\u0309", "\u1EBB")    # ẻ
  text = text.replace("\u0065\u0301", "\u00E9")    # é
  text = text.replace("\u0065\u0300", "\u00E8")    # è
  text = text.replace("\u0065\u0323", "\u1EB9")    # ẹ
  text = text.replace("\u0065\u0303", "\u1EBD")    # ẽ
  text = text.replace("\u00EA\u0309", "\u1EC3")    # ể
  text = text.replace("\u00EA\u0301", "\u1EBF")    # ế
  text = text.replace("\u00EA\u0300", "\u1EC1")    # ề
  text = text.replace("\u00EA\u0323", "\u1EC7")    # ệ
  text = text.replace("\u00EA\u0303", "\u1EC5")    # ễ
  text = text.replace("\u0079\u0309", "\u1EF7")    # ỷ
  text = text.replace("\u0079\u0301", "\u00FD")    # ý
  text = text.replace("\u0079\u0300", "\u1EF3")    # ỳ
  text = text.replace("\u0079\u0323", "\u1EF5")    # ỵ
  text = text.replace("\u0079\u0303", "\u1EF9")    # ỹ
  text = text.replace("\u0075\u0309", "\u1EE7")    # ủ
  text = text.replace("\u0075\u0301", "\u00FA")    # ú
  text = text.replace("\u0075\u0300", "\u00F9")    # ù
  text = text.replace("\u0075\u0323", "\u1EE5")    # ụ
  text = text.replace("\u0075\u0303", "\u0169")    # ũ
  text = text.replace("\u01B0\u0309", "\u1EED")    # ử
  text = text.replace("\u01B0\u0301", "\u1EE9")    # ứ
  text = text.replace("\u01B0\u0300", "\u1EEB")    # ừ
  text = text.replace("\u01B0\u0323", "\u1EF1")    # ự
  text = text.replace("\u01B0\u0303", "\u1EEF")    # ữ
  text = text.replace("\u0069\u0309", "\u1EC9")    # ỉ
  text = text.replace("\u0069\u0301", "\u00ED")    # í
  text = text.replace("\u0069\u0300", "\u00EC")    # ì
  text = text.replace("\u0069\u0323", "\u1ECB")    # ị
  text = text.replace("\u0069\u0303", "\u0129")    # ĩ
  text = text.replace("\u006F\u0309", "\u1ECF")    # ỏ
  text = text.replace("\u006F\u0301", "\u00F3")    # ó
  text = text.replace("\u006F\u0300", "\u00F2")    # ò
  text = text.replace("\u006F\u0323", "\u1ECD")    # ọ
  text = text.replace("\u006F\u0303", "\u00F5")    # õ
  text = text.replace("\u01A1\u0309", "\u1EDF")    # ở
  text = text.replace("\u01A1\u0301", "\u1EDB")    # ớ
  text = text.replace("\u01A1\u0300", "\u1EDD")    # ờ
  text = text.replace("\u01A1\u0323", "\u1EE3")    # ợ
  text = text.replace("\u01A1\u0303", "\u1EE1")    # ỡ
  text = text.replace("\u00F4\u0309", "\u1ED5")    # ổ
  text = text.replace("\u00F4\u0301", "\u1ED1")    # ố
  text = text.replace("\u00F4\u0300", "\u1ED3")    # ồ
  text = text.replace("\u00F4\u0323", "\u1ED9")    # ộ
  text = text.replace("\u00F4\u0303", "\u1ED7")    # ỗ
  text = text.replace("\u0061\u0309", "\u1EA3")    # ả
  text = text.replace("\u0061\u0301", "\u00E1")    # á
  text = text.replace("\u0061\u0300", "\u00E0")    # à
  text = text.replace("\u0061\u0323", "\u1EA1")    # ạ
  text = text.replace("\u0061\u0303", "\u00E3")    # ã
  text = text.replace("\u0103\u0309", "\u1EB3")    # ẳ
  text = text.replace("\u0103\u0301", "\u1EAF")    # ắ
  text = text.replace("\u0103\u0300", "\u1EB1")    # ằ
  text = text.replace("\u0103\u0323", "\u1EB7")    # ặ
  text = text.replace("\u0103\u0303", "\u1EB5")    # ẵ
  text = text.replace("\u00E2\u0309", "\u1EA9")    # ẩ
  text = text.replace("\u00E2\u0301", "\u1EA5")    # ấ
  text = text.replace("\u00E2\u0300", "\u1EA7")    # ầ
  text = text.replace("\u00E2\u0323", "\u1EAD")    # ậ
  text = text.replace("\u00E2\u0303", "\u1EAB")    # ẫ
  text = text.replace("\u0045\u0309", "\u1EBA")    # Ẻ
  text = text.replace("\u0045\u0301", "\u00C9")    # É
  text = text.replace("\u0045\u0300", "\u00C8")    # È
  text = text.replace("\u0045\u0323", "\u1EB8")    # Ẹ
  text = text.replace("\u0045\u0303", "\u1EBC")    # Ẽ
  text = text.replace("\u00CA\u0309", "\u1EC2")    # Ể
  text = text.replace("\u00CA\u0301", "\u1EBE")    # Ế
  text = text.replace("\u00CA\u0300", "\u1EC0")    # Ề
  text = text.replace("\u00CA\u0323", "\u1EC6")    # Ệ
  text = text.replace("\u00CA\u0303", "\u1EC4")    # Ễ
  text = text.replace("\u0059\u0309", "\u1EF6")    # Ỷ
  text = text.replace("\u0059\u0301", "\u00DD")    # Ý
  text = text.replace("\u0059\u0300", "\u1EF2")    # Ỳ
  text = text.replace("\u0059\u0323", "\u1EF4")    # Ỵ
  text = text.replace("\u0059\u0303", "\u1EF8")    # Ỹ
  text = text.replace("\u0055\u0309", "\u1EE6")    # Ủ
  text = text.replace("\u0055\u0301", "\u00DA")    # Ú
  text = text.replace("\u0055\u0300", "\u00D9")    # Ù
  text = text.replace("\u0055\u0323", "\u1EE4")    # Ụ
  text = text.replace("\u0055\u0303", "\u0168")    # Ũ
  text = text.replace("\u01AF\u0309", "\u1EEC")    # Ử
  text = text.replace("\u01AF\u0301", "\u1EE8")    # Ứ
  text = text.replace("\u01AF\u0300", "\u1EEA")    # Ừ
  text = text.replace("\u01AF\u0323", "\u1EF0")    # Ự
  text = text.replace("\u01AF\u0303", "\u1EEE")    # Ữ
  text = text.replace("\u0049\u0309", "\u1EC8")    # Ỉ
  text = text.replace("\u0049\u0301", "\u00CD")    # Í
  text = text.replace("\u0049\u0300", "\u00CC")    # Ì
  text = text.replace("\u0049\u0323", "\u1ECA")    # Ị
  text = text.replace("\u0049\u0303", "\u0128")    # Ĩ
  text = text.replace("\u004F\u0309", "\u1ECE")    # Ỏ
  text = text.replace("\u004F\u0301", "\u00D3")    # Ó
  text = text.replace("\u004F\u0300", "\u00D2")    # Ò
  text = text.replace("\u004F\u0323", "\u1ECC")    # Ọ
  text = text.replace("\u004F\u0303", "\u00D5")    # Õ
  text = text.replace("\u01A0\u0309", "\u1EDE")    # Ở
  text = text.replace("\u01A0\u0301", "\u1EDA")    # Ớ
  text = text.replace("\u01A0\u0300", "\u1EDC")    # Ờ
  text = text.replace("\u01A0\u0323", "\u1EE2")    # Ợ
  text = text.replace("\u01A0\u0303", "\u1EE0")    # Ỡ
  text = text.replace("\u00D4\u0309", "\u1ED4")    # Ổ
  text = text.replace("\u00D4\u0301", "\u1ED0")    # Ố
  text = text.replace("\u00D4\u0300", "\u1ED2")    # Ồ
  text = text.replace("\u00D4\u0323", "\u1ED8")    # Ộ
  text = text.replace("\u00D4\u0303", "\u1ED6")    # Ỗ
  text = text.replace("\u0041\u0309", "\u1EA2")    # Ả
  text = text.replace("\u0041\u0301", "\u00C1")    # Á
  text = text.replace("\u0041\u0300", "\u00C0")    # À
  text = text.replace("\u0041\u0323", "\u1EA0")    # Ạ
  text = text.replace("\u0041\u0303", "\u00C3")    # Ã
  text = text.replace("\u0102\u0309", "\u1EB2")    # Ẳ
  text = text.replace("\u0102\u0301", "\u1EAE")    # Ắ
  text = text.replace("\u0102\u0300", "\u1EB0")    # Ằ
  text = text.replace("\u0102\u0323", "\u1EB6")    # Ặ
  text = text.replace("\u0102\u0303", "\u1EB4")    # Ẵ
  text = text.replace("\u00C2\u0309", "\u1EA8")    # Ẩ
  text = text.replace("\u00C2\u0301", "\u1EA4")    # Ấ
  text = text.replace("\u00C2\u0300", "\u1EA6")    # Ầ
  text = text.replace("\u00C2\u0323", "\u1EAC")    # Ậ
  text = text.replace("\u00C2\u0303", "\u1EAA")    # Ẫ
  return text


#TEST
if __name__ == '__main__':
    #TESTCASE FILE
    testcase_file= open("testcase_intent_recognizer.txt","r",encoding='utf-8')
    output_test_file=open("output_intent_recognizer_1030_dictionary_replace_all_placeholder.txt","w+",encoding='utf-8')
    testcases=testcase_file.readlines()
    num_success_testcases=0
    num_testcases=len(testcases)
    i=0
    for testcase in testcases:
        # #print(testcase)
        i+=1
        print(i)
        success=False
        testcase_without_enter=testcase.replace('\n', '').split('|')
        message=testcase_without_enter[0]
        intent_label=testcase_without_enter[1]
        intent_predict,proba=process_message(message)
        if intent_label==intent_predict:
            success=True
            num_success_testcases+=1
        output_test_file.write("Message: {0} \t Intent_label: {1} \t Intent_predict: {2} \t Probability: {3} \t Success: {4} \n".format(message,intent_label,intent_predict,proba,success))
    output_test_file.write("Success rate: {0} %".format(100*float(num_success_testcases)/num_testcases))
    output_test_file.close()
    testcase_file.close()
    # print(process_message("tham gia xuân tình nguyện gồm những thành phần nào ?"))
    # print(vocab.stoi["sẻ"])