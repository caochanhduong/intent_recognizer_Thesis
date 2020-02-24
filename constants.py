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

#MODEL LANGUAGE, CLASSIFIER
CONST_THRESHOLD=0.75
list_label=['contact','register','activity','work','joiner']
le = preprocessing.LabelEncoder()
y = le.fit_transform(list_label)
vocab = torch.load('new_vocab.h5')
lm = get_language_model(AWD_LSTM, 27498)
lm.eval()
lm.load_state_dict(torch.load("model_cpu_add_new_vocab.pth"))
clf = joblib.load('lm_kernel_linear_svm_classifier.pkl') 
emb_sz=lm[0].emb_sz


#INTENT PATTERN MATCHING SIGNAL
list_name_place_notification=["ở đâu","chỗ nào","ở nơi nào","khu nào","địa điểm nào"]
list_address_notification=["ấp nào","phường nào","xã nào","quận nào","huyện nào","thành phố nào","tỉnh nào","đường nào","đường gì","số mấy","địa chỉ nào","địa chỉ","tên đường","số nhà"]
list_type_activity_notification=["loại hoạt động nào","loại hoạt động gì","loại nào","loại gì"]
list_name_activity_notification=["tên gì","tên là gì","tên hoạt động là gì","tên hoạt động"] #check lai
list_time_notification=["khi nào","lúc nào","thời gian nào","ngày nào","ngày bao nhiêu","giờ nào","giờ bao nhiêu","mấy giờ","mấy h ","thời gian"]
list_holder_notification=["ai tổ chức","đơn vị nào tổ chức","đơn vị tổ chức","trường nào tổ chức","clb nào tổ chức","câu lạc bộ nào tổ chức","người tổ chức"]
list_reward_notification=["mấy ngày ctxh","mấy điểm rèn luyện","mấy drl","mấy đrl","mấy ngày công tác xã hội","bao nhiêu ngày ctxh","bao nhiêu ctxh","bao nhiêu điểm rèn luyện","bao nhiêu drl","bao nhiêu đrl","bao nhiêu ngày công tác xã hội","điểm rèn luyện","được công tác xã hội","được ctxh","được thưởng gì","được cái gì","có lợi gì","lợi ích"]



#INTENT MESSAGE SIGNALS
list_question_signal=["sao vậy","không vậy","chưa vậy","thế"," nhỉ "," ai"," ai ","ở đâu","ở mô","đi đâu","bao giờ","bao lâu","khi nào","lúc nào","hồi nào","vì sao","tại sao","thì sao","làm sao","như nào","thế nào","cái chi","gì","bao nhiêu","mấy","?"," hả ","được không","vậy ạ"]
list_question_signal_last=["vậy","chưa","không","sao","à","hả","nhỉ"]
list_object=["bạn","cậu","ad","anh","chị","admin","em","mày"]
list_subject=["mình","tôi","tớ","tao","tui"]
list_verb_want=["hỏi","biết","xin"]
list_verb_have=["có","được"]


#intent not want information
list_hello_notification=["hi","hello","chào","helo"]
list_bye_notification=["bye","tạm biệt","bai"] 



#TESTCASE FILE
testcase_file= open("testcase_intent_recognizer.txt","r")
output_test_file=open("output_intent_recognizer.txt","w+")