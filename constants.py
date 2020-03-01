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
CONST_THRESHOLD=0.4
list_label=['contact','register','activity','work','joiner']
le = preprocessing.LabelEncoder()
y = le.fit_transform(list_label)
vocab = torch.load('new_vocab.h5')
lm = get_language_model(AWD_LSTM, 27498)
lm.eval()
lm.load_state_dict(torch.load("model_cpu_add_new_vocab.pth"))
clf = joblib.load('lm_kernel_linear_svm_classifier_latest_1030_with_dict_all_random.pkl') 
emb_sz=lm[0].emb_sz


#INTENT PATTERN MATCHING SIGNAL
list_name_place_notification=["nơi","tại đâu","tại chỗ nào","ở đâu","chỗ nào","tại đâu","khu nào","địa điểm nào","chỗ diễn ra","chỗ đâu","khu tổ chức","là ở","là tại","là nơi","là tập trung tại","là diễn ra tại","là diễn ra ở"]
list_address_notification=["ấp nào","phường nào","xã nào","quận nào","huyện nào","thành phố nào","tỉnh nào","đường nào","đường gì","số mấy","địa chỉ nào","địa chỉ","tên đường","số nhà","phường mấy","quận mấy","số mấy"]
list_type_activity_notification=["loại hoạt động","loại nào","loại gì","kiểu hoạt động","kiểu gì","kiểu nào"]
list_name_activity_notification=["tên gì","tên là gì","tên hoạt động là gì","tên hoạt động"] #check lai
list_time_notification=["là tháng","là ngày","là vào","tháng mấy","thứ mấy","là diễn ra vào","ngày mấy","khi nào","lúc nào","thời gian nào","ngày nào","ngày bao nhiêu","giờ nào","giờ bao nhiêu","mấy giờ","mấy h ","thời gian"]
list_holder_notification=["ban tổ chức","btc","ai tổ chức","đơn vị nào tổ chức","đơn vị tổ chức","trường nào tổ chức","clb nào tổ chức","câu lạc bộ nào tổ chức","người tổ chức","tổ chức hả","tổ chức phải không","tổ chức đúng không"]
list_reward_notification=["có được drl",'có được đrl','có được điểm rèn luyện',"được thứ gì","bao nhiêu tiền","thưởng cái gì","được lợi gì","mấy ngày ctxh","mấy điểm rèn luyện","mấy drl","mấy đrl","mấy ngày công tác xã hội","bao nhiêu ngày ctxh","bao nhiêu ctxh","bao nhiêu điểm rèn luyện","bao nhiêu drl","bao nhiêu đrl","bao nhiêu ngày công tác xã hội","có ích gì","điểm rèn luyện","được công tác xã hội","được ctxh","được thưởng gì","được gì","được cái gì","có lợi gì","lợi ích","phần thưởng","được quà gì","tặng gì","được bao nhiêu","số tiền","có được ctxh","có được ngày công tác xã hội","có được ngày ctxh","có được tặng","có được thưởng","có được cho"]



#INTENT MESSAGE SIGNALS
list_question_signal=[" hả ","chứ","có biết","phải không","đâu","là sao","nào","khi nào","nơi nào","không ạ","k ạ","là sao","nữa vậy","chưa á","ko ạ","sao ạ","chưa ạ","sao vậy","không vậy","k vậy","ko vậy","chưa vậy","thế"," nhỉ "," ai"," ai ","ở đâu","ở mô","đi đâu","bao giờ","bao lâu","khi nào","lúc nào","hồi nào","vì sao","tại sao","thì sao","làm sao","như nào","thế nào","cái chi","gì","bao nhiêu","mấy","?"," hả ","được không","được k","được ko","vậy ạ","nào vậy","nào thế","nữa không","đúng không","đúng k","đúng ko","nữa k","nữa ko","nào ấy","nào ạ"]
list_question_signal_last=["vậy","chưa","không","sao","à","hả","nhỉ","thế"]
list_object=["bạn","cậu","ad","anh","chị","admin","em","mày","bot"]
list_subject=["mình","tôi","tớ","tao","tui","anh","em"]
list_verb_want=["hỏi","biết","xin"]
list_verb_have=["có","được"]


#intent not want information
list_hello_notification=[" hi ","hello","chào","helo"]
list_bye_notification=["bye","tạm biệt","bai","gặp lại"] 

