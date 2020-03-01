#!/usr/bin/env python
# coding: utf-8

# In[13]:


# chạy mining nhớ exclude intent
from intent_recognizer import *
import time
import random
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)


@app.errorhandler(404)
def url_error(e):
    #print("---------------------")
    return msg(404, "cao chánh dương")


@app.errorhandler(500)
def server_error(e):
    return msg(500, "SERVER ERROR")


@app.route('/api/cse-assistant-conversation-manager/classify-user-message', methods=['POST'])
def post_api():
    input_data = request.get_json(force=True)
    #print(input_data)
    if "message" not in input_data.keys():
        return msg(400, "Message cannot be None")
    else:
        message = input_data["message"]
        result, probability = process_message(message)
        
    return jsonify({"code": 200, "message": result, "probability": probability})

if __name__ == '__main__':
    app.run()