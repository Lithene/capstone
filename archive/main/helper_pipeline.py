### IMPORT LIBRARIES

import re
import pandas as pd
import numpy as np
import datetime
import time

import unicodedata
import emoji

import matplotlib.pyplot as plt

#mlflow
import mlflow
from mlflow.tracking import MlflowClient

### MLFLOW
def setup_mlflow():
    EXPERIMENT_NAME = "ai_critic"
    ARTIFACT_REPO = './aicritic_mlflow'
    client = MlflowClient() # Initialize client
    mlflow.set_tracking_uri('http://127.0.0.1:5000/')
    
    # Get the experiment id if it already exists and if not create it
    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME, artifact_location=ARTIFACT_REPO)
    except Exception as err:
        print(err)
        experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    return experiment_id


### FEATURE ENGINEERING
def add_space_hashes(text_string):
    # Function to add a space behind the hash tags
    text_list = []

    # Convert any datetime text into string
    if text_string is not np.nan:
        if isinstance(text_string, datetime.time) | isinstance(text_string, datetime.datetime) | isinstance(text_string, datetime.date):
            text_string = text_string.isoformat()
    
        for char in text_string:
            if char == '#':
                char = ' ' + char
            text_list.append(char)
        
    return ''.join(text_list)

def extract_hashtags(text_string):
 
    # initializing hashtag_list variable
    hashtag_list = []
 
    # splitting the text into words
    for word in text_string.split():
        # checking the first character of every word
        if word[0] == '#':
            # adding the word to the hashtag_list
            hashtag_list.append(word[1:])
        
    return hashtag_list

def extract_mentions(text_string):
    # Function to add a space behind mentions (@)
    result = re.findall("(^|[^@\w])@(\w{1,15})", text_string) # disregards emails
    # Add to a list
    result_list = ['@' + tuple(j for j in i if j)[-1] for i in result]
    
    return result_list

def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.EMOJI_DATA)

def is_emoji(string): 
    """Returns True if the string is an emoji.""" 
    return string in unicode_codes.EMOJI_DATA 

def translate_emojis(text_string):
    result = []

    for char in text_string:
        if unicodedata.category(char) in ('So', 'Mn'):
            result.append(unicodedata.name(char))
        elif unicodedata.category(char) in ('Cs'):
            result.append('?') #char)
        else:
            result.append(char)

    return ','.join(result)

def contains_flagged_words(text_string):
    # Check if text contains flagged words
    flagged = ['financial advisor', 'advisor', 'financial adviser']
    for flag in flagged:
        if flag in text_string:
            return True
    return False

def contains_flagged_hashes(hash_list):
    # Check if hashtags contains flagged words
    flagged = ['financialadvisor', 'advisor', 'financialadviser']
    for flag in flagged:
        for hashtags in hash_list:
            if flag in hashtags:
                return True
    return False

