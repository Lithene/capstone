"""Helper file for data preprocessing"""

############################### LIBRARIES #####################################
import numpy as np
import pandas as pd
import datetime
import time

# Preprocessing libraries
import re
# from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords
import unicodedata
import emoji

# Plotting libraries
import matplotlib.pyplot as plt

############################### PRE-PROCESSING #####################################
#### REMOVAL FUNCTIONS

def remove_mentions(text):
  
  # remove tags
  text = re.sub(r"@\S+", "", text)

  return text

def remove_websites(text):
  
  # remove https
  URL_REGEX = r"""((?:(?:https|ftp|http)?:(?:\/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|org|uk)\/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|sg)\b\/?(?!@)))"""
  
  text = re.sub(URL_REGEX, "", text)

  return text

def remove_hashtags(text):
  
  # remove hashtags
  text = re.sub(r"#\S+", "", text)

  return text

def remove_apos(text):
  
  # remove apostrophes eg. abc's
  text = re.sub("\'\w+", "", text)

  return text

def remove_punc(text):
  
  # remove punctuation
  text = re.sub('[%s]' % re.escape(string.punctuation), "", text)
  text = re.sub("•", "", text)

  return text

def remove_nums(text):
  
  # remove numbers
  text = re.sub(r'\w*\d+\w*', "", text)

  return text

def remove_nextline(text):

  # remove next line
  text = text.replace('\n', " ").strip()

  return text

def remove_morespace(text):

  # remove spaces more than " "
  
  text = re.sub('\s{2,}', " ", text)

  return text


def remove_chinese(list_strings):
    """
    cleaned: data_df['cleaned_text']
    """
    
    no_chinese = []
    CHINESE_REGEX = r'[\u4e00-\u9fff]+' #r'[^\x00-\x7F]+'

    for term in list_strings:
        #new_post = []
        #for term in post:
        term.replace(CHINESE_REGEX, '')
        #if re.findall(CHINESE_REGEX, term) == []: # find chinese chars
            #new_post.append(term)
        
        no_chinese.append(term)
        
    return no_chinese

def remove_emojis(text):
  """
  INPUT
  text: string

  OUTPUT
  string
  """
  EMOJI_REGEX = re.compile("["
      u"\U0001F600-\U0001F64F"  # emoticons
      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
      u"\U0001F680-\U0001F6FF"  # transport & map symbols
      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
      u"\U00002500-\U00002BEF"  # chinese char
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      u"\U0001f926-\U0001f937"
      u"\U00010000-\U0010ffff"
      u"\u2640-\u2642" 
      u"\u2600-\u2B55"
      u"\u200d"
      u"\u23cf"
      u"\u23e9"
      u"\u231a"
      u"\ufe0f"  # dingbats
      u"\u3030"
                    "]+", re.UNICODE)
  
  no_emoji = []
  
  for term in text:
    if re.findall(EMOJI_REGEX, term) == []: # find chinese chars
      no_emoji.append(term)
      
  return "".join(no_emoji)


def text_cleaning(text):
    # remove tags
    text = re.sub("@\S+", "", str(text))
    # remove websites
    text = re.sub("https*\S+", "", str(text))
    # remove hashtags
    text = re.sub("#\S+", "", str(text))
    # remove apostrophes eg. abc's
    text = re.sub("\'\w+", "", str(text))
    # remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), "", str(text))
    # remove numbers
    text = re.sub(r'\w*\d+\w*', "", str(text))
    # lowercase -- to remove stopwords
    text = text.lower()
    # remove stopwords
    default_stopwords = stopwords.words('english')
    text = " ".join(word for word in text.split() if word not in default_stopwords)
    
    # remove spaces more than " "
    #text = re.sub('\s{2,}', " ", text)

    return text #[word for word in text.split()]


############################# FEATURE ENGINEERING ##############################
#### EXTRACTION FUNCTIONS
def extract_emails(text_string):
  """
  INPUT
  text_string: a string containing approval codes
  
  OUTPUT
  list of approval codes
  """

  EMAIL_REGEX = r'[\w.+-]+@[\w-]+\.[\w.-]+'
  emails = re.findall(EMAIL_REGEX, text_string)
      
  return emails

def extract_nonpru(text_string):    

  NON_PRU = r"""[a-zA-Z0-9_.+-]+@(?!(pruadviser)).*\.[a-zA-Z]{2,6}"""
  emails = re.findall(NON_PRU, text_string)
      
  return emails

def extract_hyperlinks(text_string):
  
  all_links = []

  URL_REGEX = r"""((?:(?:https|ftp|http)?:(?:\/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|org|uk)\/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|sg)\b\/?(?!@)))"""

  #extract hyperlinks
  links = re.findall(URL_REGEX, text_string)

  return links

def extract_mentions(text_string):
  
  # Function to add a space behind mentions (@)
  result = re.findall("(^|[^@\w])@(\w{1,15})", text_string) # disregards emails
  # Add to a list
  result_list = ['@' + tuple(j for j in i if j)[-1] for i in result]
  
  return result_list

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

def extract_emojis(text_string):
  """
  INPUT
  text_string: string containing emoji

  OUTPUT
  string of emojis
  """
  
  return [c for c in text_string if c in emoji.EMOJI_DATA]

def is_emoji(string): 
  """Returns True if the string is an emoji.""" 
  return string in unicode_codes.EMOJI_DATA 

def is_emoji(string): 
    """Returns True if the string is an emoji.""" 
    return string in unicode_codes.EMOJI_DATA 


#### CONVERSION FUNCTIONS

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

def lowercase(text_string):
  """
  INPUT
  text_string: string

  OUTPUT
  lowercased string
  """
  return text_string.lower()

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

#### CHECKING FUNCTIONS

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

def set_binary_label(data, class_col):

    data['incompliant'] = np.where(data[class_col] == 'No further action required', 0, 1)

    return data

def check_disclaimer(list_hyperlinks):
  if "https://www.prudential.com.sg/fc-disclaimer" in list_hyperlinks:
    return "Y"
  else:
    return "N"

#### CONSOLIDATED
def create_features(data_df, text_col, output_features):

    data_df = data_df.drop_duplicates(subset=[text_col])

    print("data_df", data_df.shape)
    ### Features addition ###
    # Fill Nulls in content column
    data_df[text_col] = data_df[text_col].fillna('')
    # Apply spaces behind the hastags to identify hashes
    data_df[text_col] = data_df[text_col].apply(lambda x: add_space_hashes(x))
    # Extract all hashtags
    data_df['hashtags'] = data_df[text_col].apply(lambda x: extract_hashtags(x))
    # Extract all mentions
    data_df['mentions'] = data_df[text_col].apply(lambda x: extract_mentions(x))
    # Extract all emojis
    data_df['emojis'] = data_df[text_col].apply(lambda x: extract_emojis(x))
    # Translate Emojis to text
    data_df['emojis_text'] = data_df['emojis'].apply(lambda x: translate_emojis(x))
    # Extract Approval codes
    data_df['approvals'] = data_df[text_col].apply(lambda x: extract_codes(x))
    # Extract all emails
    data_df['nonpru_emails'] = data_df[text_col].apply(lambda x: extract_nonpru(x))
    # Extract hyperlinks
    data_df['hyperlinks'] = data_df[text_col].apply(lambda x: extract_hyperlinks(x))
    
    # Check if there are words to be flagged - breach class
    data_df['breach_flagwords'] = data_df['cleaned_text'].apply(lambda x: contains_flagged_words(x, text_breach))
    # Check if there are words to be flagged in the hashes - breach class
    data_df['breach_hashes'] = data_df['hashtags'].apply(lambda x: contains_flagged_hashes(x, hastag_breach))
    # Have non-prudential email indicator
    data_df['has_nonpru_email'] = np.where(data_df["nonpru_emails"].str.len() == 0, "N", "Y")
    # Have hyperlinks
    data_df['has_hyperlinks']  = np.where(data_df["hyperlinks"].str.len() == 0, "N", "Y")
    # Have approvals
    data_df['has_approvals']  = np.where(data_df["approvals"].str.len() == 0, "N", "Y")
    # Has disclaimer
    data_df['has_disclaimer'] = data_df['hyperlinks'].apply(lambda x: check_disclaimer(x))
    
    data_df = data_df[output_features]
    return data_df

def clean_text(data_df, target_var):
    
    # Fill Nulls in content column
    data_df[target_var] = data_df[target_var].fillna('')
    # Text Cleaning
    data_df['cleaned_text'] = data_df[target_var]
    data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_mentions(x))
    data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_websites(x))
    data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_hashtags(x))
    #data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_apos(x))
    data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_punc(x))
    data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_nums(x))
    data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_nextline(x))
    data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_emojis(x))
    data_df['cleaned_text'] = data_df.cleaned_text.apply(lambda x: remove_stopwords(x))

    return data_df['cleaned_text']

######################### NAME ENTITY RECOGNITION ##############################

def get_ner(text, nlp_model):

    ner_list = {}
    # Loading the Spacy Pipeline
    doc = nlp_model(text)

    for ent in doc.ents:
        ner_list[ent.text] = ent.label_

    return ner_list

def check_monetary(ner_dictionary):
    if any([k for k, v in ner_dictionary.items() if v == 'MONEY']):
      return "Y"
    else:
      return "N"

def get_ner_features(data_df):

    # Loading the Spacy Pipeline
    spacy_path = '/dbfs/mnt/datahub-apps/ai_critic/libraries/en_core_web_sm-2.3.1'
    nlp_model = spacy.load(spacy_path)
    
    # NER modelling
    data_df['ner'] = data_df['cleaned_text'].apply(lambda x: get_ner(x, nlp_model))
    # Feature - Check context for monetary compensations
    data_df['contains_monetary'] = data_df['ner'].apply(lambda x: check_monetary(x))

    return data_df


######################### ENCODING ##############################

def get_onehot(data, feature_list:list, save_dir:str = None, encoder = None):

    """
    This function takes a dataset converts the defined feature list into one-hot encoded features

    Args:
        data (pandas.DataFrame): 
            the full dataset with target column
        feature_list (list): 
            the feature list to be target encoded
        save_dir (str, optional): 
            the directory to save the encoder
        encoder(category_encoders.OneHotEncoder, optional): 
            load the previously saved encoder

    Returns:
        full_with_enc (pandas.DataFrame):
            full dataframe with encoded + non-encoded features + target column
        ohe_features:
            one-hot encoded feature list

    Example:
        To get the one-hot-encoded marital status features along with the whole dataset
        get_onehot(data, ['marryd'])

    """

    # Seperate data to be encoded
    data_encoding = data[feature_list].copy(deep=True)
    data_same = data[data.columns[~data.columns.isin(feature_list)]]

    if encoder == None: # Use encoder if specified
        encoder = ce.OneHotEncoder(cols=feature_list, use_cat_names=True)
        encoder.fit(data_encoding)

        if save_dir !=None: # Save encoder as a Artifact
            with open(save_dir + 'ohe_encoder.pkl', 'wb') as outfile:
                pickle.dump(encoder, outfile)

    data_enc = encoder.transform(data_encoding)
    data_enc = data_enc.astype('float32')

    full_with_enc = pd.merge(data_same, data_enc, left_index=True, right_index=True)

    return full_with_enc


def get_target_encode(data, feature_list:list, target:str = None, save_dir:str = None, encoder = None):

    """
    This function takes a dataset converts the defined feature list into target encoded features

    Args:
        data (pandas.DataFrame): 
            the full dataset
        feature_list (list): 
            the feature list to be target encoded
        save_dir (str, optional): 
            the directory to save the encoder
        encoder(category_encoders.TargetEncoder, optional): 
            load the previously saved encoder

    Returns:
        full_with_enc (pandas.DataFrame):
            full dataframe with encoded + non-encoded features

    """
    # Set Index
    data = data.set_index('Serial No.')
    # Seperate data to be encoded
    data_same = data[data.columns[~data.columns.isin(feature_list + [target])]]
    data_encoding = data[feature_list + [target]].copy(deep=True)
    #for feature in feature_list:
    #  data_encoding[feature] = [','.join(map(str, l)) for l in data_encoding[feature]]

    if encoder == None: # Use encoder if specified
        encoder = ce.TargetEncoder(cols=feature_list).fit(data_encoding, data_encoding[target])

        # Save encoder as a Artifact
        if save_dir != None:
            with open(save_dir + 'target_encoder', 'wb') as outfile:
                pickle.dump(encoder, outfile)

    data_enc = encoder.transform(data_encoding)
    full_with_enc = pd.merge(data_same, data_enc, left_index=True, right_index=True)

    return full_with_enc



############################### END OF SCRIPT #####################################