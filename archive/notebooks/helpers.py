"""Helper file for all scripts"""

############################### LIBRARIES #####################################
import numpy as np
import pandas as pd

# Preprocessing libraries
import re
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords

# Modelling libraries
# Importing Gensim
import gensim
from gensim import corpora
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import linear_model

from sklearn.metrics import f1_score


############################### PRE-PROCESSING #####################################

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

def remove_emojis(list_strings):
    emoj = re.compile("["
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
    
    for term in list_strings:
        #new_post = []
        #for term in post:
        term = re.sub(emoj, '', term)
        #new_post.append(term)
            
        no_emoji.append(term)
        
    return no_emoji


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



############################### TOPIC MODEL LR #####################################

def train_topicmodel_features(data_df, train_corpus, doc_clean, NUM_TOPICS, NUM_WORDS = 10):

    # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    
    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    #NUM_TOPICS = 10
    ldamodel = Lda(doc_term_matrix, num_topics=NUM_TOPICS, id2word = dictionary, passes=50)
    
    for topic in range(len(ldamodel.print_topics(num_words=NUM_WORDS))):
        print(ldamodel.print_topics(num_words=NUM_WORDS)[topic], "\n")
        
    # Get feature vector
    train_vecs = []
    for i in range(len(data_df)):
        top_topics = ldamodel.get_document_topics(train_corpus[i], minimum_probability=0.0)
        topic_vec = [top_topics[i][1] for i in range(NUM_TOPICS)]
        topic_vec.extend([len(data_df.iloc[i].cleaned_text)]) # length review
        train_vecs.append(topic_vec)
    
    X = np.array(train_vecs)
    y = np.array(data_df.non_compliant)

    kf = KFold(5, shuffle=True, random_state=42)
    cv_lr_f1, cv_lrsgd_f1, cv_svcsgd_f1,  = [], [], []

    for train_ind, val_ind in kf.split(X, y):
        # Assign CV IDX
        X_train, y_train = X[train_ind], y[train_ind]
        X_val, y_val = X[val_ind], y[val_ind]

        # Scale Data
        scaler = StandardScaler()
        X_train_scale = scaler.fit_transform(X_train)
        X_val_scale = scaler.transform(X_val)

        # Logisitic Regression
        lr = LogisticRegression(
            class_weight= 'balanced',
            solver='newton-cg',
            fit_intercept=True
        ).fit(X_train_scale, y_train)

        y_pred = lr.predict(X_val_scale)
        cv_lr_f1.append(f1_score(y_val, y_pred, average='binary'))

        # Logistic Regression SGD
        sgd = linear_model.SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            loss='log_loss',
            class_weight='balanced'
        ).fit(X_train_scale, y_train)

        y_pred = sgd.predict(X_val_scale)
        cv_lrsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

        # SGD Modified Huber
        sgd_huber = linear_model.SGDClassifier(
            max_iter=1000,
            tol=1e-3,
            alpha=20,
            loss='modified_huber',
            class_weight='balanced'
        ).fit(X_train_scale, y_train)

        y_pred = sgd_huber.predict(X_val_scale)
        cv_svcsgd_f1.append(f1_score(y_val, y_pred, average='binary'))

    print(f'Logistic Regression Val f1: {np.mean(cv_lr_f1):.3f} +- {np.std(cv_lr_f1):.3f}')
    print(f'Logisitic Regression SGD Val f1: {np.mean(cv_lrsgd_f1):.3f} +- {np.std(cv_lrsgd_f1):.3f}')
    print(f'SVM Huber Val f1: {np.mean(cv_svcsgd_f1):.3f} +- {np.std(cv_svcsgd_f1):.3f}')
    