"""Config File of potential breach words"""

############################### LIBRARIES #####################################
import re
import string

from itertools import permutations, combinations

from nltk.tokenize import WordPunctTokenizer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures, QuadgramAssocMeasures


######################### SYNTHESIZE ##############################

def synthesize_words(word_list):
    """
    INPUT
    word_list: list of words 
    OUTPUT
    return a list of words that are made up of combinations of the original list
    """

    # Lowercase and joined the list into a single string
    lowercase_list = []
    for words in word_list:
        lowercase_list.append(words.lower())
        joined_string = ' '.join(lowercase_list)

    # Initialise Tokenizer
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(joined_string)

    bigram_finder = BigramCollocationFinder.from_words(tokens)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500) # take 500 bigrams with highest chi_sq , pmi

    trigram_finder = TrigramCollocationFinder.from_words(tokens)
    trigrams = trigram_finder.nbest(TrigramAssocMeasures.chi_sq, 500)

    quadgram_finder = QuadgramCollocationFinder.from_words(tokens)
    quadgrams = quadgram_finder.nbest(QuadgramAssocMeasures.chi_sq, 500)

    # Add biwords to the list of terms
    biwords, biperm = [], []
    for tuplestring in bigrams:
        biwords.append(' '.join(tuplestring))
    # Further permutate the words -- swwitching the order of the biwords around
    for each_word in biwords:
        for each in permutate(each_word):
            biperm.append(each)

    # Add triwords to the list of terms
    triwords, triperm = [], []
    for tuplestring in trigrams:
        triwords.append(' '.join(tuplestring))
    for each_word in triwords: 
        for each in permutate(each_word):
            triperm.append(each)

    # Add quadwords to the list of terms
    quadwords,qperm = [], []
    for tuplestring in quadgrams:
        quadwords.append(' '.join(tuplestring))
    for each_word in quadwords: 
        for each in permutate(each_word):
            qperm.append(each)  

    # return unique set of words
    return set(lowercase_list + biwords + biperm + triwords + triperm) #+ quadwords + qpe

def get_combinations(single_string):
    """
    INPUT
    single_string: a string of different terms
    OUTPUT
    returns a list of different combinations of the original string of terms
    """
    # remove punctuation
    single_string = re.sub('[%s]' % re.escape(string.punctuation), "", single_string)
    # split string by space or brackets
    string_list = re.split(r"[()| ]+", single_string)
    word_combi = [" ".join(items) for items in combinations(string_list, r=len(string_list)-1)]

    return set(word_combi)

def permutate(single_string):
  """
  INPUT
  single_string: a string of different terms
  OUTPUT
  returns a list of different permutations of the original string of terms
  """
  # split string by space or brackets
  string_list = re.split(r"[()| ]+", single_string)
  word_perm = [" ".join(items) for items in permutations(string_list, r=len(string_list))]

  return set(word_perm)

############################### LIST OF WORDS #####################################
given_list_of_designations = [ "Finance company"
                            , "Legal entity"
                            , "Legal company"
                            , "Insurance company"
                            , "Corporate financial solutions"
                            , "Medical concierge services"
                            , "Financial adviser"
                            , "Financial advisor"
                            , "Finance consultant"
                            , "Financial doctor"
                            , "Life insurance broker"
                            , "Portfolio manager"
                            , "Fund manager"
                            , "Asset manager"
                            , "Risk adviser"
                            , "Risk advisor"
                            , "Wealth management specialist"]


given_list_of_promo= [ "discount"
                      ,"promotion"
                      ,"promo"
                      ,"rebate"
                      ,"deduct"
                      ,"price"
                      ,"sale"
                      ,"concession"
                      ,"premium"
                      ,"bargain"
                      ,"discounted"
                      ,"wholesale"
                      ,"salary"]




