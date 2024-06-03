import pycountry
#import nltk
from nltk.corpus import names
from contextlib import redirect_stdout 
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.metrics.distance import jaccard_distance
from nltk.util import ngrams
from nltk.corpus import words
import string
import pandas as pd
import pickle
import math
import re
import MAIN
additional_stop_words = [
    'infinite','infinity','difference','nt','quick','difference','find','soo' ,'could','would','continue','able', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 
    'added', 'adj', 'affected', 'affecting', 'affects', 'afterwards', 'ah', 'almost', 'alone', 
    'along', 'already', 'also', 'although', 'always', 'among', 'amongst', 'announce', 
    'another', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 
    'anywhere', 'apparently', 'approximately', 'arent', 'arise', 'around', 'aside', 'ask', 
    'asking', 'auth', 'available', 'away', 'awfully', 'back', 'became', 'become', 'becomes', 
    'becoming', 'beforehand', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 
    'believe', 'beside', 'besides', 'beyond', 'biol', 'brief', 'briefly', 'ca', 'came', 
    'cannot', 'can\'t', 'cause', 'causes', 'certain', 'certainly', 'co', 'com', 'come', 
    'comes', 'contain', 'containing', 'contains', 'couldnt', 'date', 'different', 'done', 
    'downwards', 'due', 'ed', 'edu', 'effect', 'eight', 'eighty', 'else', 'elsewhere', 'end', 
    'ending', 'enough', 'especially', 'et', 'et-al', 'etc', 'ever', 'every', 'everybody', 
    'everyone', 'everything', 'everywhere', 'ex', 'except', 'far', 'ff', 'fifth', 'first', 
    'five', 'fix', 'followed', 'following', 'follows', 'former', 'formerly', 'forth', 'found', 
    'four', 'furthermore', 'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives', 
    'giving', 'go', 'goes', 'gone', 'got', 'gotten', 'happens', 'hardly', 'hed', 'hence', 
    'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hes', 'hi', 'hid', 'hither', 
    'home', 'howbeit', 'however', 'hundred', 'id', 'ie', 'im', 'immediate', 'immediately', 
    'importance', 'important', 'inc', 'indeed', 'index', 'information', 'instead', 'invention', 
    'inward', 'itd', 'it\'ll', 'j', 'k', 'keep', 'keeps', 'kept', 'kg', 'km', 'know', 'known', 
    'knows', 'largely', 'last', 'lately', 'later', 'latter', 'latterly', 'lest', 'let', 
    'lets', 'like', 'liked', 'likely', 'line', 'little', 'look', 'looking', 'looks', 'made', 
    'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 
    'meanwhile', 'merely', 'mg', 'might', 'million', 'miss', 'ml', 'moreover', 'mostly', 
    'mr', 'mrs', 'much', 'must', 'myself', 'na', 'name', 'namely', 'nay', 'nd', 'near', 
    'nearly', 'necessarily', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless',
    'new', 'next', 'nine', 'ninety', 'nobody', 'non', 'none', 'nonetheless', 'noone', 
    'normally', 'nos', 'noted', 'nothing', 'nowhere', 'obtain', 'obtained', 'obviously', 
    'often', 'oh', 'ok', 'okay', 'old', 'omitted', 'one', 'ones', 'onto', 'ord', 'others', 
    'otherwise', 'outside', 'overall', 'owing', 'page', 'pages', 'part', 'particular', 
    'particularly', 'past', 'per', 'perhaps', 'placed', 'please', 'plus', 'poorly', 
    'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present', 'previously', 
    'primarily', 'probably', 'promptly', 'proud', 'provides', 'put', 'que', 'quickly', 
    'quite', 'ran', 'rather', 'rd', 'readily', 'really', 'recent', 'recently', 'ref', 
    'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 
    'respectively', 'resulted', 'resulting', 'results', 'run', 'said', 'saw', 'say', 
    'saying', 'says', 'sec', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 
    'seems', 'seen', 'self', 'selves', 'sent', 'seven', 'several', 'shall', 'shed', 
    'shes', 'show', 'showed', 'shown', 'showns', 'shows', 'significant', 'significantly', 
    'similar', 'similarly', 'since', 'six', 'slightly', 'somebody', 'somehow', 'someone', 
    'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 
    'sorry', 'specifically', 'specified', 'specify', 'specifying', 'still', 'stop', 
    'strongly', 'sub', 'substantially', 'successfully', 'sufficiently', 'suggest', 'sup', 
    'sure', 'take', 'taken', 'taking', 'tell', 'tends', 'th', 'thank', 'thanks', 'thanx', 
    'thats', 'that\'ve', 'thence', 'thereafter', 'thereby', 'thered', 'therefore', 'therein', 
    'there\'ll', 'thereof', 'therere', 'theres', 'thereto', 'thereupon', 'there\'ve', 
    'theyd', 'theyre', 'thick', 'thin', 'thou', 'though', 'thoughh', 'thousand', 'throug', 
    'throughout', 'thru', 'thus', 'til', 'tip', 'today', 'together', 'took', 'toward', 
    'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'twice', 'two', 'un', 
    'unfortunately', 'unless', 'unlike', 'unlikely', 'unto', 'upon', 'ups', 'use', 'used', 
    'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'value', 'various', 
    'very', 'via', 'viz', 'vol', 'vols', 'vs', 'want', 'wants', 'wasnt', 'way', 'wed', 
    'welcome', 'went', 'werent', 'whatever', 'what\'ll', 'whats', 'whence', 'whenever', 
    'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 
    'whether', 'whim', 'whither', 'whod', 'whoever', 'whole', 'who\'ll', 'whomever', 
    'whos', 'whose', 'widely', 'willing', 'wish', 'within', 'without', 'wont', 'words', 
    'world', 'wouldnt', 'www', 'yes', 'yet', 'youd', 'youre', 'zero'
]
correct_word=words.words()
stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(additional_stop_words)
stop2=['well','without','with','ever','never','usual', 'much', 'less','lot',"ok","''",'""','whi','new', '’', '..',"“", 'etc','','example', 'blah', 'ummm+']
lemmatizer = WordNetLemmatizer()

male_names = set(names.words('male.txt'))
female_names = set(names.words('female.txt'))
all_names = male_names.union(female_names)

def get_wordnet_pos(tag_parameter):

    tag = tag_parameter[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    
    return tag_dict.get(tag, wordnet.NOUN)

def is_number(token1):
    try:
        float(token1)
        return True
    except ValueError:
        return False


def text_processing(doc_column):
    words_doc = []
    if not isinstance(doc_column, str):
        return []
    doc_column = re.sub(r'http\S+', '', doc_column) # Remove URLs  # Remove non-alphanumeric characters and extra spaces, tabs, and newlines
    # Remove URLs starting with www.
    doc_column = re.sub(r'www\.\S+', '', doc_column)
    doc_column = re.sub(r'\t+', '', doc_column)
    doc_column = re.sub(r'\s+', ' ', doc_column)
    words_doc.append(word_tokenize(doc_column))
    filtered_docs = []
    
    for row in words_doc:
        pos_tags = pos_tag(row)
        lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
        for lemma_word in lemmatized_words:
            if lemma_word not in all_names:
                lower_word = lemma_word.lower()
                word_no_punct = lower_word.translate(str.maketrans('', '', string.punctuation))
                if word_no_punct not in stop_words and word_no_punct not in stop2:
                    if is_number(word_no_punct):
                        word_no_punct2=word_no_punct
                    elif word_no_punct not in MAIN.unique_term_proces.keys():
                        if (len(word_no_punct) <2):
                            continue
                        try:
                            word_no_punct=pycountry.countries.lookup(word_no_punct)[0].name
                            word_no_punct = word_no_punct.lower()
                        except:
                            word_no_punct=word_no_punct
                        temp=[(jaccard_distance(set(ngrams(word_no_punct,2)),set(ngrams(w,2))),w) for w in correct_word if w[0]==word_no_punct[0]]
                        if len(temp) >0:
                            word_no_punct2=sorted(temp,key=lambda val:val[0])[0][1]
                            MAIN.unique_term_proces[word_no_punct]=word_no_punct2
                        else:
                            word_no_punct2=word_no_punct
                            MAIN.unique_term_proces[word_no_punct]=word_no_punct
                    else:
                        word_no_punct2=MAIN.unique_term_proces[word_no_punct]
                    filtered_docs.append(word_no_punct2)
    return filtered_docs


def remove_dublicated_words(filtered_docs,unique_filtered_docs):
    # Remove duplicated words
    for word in filtered_docs:
        if word not in unique_filtered_docs:
            unique_filtered_docs.append(word)

    return unique_filtered_docs

