import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer,sent_tokenize
from fetch import fetch_filing

def replace_url(txt):
    '''
    Replace -index.htm with .txt in the urls
    '''
    txt=txt.replace('-index.htm','.txt')
    return txt

def word_list(word_type):
    '''
    Filters specific word types and returns list with all words in lowercase
    '''
    filter = list(master_dict['Word'].loc[master_dict[word_type]!=0]) 
    filter = [x.lower() for x in filter]
    return filter

def remove_stopwords(tokens):
    '''
    Remove Stop words
    '''
    return list(filter(lambda x: x not in stop_words, tokens))

def tokenize(txt):
    '''
    Tokenize text
    '''
    txt = txt.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(txt)
    return remove_stopwords(tokens)

def calc_positive_score(txt):
    '''
    Calculates positive word score
    '''
    n_positive = 0
    tokens = tokenize(txt)
    for token in tokens:
        if token in positive_words:
            n_positive += 1
    return n_positive

def calc_negative_score(txt):
    '''
    Calculates negative word score
    '''
    n_negative = 0
    tokens = tokenize(txt)
    for token in tokens:
        if token in negative_words:
            n_negative -= 1
    negative_score = -1 * n_negative
    return negative_score

def calc_polarity_score(positive_score,negative_score):
    '''
    Calculates polarity score
    '''
    polarity_score = (positive_score-negative_score)/((positive_score+negative_score)+0.000001)
    return polarity_score

def calc_avg_sentence_length(txt):
    '''
    Calculates average sentence length
    '''
    avg_sentence_length = 0
    sentences = sent_tokenize(txt)
    tokens = tokenize(txt)
    if len(sentences) != 0:
        avg_sentence_length = len(tokens)/len(sentences)
    return avg_sentence_length

def calc_wordcount(txt):
    '''
    Calculates word count
    '''
    n_words = len(tokenize(txt))
    return n_words

def calc_complexword_count(txt):
    '''
    Calculates complex word count
    '''
    tokens = tokenize(txt)
    n_complexwords = 0
    vowels = 'aeiou'
    for token in tokens:
        n_syllables = 0
        if token.endswith(('es','ed')):
            pass
        if token[0] in vowels:
            n_syllables += 1
        for i in range(1,len(token)):
            if token[i] in vowels and token[i-1] not in vowels:
                n_syllables += 1
        if n_syllables == 0:
            n_syllables += 1
        if n_syllables > 2:
            n_complexwords += 1
    return n_complexwords

def calc_percentage_complexwords(txt):
    '''
    Calculates complex word percentage
    '''
    percentage_complexwords = 0
    n_complexwords = calc_complexword_count(txt)
    n_words = calc_wordcount(txt)
    if n_words != 0:
        percentage_complexwords = n_complexwords/n_words
    return percentage_complexwords

def calc_fog_index(avg_sentence_length,percentage_complexwords):
    '''
    Calculates fog index
    '''
    fog_index = 0.4 * (avg_sentence_length + percentage_complexwords)
    return fog_index

def calc_uncertainty_score(txt):
    '''
    Calculates uncertainity score
    '''
    uncertainty_score = 0
    tokens = tokenize(txt)
    for token in tokens:
        if token in uncertainty_words:
            uncertainty_score += 1
    return uncertainty_score

def calc_constraining_score(txt):
    '''
    Calculates constraining score (No. of constaring words)
    '''
    constraining_score = 0
    tokens = tokenize(txt)
    for token in tokens:
        if token in constraining_words:
            constraining_score +=1
    return constraining_score

def calc_litigious_score(txt):
    '''
    Calculates litigious score (No. of litigious words)
    '''
    litigious_score=0
    tokens = tokenize(txt)
    for token in tokens:
        if token in litigious_words:
            litigious_score +=1
    return litigious_score

def calc_positiveword_proportion(positive_score,n_words):
    '''
    Calculates positive word proportion
    '''
    positiveword_proportion = 0
    if n_words != 0:
        positiveword_proportion = positive_score/n_words
    return positiveword_proportion

def calc_negativeword_proportion(negative_score,n_words):
    '''
    Calculates negative word proportion
    '''
    negativeword_proportion = 0
    if n_words != 0:
        negativeword_proportion = negative_score/n_words
    return negativeword_proportion

#Prepping the master dictionary
master_dict=pd.read_csv('resources/LoughranMcDonald_MasterDictionary_2018.csv')

positive_words = word_list('Positive')
negative_words = word_list('Negative')
uncertainty_words = word_list('Uncertainty')
constraining_words = word_list('Constraining')
litigious_words = word_list('Litigious')

stop_words = list(stopwords.words('english'))
stop_words = [x.lower() for x in stop_words]

#Importing the info file
info = pd.read_excel('resources/tesla.xlsx')

report = pd.DataFrame()
all_filings = []
report['Filing URL'] = info['Filings URL'].map(replace_url)

for url in report['Filing URL']:
    all_filings.append(fetch_filing(url))

report['Positive score'] = list(map(calc_positive_score, all_filings))
report['Negative score'] = list(map(calc_negative_score, all_filings))
report['Polarity score'] = np.vectorize(calc_polarity_score)(report['Positive score'], report['Negative score'])
report['Average Sentence Length'] = list(map(calc_avg_sentence_length, all_filings))
report['Percentage of Complex words'] = list(map(calc_percentage_complexwords, all_filings))
report['fog index'] = np.vectorize(calc_fog_index)(report['Average Sentence Length'], report['Percentage of Complex words'])
report['Complex word count'] = list(map(calc_complexword_count, all_filings))    
report['Word count'] = list(map(calc_wordcount, all_filings))
report['Uncertainty Score'] = list(map(calc_uncertainty_score, all_filings))
report['Constraining word Score'] = list(map(calc_constraining_score, all_filings))
report['Litigious word Score'] = list(map(calc_litigious_score, all_filings))
report['Positive word_proportion'] = np.vectorize(calc_positiveword_proportion)(report['Positive score'], report['Word count'])
report['Negative word proportion'] = np.vectorize(calc_negativeword_proportion)(report['Negative score'], report['Word count'])

print(report)