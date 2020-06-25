#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:52:33 2020

@author: ahmetcakmak
"""


import numpy as np
import re
import itertools
from collections import Counter

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed()
import nltk
nltk.download('wordnet')
import re
import pandas as pd


# Stemming allows us to use the root of the word only. 
# Lemmatization allows us that words in third person are changed to first person 
# and verbs in past and future tenses are changed into present.
stemmer = SnowballStemmer("english")

def lemm_stemm(x_text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(x_text, pos='v'))

# Tokenize and lemmatize. Tokenization splits the text into sentences and the sentences into words. 
# Words that have fewer than 3 characters are removed.
def preprocess(x_text):
    result=''
    for token in gensim.utils.simple_preprocess(x_text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token!='the':
            # Perform lemm_stemm on the token 
            result+=' '+lemm_stemm(token) 
    return result

# Clean the text and lower case the words
def clean_str(string):

    string=re.sub(r'[^a-zA-Z ]+', '', string)
    
    return string.strip().lower()

# n=10 stands for each sentence contains 10 words. That is, in cat file, after lemmatize and stemming, each sentence will contain 10 words.
#  If a sentence has less than 10 words then it repeats the existing word from beginning. The same applied for the dog file.
def loadfromfile(file,minword=5,minchar=20,n=10,test=False):
    cat_examples = list(open(file, "r", encoding='latin-1').readlines())
    cat_examples= [clean_str(sent) for sent in cat_examples]
    CS=[]
    prev=''
    for i in range(0,len(cat_examples)):
        cat_examples[i]+=prev
        if len(cat_examples[i])>minchar and len(cat_examples[i].split(" "))>minword:
            CS.append(preprocess(cat_examples[i]))
            prev=''
        else:
            prev+=cat_examples[i]
    if test:
        CSS=[]
        for s in CS:
            s=s.split(" ")
            tmp=[]
            for ss in s:
                if len(ss)>2:
                    tmp.append(ss)
            start=len(tmp)
            for i in range(start,n):
                tmp.append(tmp[np.mod(i-start,start)])
            CSS.append(tmp[0:n])
    else:
        cat = ''
        for s in CS:
            cat+=s+' '
        
        cat_text =cat.split(" ")
        CS=[]
        for c in cat_text:
            if len(c)>2:
                CS.append(c)
        
        CSS=[]
        cut=list(range(0,len(CS),n))
        for i in range(0,len(cut)-1):
            CSS.append(CS[cut[i]:cut[i+1]])


    return CSS





# Load data from files and label it
def load_data_and_labels():

    n=10
    cat_examples=loadfromfile("data/catWiki.txt",n=n)
    dog_examples=loadfromfile("data/dogWiki.txt",n=n)

    test_examples=loadfromfile("data/testSentences.txt",minword=1,minchar=1,n=n,test=True)

    x_text = cat_examples + dog_examples

    # Generate labels
    cat_labels = [[0, 1] for _ in cat_examples]
    dog_labels = [[1, 0] for _ in dog_examples]
    y = np.concatenate([cat_labels, dog_labels], 0)
    return [x_text, y,test_examples]
    

    
# This function pads all sentences to the same length. The length is defined by the longest sentence. 
# Then returns padded sentences.
    
def pad_sentences(sentences, padding_word="<PAD/>"):

    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


# This buuilds a vocabulary mapping from word to index based on the sentences. 
# Then returns vocabulary mapping and inverse vocabulary mapping.

def build_vocab(sentences):

    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


# It maps sentences and labels to vectors based on a vocabulary.

def build_input_data(sentences, labels, vocabulary):

    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


# Loads and preprocessed data for the dataset. Returns input vectors, labels, vocabulary, and inverse vocabulary.

def load_data():

    # Load and preprocess data
    sentences, labels,test = load_data_and_labels()
       
# Build a dictionary from 2529 train vocabulary. Then builds an input with addressing each word by a number
# There exist 405x10 = 4050 words in dog file, and 371x10 = 3710 words in cat file 
    vocabulary, vocabulary_inv = build_vocab(sentences)
    x, y = build_input_data(sentences, labels, vocabulary)

# We do not need to apply the same enumeration method for test file. 
# Also, if a sentence left out by  one word after cleaning in test file, it will append the same word to
# make it 10 words.
    xtest=[]
    for sentences in test:
        tmp=[]
        for word in sentences:
            try:
                tmp.append(vocabulary[word])
            except:
                pass
        if len(tmp)<x.shape[1]:
            start=len(tmp)
            for i in range(start,x.shape[1]):
                tmp.append(tmp[np.mod(i-start,start)])
        xtest.append(tmp)
        
    xtest = np.array(xtest)   
# The dimension of x is: 776x10 (776 rows, 10 columns) 
#The dimension of xtest is: 10x10 (1o rows, 10 columns.)   
    return [x, y,xtest,test, vocabulary, vocabulary_inv]

