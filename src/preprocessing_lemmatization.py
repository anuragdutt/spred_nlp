import pandas as pd
import numpy as np
import tensorflow
import ast
import re
import os
import pickle

from nltk.corpus import stopwords
stop_words = stopwords.words("english")
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
wordnet_lemmatizer = WordNetLemmatizer()
import string
punctuations = string.punctuation
import matplotlib.pyplot as plt
import seaborn as sns

import dask.dataframe as dd
from dask.multiprocessing import get
from dask.diagnostics import ProgressBar

def cleanup_text(doc, logging=False):
    doc = re.sub( '\s+', ' ', doc ).strip()
    doc = nlp(doc, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc]
    tokens = [tok for tok in tokens if tok.isalpha()]
    tokens = [tok for tok in tokens if tok not in stop_words and tok not in punctuations]
    tokens_len = len(tokens)
    tokens = ' '.join(tokens)
    return tokens,tokens_len

def nltk_tokenizer(text):
    try:
        tokens = [word for word in word_tokenize(text) if word.isalpha()]
        tokens = list(filter(lambda t: t not in punctuations, tokens))
        tokens = list(filter(lambda t: t.lower() not in stop_words, tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        filtered_tokens = list(
            map(lambda token: wordnet_lemmatizer.lemmatize(token.lower()), filtered_tokens))
        filtered_tokens = list(filter(lambda t: t not in punctuations, filtered_tokens))
        return filtered_tokens
    except Exception as e:
        raise e

def dask_tokenizer(df):
    df['processed_text'] = df['text'].map(nltk_tokenizer)
    df['text_len'] = df['processed_text'].map(lambda x: len(x))
    return df



if __name__ == "__main__":

	count = 0
	f = pd.read_csv("../data/8k-gz/AAPL.csv.gz", compression = "gzip")
	col = f.columns.tolist()
	col.extend(['signal'])
	dat = pd.DataFrame(columns = col)

	lemmatize_dat = []

	for f in os.listdir("../data/8k-gz"):
		df = pd.read_csv(os.path.join('../data/8k-gz', f), compression = "gzip")
		print(df['ticker'].unique())

		df["signal"] = df['price_change'].map(lambda x: "stay" if -1<x<1 else ("up" if x>1 else "down"))
		ldf = dask_tokenizer(df)
		dat = dat.append(ldf)
		lemmatize_dat.append(dat)

		count = count + 1
		if count == 3:

			with open("../data/pickles/lemmatized_data.pickle", "wb") as fwrite:
				pickle.dump(lemmatize_dat, fwrite)
				fwrite.close()
			exit(0)
