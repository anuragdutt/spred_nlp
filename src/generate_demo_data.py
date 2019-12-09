import os
import sys
import numpy as np
import pandas as pd
import re
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import pickle
import pandas
import random


if __name__ == "__main__":

	max_words = 34603
	embed_dim = 100

	wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
	cik_df = pd.read_html(wiki_url,header=0,index_col=0)[0]
	cik_df['GICS Sector'] = cik_df['GICS Sector'].astype("category")
	cik_df['GICS Sub Industry'] = cik_df['GICS Sector'].astype("category")

	cik_df = cik_df.loc[:, ['CIK', 'GICS Sector', 'GICS Sub Industry']]
	cik_df.columns = ['cik', 'GICS Sector', 'GICS Sub Industry']

	df = pd.read_pickle("../data/pickles/lemmatized_data.pickle")
	print("******************************************************")

	df = pd.read_csv("../data/embedded_data/final_dataset.csv.gz", compression = "gzip")
	df = df.ix[random.sample(df.index, 3000)]
	df.to_csv("../data/embedded_data/sample_data.csv.gz", compression = "gzip", index = False)
	df = pd.read_csv("../data/embedded_data/sample_data_sample.csv.gz", compression = "gzip")
	# df = df.dropna()
	print(df.shape)
	exit(0)
