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

	df = pd.read_csv("../data/embedded_data/final_dataset.csv.gz", compression = "gzip")
	df = df.ix[random.sample(df.index, 3000)]
	df.to_csv("../data/embedded_data/sample_data.csv.gz", compression = "gzip", index = False)
	# df = pd.read_csv("../data/embedded_data/sample_data_mlp.csv.gz", compression = "gzip")
	# df = df.dropna()
	print(df.shape)
	
