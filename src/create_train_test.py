import os
import sys
import numpy as np
import pandas as pd
import re
import ast
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
import argparse
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

global aux_shape
global vocab_size  
global embed_dim
global max_words



def load_embeddings(vec_file):
    print("Loading Glove Model")
    f = open(vec_file,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done. {} words loaded!".format(len(model)))
    return model


def tokenize_and_pad(docs,max_words=34603):
    global t
    t = Tokenizer()
    t.fit_on_texts(docs)
    docs = pad_sequences(sequences = t.texts_to_sequences(docs),maxlen = max_words, padding = 'post')
    global vocab_size
    vocab_size = len(t.word_index) + 1
    
    return docs

def oversample(X,docs,y):
    # Get number of rows with imbalanced class
    target = y.sum().idxmax()
    n = y[target].sum()
    # identify imbalanced targets
    imbalanced = y.drop(target,axis=1)
    #For each target, create a dataframe of randomly sampled rows, append to list
    append_list =  [y.loc[y[col]==1].sample(n=n-y[col].sum(),replace=True,random_state=20) for col in imbalanced.columns]
    append_list.append(y)
    y = pd.concat(append_list,axis=0)
    # match y indexes on other inputs
    X = X.loc[y.index]
    docs = pd.DataFrame(docs_train,index=y_train.index).loc[y.index]
    assert (y.index.all() == X.index.all() == docs.index.all())
    return X,docs.values,y



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Test Split')
    parser.add_argument('--max_words', type=int, help="Max words considered for each 8k filing", default=34603)
    parser.add_argument('--embed_dim', type=int, help="Dimension of Embeddings", default="100")
    parser.add_argument('--embed-file', type=str, help="embedding location", default='../data/glove/glove.6B.100d.txt')
    parser.add_argument('--filing-file', type=str, help="data for all sec filings", default="../data/embedded_data/sample_data.csv.gz")
    parser.add_argument('--sp-summary', type=str, help="save folder for all generated summary stats", default="../data/sumstats")
    parser.add_argument('--sp-pickles', type=str, help="save folder for all generated pickles", default="../data/pickles")


    args = parser.parse_args()

    max_words = args.max_words
    embed_dim = args.embed_dim
    embed_file = args.embed_file
    filing_file = args.filing_file
    sp_summary = args.sp_summary
    sp_pickles = args.sp_pickles


    wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    cik_df = pd.read_html(wiki_url,header=0,index_col=0)[0]
    cik_df['GICS Sector'] = cik_df['GICS Sector'].astype("category")
    cik_df['GICS Sub Industry'] = cik_df['GICS Sector'].astype("category")

    cik_df = cik_df.loc[:, ['CIK', 'GICS Sector', 'GICS Sub Industry']]
    cik_df.columns = ['cik', 'GICS Sector', 'GICS Sub Industry']




    # df = pd.read_pickle("../data/pickles/lemmatized_data.pickle")
    # print("******************************************************")

    df = pd.read_csv(filing_file, compression = "gzip")
    # df = df.loc[:200]
    # df.to_csv("../data/embedded_data/sample_data.csv.gz", compression = "gzip", index = False)
    # df = pd.read_csv("../data/embedded_data/sample_data.csv.gz", compression = "gzip")
    # exit(0)

    df = df.dropna()
    print(df.shape)
    # exit(0)

    df = pd.merge(df, cik_df, on = "cik", how = "left")


    mlb = MultiLabelBinarizer()
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('items')),columns=mlb.classes_,),sort=False,how="left")
    df_sumstats_path = os.path.join(sp_summary, "main_data.csv.gz")
    df.to_csv(df_sumstats_path, compression = "gzip")

    cols = ['GICS Sector','vix','rm_week','rm_month', 'rm_qtr', 'rm_year']
    cols.extend(list(mlb.classes_))
    X = df[cols]
    docs = df['processed_text']
    y = df['signal']    


    # Get Dummies

    docs = tokenize_and_pad(docs)
    X = pd.get_dummies(columns = ['GICS Sector'],prefix="sector",data=X)
    y = pd.get_dummies(columns=['signal'],data=y)


    aux_shape = len(X.columns)
    X_train, X_test, y_train, y_test, docs_train, docs_test = train_test_split(X, y,docs,
                                                    stratify=y, 
                                                    test_size=0.3,
                                                    random_state = 20)


    cont_features = ['vix','rm_week','rm_month', 'rm_qtr', 'rm_year']
    aux_features = cont_features + [item for item in mlb.classes_]
    x_scaler = StandardScaler()
    X_train[cont_features] = x_scaler.fit_transform(X_train[cont_features])
    X_test[cont_features] = x_scaler.transform(X_test[cont_features])

    X_train, docs_train, y_train = oversample(X_train, docs_train, y_train)


    embeddings_index = load_embeddings(embed_file)
    words_not_found = []


    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)

    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))



    # Save data
    fn_docs_train = os.path.join(sp_pickles, "docs_train.npy")
    fn_docs_test = os.path.join(sp_pickles, "docs_test.npy")

    np.save(fn_docs_train, docs_train)
    np.save(fn_docs_test,docs_test)

    fn_x_train = os.path.join(sp_pickles, "X_train.pkl")
    fn_x_test = os.path.join(sp_pickles, "X_test.pkl")

    X_train.to_pickle(fn_x_train)
    X_test.to_pickle(fn_x_test)

    fn_y_train = os.path.join(sp_pickles, "y_train.pkl")
    fn_y_test = os.path.join(sp_pickles, "y_test.pkl")

    y_train.to_pickle(fn_y_train)
    y_test.to_pickle(fn_y_train) 

    embedding_path = os.path.join(sp_pickles, "embedding_matrix.npy")
    np.save(embedding_path,embedding_matrix)


    config = {'embedding_matrix':embedding_matrix, 
              'aux_shape':aux_shape, 
              'vocab_size':vocab_size, 
              'embed_dim':embed_dim, 
              'max_words':max_words}
    fn_config = os.path.join(sp_pickles, 'config.pkl')
    # config.to_pickle(fn_config)
    with open(fn_config, 'wb') as f:
        pickle.dump(config, f)   


    fn_train = os.path.join(sp_pickles, 'train_input.pkl')
    train_input = {'docs_train':docs_train, 'X_train':X_train, 'y_train':y_train}
    # train_input.to_pickle(fn_train)
    with open(fn_train, 'wb') as f:
        pickle.dump(train_input, f)   


    fn_test = os.path.join(sp_pickles, 'test_input.pkl')
    test_input = {'docs_test':docs_test, 'X_train':X_test, 'ytrain':y_test}
    # test_input.to_pickle(fn_test)
    with open(fn_test, 'wb') as f:
        pickle.dump(test_input, f)   
