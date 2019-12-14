{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.7.0"
    },
    "colab": {
      "name": "Spred_NLP_LSTM_Parser.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "toc_visible": true
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "tsEYw0WomO4C",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 101
        },
        "outputId": "97452712-77fd-49b8-a643-fc9e5d5f1c97"
      },
      "source": [
        "import os\n",
        "\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "import math\n",
        "import random\n",
        "\n",
        "import datetime\n",
        "import time\n",
        "import io\n",
        "\n",
        "import tensorflow as tf\n",
        "from keras import backend as K\n",
        "\n",
        "from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Dropout, Bidirectional, Input, concatenate, add, multiply\n",
        "from keras.layers import Conv1D, MaxPooling1D, Flatten, Reshape, GlobalMaxPooling1D, Highway, Permute, Lambda\n",
        "from keras.layers.advanced_activations import PReLU, LeakyReLU\n",
        "from keras.models import Model\n",
        "from keras.optimizers import Adam\n",
        "\n",
        "from nltk import pos_tag, word_tokenize\n",
        "from nltk.corpus import wordnet\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "\n",
        "from sklearn.metrics import mean_squared_error\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.decomposition import TruncatedSVD\n",
        "from sklearn.svm import SVR\n",
        "\n",
        "from gensim.models.wrappers import FastText"
      ],
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/html": [
              "<p style=\"color: red;\">\n",
              "The default version of TensorFlow in Colab will soon switch to TensorFlow 2.x.<br>\n",
              "We recommend you <a href=\"https://www.tensorflow.org/guide/migrate\" target=\"_blank\">upgrade</a> now \n",
              "or ensure your notebook will continue to use TensorFlow 1.x via the <code>%tensorflow_version 1.x</code> magic:\n",
              "<a href=\"https://colab.research.google.com/notebooks/tensorflow_version.ipynb\" target=\"_blank\">more info</a>.</p>\n"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {
            "tags": []
          }
        },
        {
          "output_type": "stream",
          "text": [
            "Using TensorFlow backend.\n"
          ],
          "name": "stderr"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "F2qgutc2sn6b",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "from gensim.models.wrappers import FastText"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ep1sh5zzmO4U",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 131
        },
        "outputId": "c3b54af1-3452-47a6-f2c3-f40f8955713c"
      },
      "source": [
        "wordnet_lemmatizer = WordNetLemmatizer()\n",
        "\n",
        "os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' \n",
        "\n",
        "session = tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads = 0,\n",
        "                                             intra_op_parallelism_threads = 0,\n",
        "                                             log_device_placement = True))\n",
        "\n",
        "K.set_session(session)"
      ],
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Device mapping:\n",
            "/job:localhost/replica:0/task:0/device:XLA_CPU:0 -> device: XLA_CPU device\n",
            "/job:localhost/replica:0/task:0/device:XLA_GPU:0 -> device: XLA_GPU device\n",
            "/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla P100-PCIE-16GB, pci bus id: 0000:00:04.0, compute capability: 6.0\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "q_9IJcD_mO4f",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 840
        },
        "outputId": "28c91ff7-7061-4e48-d8cb-b4774052a6cc"
      },
      "source": [
        "# Reading dataset from files\n",
        "\n",
        "def isNaN(num):\n",
        "    return num != num\n",
        "\n",
        "def is_ascii(s):\n",
        "    return (len(s) == len(s.encode()))\n",
        "\n",
        "def read_pr(file, newline_token):\n",
        "    xl = pd.ExcelFile(file)\n",
        "    df = xl.parse(\"SignificantDevelopment\")\n",
        "\n",
        "    df['Release Date'] = df['Release Date'].apply(lambda x: x.split(\" \")[0])\n",
        "    \n",
        "    non_ascii_count = 0\n",
        "\n",
        "    for i in df.index:\n",
        "        if isNaN(df.loc[i, 'Headline']):\n",
        "            df.at[i, :] = np.nan\n",
        "        elif (not is_ascii(df.loc[i, 'Headline'])):\n",
        "                df.at[i, :] = np.nan\n",
        "                non_ascii_count += 1\n",
        "\n",
        "    df.dropna(inplace = True)\n",
        "    \n",
        "    for d in set(df['Release Date']):\n",
        "        df_d = df[df['Release Date'] == d]\n",
        "\n",
        "        if (len(df_d.index) > 1):\n",
        "            df_row = {}\n",
        "            df_row['Topic'] = newline_token.join(set(df_d['Topic'].values))\n",
        "            df_row['Release Date'] = d\n",
        "            df_row['Company'] = df_d['Company'].values[0]\n",
        "            df_row['Headline'] = newline_token.join(df_d['Headline'].values)\n",
        "            \n",
        "            df = df.drop(df_d.index)\n",
        "            df = df.append(df_row, ignore_index=True)\n",
        "    \n",
        "    return df, non_ascii_count\n",
        "\n",
        "def read_sd(file1, file2):\n",
        "    df1 = pd.read_excel(file1, sheet_name='Sheet1', skiprows=[0])\n",
        "    df2 = pd.read_excel(file2, sheet_name='Sheet1', skiprows=[0])\n",
        "    df = pd.concat([df1, df2], ignore_index=True, sort=True).copy().sort_values(by='Date')\n",
        "\n",
        "    if (len(df.index) != len(df1.index) + len(df2.index)):\n",
        "        print (\"Some rows are missed!\") \n",
        "\n",
        "    df.fillna(method='ffill', inplace = True)\n",
        "    df.fillna(df.mean(), inplace = True)\n",
        "\n",
        "    ci_s = [ci for c in df.columns for ci in c.split() if ci != 'Close']\n",
        "    stock_name = max(set(ci_s), key = ci_s.count)\n",
        "    df.columns = [c.replace(stock_name, '').strip() for c in df.columns]\n",
        "    \n",
        "    df['r_stock'] = (df['Close'] / df['Close'].shift(1) - 1.0)\n",
        "    df['r_index'] = (df['.SPX-US Close'] / df['.SPX-US Close'].shift(1) - 1.0)\n",
        "    df['Release Date'] = df['Date'].apply(lambda x: \"/\".join([x.split(\"/\")[1], x.split(\"/\")[2], x.split(\"/\")[0][-2:]]))\n",
        "    \n",
        "    df.dropna(inplace = True)\n",
        "    \n",
        "    cols = ['Release Date', 'r_stock', 'r_index', 'MFI', 'ForPE', 'SIP']\n",
        "    for c in cols:\n",
        "        if c not in df.columns:\n",
        "            df[c] = np.nan\n",
        "    \n",
        "    df['MFI'] = df['MFI'].shift(1)\n",
        "    df['ForPE'] = df['ForPE'].shift(1)\n",
        "    df['SIP'] = df['SIP'].shift(1)\n",
        "    \n",
        "    return df[cols]\n",
        "\n",
        "newline_token = \" nnnewlineee \"\n",
        "\n",
        "dfs = []\n",
        "for f in os.listdir(\"/content/drive/My Drive/Colab Notebooks/NLP_forecasting-master/press_releases/\"):\n",
        "    if f.endswith(\".xls\"):\n",
        "        df, non_ascii_count = read_pr(\"/content/drive/My Drive/Colab Notebooks/NLP_forecasting-master/press_releases/\" + f, newline_token)\n",
        "        \n",
        "        if (f not in os.listdir('/content/drive/My Drive/Colab Notebooks/NLP_forecasting-master/stock_data/10-14/')) or (f not in os.listdir('/content/drive/My Drive/Colab Notebooks/NLP_forecasting-master/stock_data/15-19/')):\n",
        "            print (\"Stock\", f ,\"was not found!\")\n",
        "            pass\n",
        "        sd = read_sd('/content/drive/My Drive/Colab Notebooks/NLP_forecasting-master/stock_data/10-14/' + f, '/content/drive/My Drive/Colab Notebooks/NLP_forecasting-master/stock_data/15-19/' + f)\n",
        "        \n",
        "        dfs.append(df.merge(sd, on = 'Release Date', how='left'))       \n",
        "        print (f, \"was processed. Non ascii lines removed:\", non_ascii_count, \"Total lines left:\", len(df.index))\n",
        "\n",
        "data = pd.concat(dfs, ignore_index = True)\n",
        "data = data[~data['r_stock'].isnull()]\n",
        "data.fillna(data.mean(), inplace = True)\n",
        "\n",
        "del df, dfs"
      ],
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "AAPL-US.xls was processed. Non ascii lines removed: 10 Total lines left: 249\n",
            "7203-TO.xls was processed. Non ascii lines removed: 10 Total lines left: 304\n",
            "005930-SE.xls was processed. Non ascii lines removed: 5 Total lines left: 182\n",
            "ABBV-US.xls was processed. Non ascii lines removed: 6 Total lines left: 161\n",
            "ABT-US.xls was processed. Non ascii lines removed: 4 Total lines left: 228\n",
            "ACN-US.xls was processed. Non ascii lines removed: 5 Total lines left: 165\n",
            "AMGN-US.xls was processed. Non ascii lines removed: 3 Total lines left: 243\n",
            "BA-US.xls was processed. Non ascii lines removed: 11 Total lines left: 374\n",
            "AMZN-US.xls was processed. Non ascii lines removed: 11 Total lines left: 252\n",
            "BAYN-XE.xls was processed. Non ascii lines removed: 22 Total lines left: 260\n",
            "BABA-US.xls was processed. Non ascii lines removed: 6 Total lines left: 145\n",
            "BBL-US.xls was processed. Non ascii lines removed: 4 Total lines left: 195\n",
            "BP.-LN.xls was processed. Non ascii lines removed: 17 Total lines left: 303\n",
            "FP-FR.xls was processed. Non ascii lines removed: 13 Total lines left: 203\n",
            "GOOGL-US.xls was processed. Non ascii lines removed: 16 Total lines left: 164\n",
            "GE-US.xls was processed. Non ascii lines removed: 23 Total lines left: 364\n",
            "FB-US.xls was processed. Non ascii lines removed: 11 Total lines left: 176\n",
            "C-US.xls was processed. Non ascii lines removed: 16 Total lines left: 277\n",
            "CSCO-US.xls was processed. Non ascii lines removed: 6 Total lines left: 162\n",
            "CVX-US.xls was processed. Non ascii lines removed: 6 Total lines left: 146\n",
            "BRKA-US.xls was processed. Non ascii lines removed: 14 Total lines left: 153\n",
            "HON-US.xls was processed. Non ascii lines removed: 5 Total lines left: 165\n",
            "HSBA-LN.xls was processed. Non ascii lines removed: 4 Total lines left: 217\n",
            "IBM-US.xls was processed. Non ascii lines removed: 8 Total lines left: 272\n",
            "INTC-US.xls was processed. Non ascii lines removed: 6 Total lines left: 160\n",
            "JNJ-US.xls was processed. Non ascii lines removed: 10 Total lines left: 291\n",
            "JPM-US.xls was processed. Non ascii lines removed: 8 Total lines left: 265\n",
            "MMM-US.xls was processed. Non ascii lines removed: 0 Total lines left: 140\n",
            "MRK-US.xls was processed. Non ascii lines removed: 11 Total lines left: 272\n",
            "MSFT-US.xls was processed. Non ascii lines removed: 10 Total lines left: 184\n",
            "PFE-US.xls was processed. Non ascii lines removed: 14 Total lines left: 416\n",
            "RDSA-AE.xls was processed. Non ascii lines removed: 27 Total lines left: 287\n",
            "ROG-VX.xls was processed. Non ascii lines removed: 20 Total lines left: 313\n",
            "NOVN-VX.xls was processed. Non ascii lines removed: 14 Total lines left: 342\n",
            "ORCL-US.xls was processed. Non ascii lines removed: 6 Total lines left: 143\n",
            "NESN-VX.xls was processed. Non ascii lines removed: 26 Total lines left: 111\n",
            "NOVO'B-KO.xls was processed. Non ascii lines removed: 10 Total lines left: 225\n",
            "T-US.xls was processed. Non ascii lines removed: 9 Total lines left: 193\n",
            "VOW3-XE.xls was processed. Non ascii lines removed: 7 Total lines left: 353\n",
            "WFC-US.xls was processed. Non ascii lines removed: 9 Total lines left: 167\n",
            "WMT-US.xls was processed. Non ascii lines removed: 10 Total lines left: 199\n",
            "XOM-US.xls was processed. Non ascii lines removed: 10 Total lines left: 204\n",
            "SIE-XE.xls was processed. Non ascii lines removed: 4 Total lines left: 253\n",
            "SAN-MC.xls was processed. Non ascii lines removed: 27 Total lines left: 208\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "M_hXhk4omO4n",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "outputId": "bd9cca62-8b29-494e-9e7a-1763166a130f"
      },
      "source": [
        "# Company to index\n",
        "\n",
        "Company_num = len(set(data['Company']))\n",
        "\n",
        "Company2idx = {}\n",
        "Company_embeddings = np.identity(Company_num, float)\n",
        "\n",
        "for company in set(data['Company']):\n",
        "    if company not in Company2idx:\n",
        "        Company2idx[company] = len(Company2idx)\n",
        "\n",
        "print('List of companies:')\n",
        "Company2idx.keys()"
      ],
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "List of companies:\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "dict_keys(['Amazon.com, Inc.', 'Facebook Inc', 'Wells Fargo & Co', 'Oracle Corporation', 'Total SA (ADR)', 'International Business Machines Corp.', 'Cisco Systems, Inc.', 'Accenture Plc', 'Berkshire Hathaway Inc.', 'Intel Corporation', 'Johnson & Johnson (OLD)', 'Novartis AG (ADR)', 'Microsoft Corporation', 'Travelers Group Inc', 'Roche Holding Ltd. (ADR)', 'HSBC Holdings plc (ADR)', 'Alphabet Inc', 'AbbVie Inc', 'BP plc (ADR)', 'AT&T Inc.', 'Toyota Motor Corp (ADR)', 'Apple Inc.', 'Exxon Corporation', 'Samsung Electronics Co Ltd', '3M Co', 'Pfizer, Inc. (OLD)', 'Siemens AG (ADR)', 'Abbott Laboratories', 'Chevron Corporation', 'BHP Group PLC', 'Boeing Co', 'Honeywell International Inc.', 'Bayer AG (ADR)', 'Novo Nordisk A/S (ADR)', 'Royal Dutch Shell Plc', 'Volkswagen AG (ADR)', 'Walmart Inc', 'Nestle SA', 'Alibaba Group Holding Ltd', 'General Electric Company', 'The Chase Manhattan Corp', 'Banco Santander SA (ADR)', 'Merck & Co., Inc.', 'Amgen, Inc.'])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4DJo1cS4mO4x",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 74
        },
        "outputId": "35c09982-b409-4c7d-d929-0f135049472e"
      },
      "source": [
        "# Char to index\n",
        "\n",
        "char2idx = {}\n",
        "char2idx[\"PADDING_TOKEN\"] = 0\n",
        "char2idx[\"NEWLINE_TOKEN\"] = 1\n",
        "\n",
        "# Token length for char CNN implementation\n",
        "cnn_len = 16\n",
        "\n",
        "for sentence in data['Headline'].values:\n",
        "    for char in list(sentence):\n",
        "        if char not in char2idx:\n",
        "            char2idx[char] = len(char2idx)\n",
        "\n",
        "print('Char vocabulary:')\n",
        "char2idx.keys()"
      ],
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Char vocabulary:\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "dict_keys(['PADDING_TOKEN', 'NEWLINE_TOKEN', 'A', 'p', 'l', 'e', ',', ' ', 'G', 'o', 'd', 'm', 'a', 'n', 'S', 'c', 'h', 's', 'T', 'U', 'O', 'C', 'r', 'i', 't', 'P', 'W', '-', 'J', 'B', 'u', 'y', 'V', 'g', 'x', 'M', 'v', 'b', 'k', 'D', 'H', 'R', 'I', '2', '0', 'E', 'F', 'j', 'N', 'f', 'w', \"'\", 'Y', '1', '8', '$', '5', '.', '7', '4', 'Q', '9', '\"', 'q', 'L', 'z', '6', '3', 'Z', '&', 'X', 'K', '%', '<', '>', '+', ';', '/', ':', '(', ')', '`', '!', '[', ']', '*', '#', '_'])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 6
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "WsNEyCNbmO45",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 36
        },
        "outputId": "a694dc7c-f521-4b0f-b984-e930a74e87ef"
      },
      "source": [
        "# Loading FASTTEXT english model bin\n",
        "\n",
        "word_embeddings_path = '/content/drive/My Drive/Colab Notebooks/NLP_forecasting-master/cc.en.300.bin'\n",
        "lang_model = FastText.load_fasttext_format(word_embeddings_path)\n",
        "\n",
        "embedding_size = len(lang_model['size'])\n",
        "print ('Embedding size:', embedding_size)"
      ],
      "execution_count": 7,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Embedding size: 300\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oDGkV8eymO5A",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 695
        },
        "outputId": "980acf79-2e80-4f88-81e9-36bc47ea58cf"
      },
      "source": [
        "# Lemmatization of text tokens\n",
        "\n",
        "def get_wordnet_pos(treebank_tag):\n",
        "\n",
        "    if treebank_tag.startswith('J'):\n",
        "        return {'pos': wordnet.ADJ}\n",
        "    elif treebank_tag.startswith('V'):\n",
        "        return {'pos': wordnet.VERB}\n",
        "    elif treebank_tag.startswith('N'):\n",
        "        return {'pos': wordnet.NOUN}\n",
        "    elif treebank_tag.startswith('R'):\n",
        "        return {'pos': wordnet.ADV}\n",
        "    else:\n",
        "        return {}\n",
        "\n",
        "headlines_processed = []\n",
        "\n",
        "for ind, row in data.iterrows():\n",
        "    sentence_clean = row['Headline'].replace('\"', '').replace(\"'\", '')\n",
        "    sentence_list = pos_tag(word_tokenize(sentence_clean))\n",
        "    sentence_parsed = [wordnet_lemmatizer.lemmatize(token.lower(), **get_wordnet_pos(pos)) for token, pos in sentence_list]\n",
        "    headlines_processed.append(sentence_parsed)\n",
        "\n",
        "data['Headline_proc'] = headlines_processed\n",
        "data.iloc[:5]"
      ],
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Release Date</th>\n",
              "      <th>Company</th>\n",
              "      <th>Headline</th>\n",
              "      <th>Topic</th>\n",
              "      <th>r_stock</th>\n",
              "      <th>r_index</th>\n",
              "      <th>MFI</th>\n",
              "      <th>ForPE</th>\n",
              "      <th>SIP</th>\n",
              "      <th>Headline_proc</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>02/21/19</td>\n",
              "      <td>Apple Inc.</td>\n",
              "      <td>Apple, Goldman Sachs Team Up On Credit Card Pa...</td>\n",
              "      <td>Strategic Combinations, Debt Ratings</td>\n",
              "      <td>-0.005639</td>\n",
              "      <td>-0.003526</td>\n",
              "      <td>67.327</td>\n",
              "      <td>14.652</td>\n",
              "      <td>0.846</td>\n",
              "      <td>[apple, ,, goldman, sachs, team, up, on, credi...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>02/15/19</td>\n",
              "      <td>Apple Inc.</td>\n",
              "      <td>Apple Buys Voice App Startup Pullstring - Axios</td>\n",
              "      <td>Mergers &amp; Acquisitions</td>\n",
              "      <td>-0.002225</td>\n",
              "      <td>0.010879</td>\n",
              "      <td>66.299</td>\n",
              "      <td>14.533</td>\n",
              "      <td>0.856</td>\n",
              "      <td>[apple, buy, voice, app, startup, pullstring, ...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>02/14/19</td>\n",
              "      <td>Apple Inc.</td>\n",
              "      <td>Charlie Munger discusses investing, banks, Chi...</td>\n",
              "      <td>Regulatory / Company Investigation</td>\n",
              "      <td>0.003643</td>\n",
              "      <td>-0.002652</td>\n",
              "      <td>69.938</td>\n",
              "      <td>14.480</td>\n",
              "      <td>0.856</td>\n",
              "      <td>[charlie, munger, discuss, investing, ,, bank,...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>02/11/19</td>\n",
              "      <td>Apple Inc.</td>\n",
              "      <td>Apple Says Health Records On Iphone Will Be Av...</td>\n",
              "      <td>General Products, Expansion</td>\n",
              "      <td>-0.005751</td>\n",
              "      <td>0.000709</td>\n",
              "      <td>64.911</td>\n",
              "      <td>14.499</td>\n",
              "      <td>0.856</td>\n",
              "      <td>[apple, say, health, record, on, iphone, will,...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>01/24/19</td>\n",
              "      <td>Apple Inc.</td>\n",
              "      <td>Apple Dismissed More Than 200 Employees From P...</td>\n",
              "      <td>Layoffs</td>\n",
              "      <td>-0.007926</td>\n",
              "      <td>0.001376</td>\n",
              "      <td>54.440</td>\n",
              "      <td>12.916</td>\n",
              "      <td>0.982</td>\n",
              "      <td>[apple, dismiss, more, than, 200, employee, fr...</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "  Release Date  ...                                      Headline_proc\n",
              "0     02/21/19  ...  [apple, ,, goldman, sachs, team, up, on, credi...\n",
              "1     02/15/19  ...  [apple, buy, voice, app, startup, pullstring, ...\n",
              "2     02/14/19  ...  [charlie, munger, discuss, investing, ,, bank,...\n",
              "3     02/11/19  ...  [apple, say, health, record, on, iphone, will,...\n",
              "4     01/24/19  ...  [apple, dismiss, more, than, 200, employee, fr...\n",
              "\n",
              "[5 rows x 10 columns]"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "lTeYC9skKAYq",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 73
        },
        "outputId": "6e7ced2d-ab99-4aad-ebf8-e4d8e7e5f35c"
      },
      "source": [
        "import nltk\n",
        "nltk.download('wordnet')"
      ],
      "execution_count": 13,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/wordnet.zip.\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 13
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mAgf9-XEmO5F",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 185
        },
        "outputId": "77af7006-f142-4239-967a-3010a81109bc"
      },
      "source": [
        "# Word to embedding\n",
        "\n",
        "word2idx = {}\n",
        "word_embeddings = []\n",
        "\n",
        "# Initialization of embeddings for pads and OOV\n",
        "word2idx[\"PADDING_TOKEN\"] = len(word2idx)\n",
        "word_embeddings.append(np.zeros(embedding_size))\n",
        "\n",
        "word2idx[\"UNKNOWN_TOKEN\"] = len(word2idx)\n",
        "word_embeddings.append(np.random.uniform(-0.25, 0.25, embedding_size))\n",
        "\n",
        "# Получаем вектора токенов из словаря\n",
        "for sentence_list in data['Headline_proc'].values:\n",
        "    for token in sentence_list:\n",
        "        if token not in word2idx:\n",
        "            try:\n",
        "                word_embeddings.append(lang_model[token])\n",
        "                word2idx[token] = len(word2idx)\n",
        "            except:\n",
        "                pass\n",
        "\n",
        "word_embeddings = np.array(word_embeddings, dtype='float32')\n",
        "\n",
        "print ('Word embeddings:')\n",
        "word_embeddings[:4]"
      ],
      "execution_count": 15,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Word embeddings:\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[ 0.        ,  0.        ,  0.        , ...,  0.        ,\n",
              "         0.        ,  0.        ],\n",
              "       [-0.2002255 ,  0.15732543,  0.19950478, ...,  0.12299345,\n",
              "        -0.11181178,  0.12901825],\n",
              "       [ 0.01440609, -0.01854071,  0.07781687, ...,  0.1554127 ,\n",
              "         0.09066156, -0.07712348],\n",
              "       [ 0.12502378, -0.10790165,  0.02450176, ...,  0.23047423,\n",
              "        -0.06955914, -0.0214496 ]], dtype=float32)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 15
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "5Pnih7SrmO5O",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 54
        },
        "outputId": "c9f02d18-6133-494a-c881-1fad436cff1b"
      },
      "source": [
        "# Train-dev-test split\n",
        "\n",
        "train_dataset = data.sample(frac=0.95, random_state=1234)\n",
        "test_dataset = data.drop(train_dataset.index)\n",
        "\n",
        "print ('Train size:', len(train_dataset.index))\n",
        "print ('Test size:', len(test_dataset.index))"
      ],
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train size: 9235\n",
            "Test size: 486\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "TrDYEBn6mO5U",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 54
        },
        "outputId": "349e60c3-e5ec-465e-e7bf-433ddb05aac8"
      },
      "source": [
        "# Converting unstructured data to matrices\n",
        "\n",
        "def create_matrices(df, df_type):   \n",
        "    total_tokens = 0\n",
        "    unknown_tokens = 0\n",
        "    dataset = []\n",
        "    \n",
        "    for ind, row in df.iterrows():\n",
        "        \n",
        "        # Get company code\n",
        "        company_index = Company2idx[row['Company']]\n",
        "        \n",
        "        # Get word embedding indices\n",
        "        word_indices = [word2idx['PADDING_TOKEN']]\n",
        "        \n",
        "        for token in row['Headline_proc']:\n",
        "            if (token != newline_token):\n",
        "                total_tokens += 1\n",
        "                \n",
        "                if token in word2idx:\n",
        "                    word_idx = word2idx[token]\n",
        "                else:\n",
        "                    word_idx = word2idx[\"UNKNOWN_TOKEN\"]\n",
        "                    unknown_tokens += 1\n",
        "            else:\n",
        "                word_idx = word2idx['PADDING_TOKEN']\n",
        "            \n",
        "            word_indices.append(word_idx)\n",
        "        \n",
        "        word_indices.append(word2idx['PADDING_TOKEN'])\n",
        "        \n",
        "        # Get char indices\n",
        "        char_codes = [[char2idx[\"PADDING_TOKEN\"]] * cnn_len]\n",
        "        \n",
        "        for token in word_tokenize(row['Headline'].replace('\"', '').replace(\"'\", '')):\n",
        "            if (token != newline_token):\n",
        "                token_trunc = token[:cnn_len]\n",
        "                token_chars = [char2idx[char] for char in list(token_trunc)]\n",
        "                token_chars = token_chars + [char2idx[\"PADDING_TOKEN\"]] * (cnn_len - len(token_trunc))\n",
        "            else:\n",
        "                token_chars = [char2idx[\"PADDING_TOKEN\"]] * cnn_len\n",
        "                \n",
        "            char_codes.append(token_chars)\n",
        "            \n",
        "        char_codes.append([char2idx[\"PADDING_TOKEN\"]] * cnn_len)        \n",
        "        \n",
        "        # Get true label\n",
        "        label = row['r_stock']\n",
        "        \n",
        "        # Get numerical features\n",
        "        r_index = row['r_index']\n",
        "        MFI = row['MFI']\n",
        "        ForPE = row['ForPE']\n",
        "        SIP = row['SIP']        \n",
        "\n",
        "        # Save sample\n",
        "        dataset.append([company_index, np.array(word_indices), np.array(char_codes), r_index, MFI, ForPE, SIP, label])\n",
        "        \n",
        "    unknown_percent = 0.0\n",
        "    if total_tokens != 0:\n",
        "        unknown_percent = 100 * float(unknown_tokens) / total_tokens\n",
        "    print(df_type + \" data: {} tokens, {} unknown, {:.3}%\".format(total_tokens, unknown_tokens, unknown_percent))\n",
        "    \n",
        "    return np.array(dataset)\n",
        "\n",
        "train_data = create_matrices(train_dataset, 'Train')\n",
        "test_data = create_matrices(test_dataset, 'Test')"
      ],
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train data: 133544 tokens, 852 unknown, 0.638%\n",
            "Test data: 7004 tokens, 53 unknown, 0.757%\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fFp_SrCmmO5a",
        "colab_type": "text"
      },
      "source": [
        "## Full model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "i09XTBaPmO5e",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "776d2770-0f96-4b8c-9d73-302958ab6ee3"
      },
      "source": [
        "dim_HIDDEN = 16\n",
        "CNN_FILTERS = 64\n",
        "dim_company = 4\n",
        "dim_CHAR = 32\n",
        "CNN_WIN = 5\n",
        "\n",
        "# Input layers and embeddings\n",
        "company_input = Input(dtype='int32', shape=(1,), name='company_input')\n",
        "company_embedding_layer = Embedding(input_dim=len(Company2idx), output_dim=dim_company,\n",
        "                                    trainable=True, name='company_embeddings')\n",
        "company = company_embedding_layer(company_input)\n",
        "company = Lambda(lambda x: K.squeeze(x, 1), name='company')(company)\n",
        "\n",
        "r_index = Input(dtype='float32', shape=(1,), name='r_index')\n",
        "MFI = Input(dtype='float32', shape=(1,), name='MFI')\n",
        "ForPE = Input(dtype='float32', shape=(1,), name='ForPE')\n",
        "SIP = Input(dtype='float32', shape=(1,), name='SIP')\n",
        "\n",
        "token_input = Input(dtype='int32', shape=(None,), name='token_input')\n",
        "token_embedding_layer = Embedding(input_dim=word_embeddings.shape[0], \n",
        "                                   output_dim=word_embeddings.shape[1],\n",
        "                                   weights=[word_embeddings], trainable=False, \n",
        "                                   name='token_embeddings')\n",
        "tokens = token_embedding_layer(token_input)\n",
        "\n",
        "char_input = Input(dtype='int32', shape=(None, cnn_len), name='char_input')\n",
        "char_embedding_layer = Embedding(input_dim=len(char2idx), output_dim=dim_CHAR, name='char_embedding_layer')\n",
        "char_embeddings = char_embedding_layer(char_input)\n",
        "\n",
        "# Implementation of char CNN\n",
        "char_cnn = TimeDistributed(Conv1D(filters=CNN_FILTERS, kernel_size=CNN_WIN), name='char_cnn')(char_embeddings)\n",
        "char_activation = TimeDistributed(PReLU(), name='char_activation')(char_cnn)\n",
        "char_pooling = TimeDistributed(GlobalMaxPooling1D(), name='char_pooling')(char_activation)\n",
        "char_highway = TimeDistributed(Highway(), name='char_highway')(char_pooling)\n",
        "chars = TimeDistributed(Dropout(0.30), name = \"chars\")(char_highway)\n",
        "\n",
        "merged_embeddings = concatenate([tokens, chars], name='merged_embeddings')\n",
        "\n",
        "# Implementation of BLSTM\n",
        "blstm, forward_h, forward_c, backward_h, backward_c = Bidirectional(\n",
        "    LSTM(dim_HIDDEN, return_sequences=True, return_state=True, implementation=2), name='blstm')(merged_embeddings)\n",
        "\n",
        "# Implementation of attention\n",
        "state_h_concat = concatenate([forward_h, backward_h], name = 'state_h_concat')\n",
        "state_h = Lambda(lambda x: tf.expand_dims(x, axis = 1), name = 'state_h')(state_h_concat)\n",
        "\n",
        "attention_W1 = TimeDistributed(Dense(dim_HIDDEN), name = 'attention_W1')(blstm)\n",
        "attention_W2 = TimeDistributed(Dense(dim_HIDDEN), name = 'attention_W2')(state_h)\n",
        "attention_W = add([attention_W1, attention_W2], name = 'attention_W')\n",
        "\n",
        "attention_scores = Lambda(lambda x: tf.nn.tanh(x), name = 'attention_scores')(attention_W)\n",
        "attention_V = TimeDistributed(Dense(1), name = 'attention_V')(attention_scores)\n",
        "attention_weights = Lambda(lambda x: tf.nn.softmax(x, axis = 1), name = 'attention_weights')(attention_V)\n",
        "\n",
        "# Weighting context embeddings by attention\n",
        "context_vector = multiply([attention_weights, blstm], name = \"context_vector\")\n",
        "context_agg = Lambda(lambda x: tf.reduce_sum(x, axis=1), name = \"context_agg\")(context_vector)\n",
        "context = Dense(dim_HIDDEN, name='context')(context_agg)\n",
        "\n",
        "# Combining vector of features\n",
        "features = concatenate([company, context, r_index, MFI, ForPE, SIP], name = 'features')\n",
        "features_dropout = Dropout(0.30, name = \"features_dropout\")(features)\n",
        "\n",
        "# Output regression\n",
        "dense = Dense(dim_HIDDEN, name='dense')(features_dropout)\n",
        "activation = PReLU(name='activation')(dense)\n",
        "result = Dense(1, name='result')(activation)\n",
        "\n",
        "# Compiling model\n",
        "model = Model(inputs=[company_input, token_input, char_input, r_index, MFI, ForPE, SIP], outputs=result)\n",
        "model.compile(loss='mse', optimizer=Adam())\n",
        "model.summary()"
      ],
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:541: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:4432: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:190: The name tf.get_default_session is deprecated. Please use tf.compat.v1.get_default_session instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:207: The name tf.global_variables is deprecated. Please use tf.compat.v1.global_variables instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:216: The name tf.is_variable_initialized is deprecated. Please use tf.compat.v1.is_variable_initialized instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:223: The name tf.variables_initializer is deprecated. Please use tf.compat.v1.variables_initializer instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:66: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:148: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:3733: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.6/dist-packages/keras/legacy/layers.py:200: UserWarning: The `Highway` layer is deprecated and will be removed after 06/2017.\n",
            "  warnings.warn('The `Highway` layer is deprecated '\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/optimizers.py:793: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.\n",
            "\n",
            "Model: \"model_1\"\n",
            "__________________________________________________________________________________________________\n",
            "Layer (type)                    Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            "char_input (InputLayer)         (None, None, 16)     0                                            \n",
            "__________________________________________________________________________________________________\n",
            "char_embedding_layer (Embedding (None, None, 16, 32) 2816        char_input[0][0]                 \n",
            "__________________________________________________________________________________________________\n",
            "char_cnn (TimeDistributed)      (None, None, 12, 64) 10304       char_embedding_layer[0][0]       \n",
            "__________________________________________________________________________________________________\n",
            "char_activation (TimeDistribute (None, None, 12, 64) 768         char_cnn[0][0]                   \n",
            "__________________________________________________________________________________________________\n",
            "char_pooling (TimeDistributed)  (None, None, 64)     0           char_activation[0][0]            \n",
            "__________________________________________________________________________________________________\n",
            "token_input (InputLayer)        (None, None)         0                                            \n",
            "__________________________________________________________________________________________________\n",
            "char_highway (TimeDistributed)  (None, None, 64)     8320        char_pooling[0][0]               \n",
            "__________________________________________________________________________________________________\n",
            "token_embeddings (Embedding)    (None, None, 300)    3921300     token_input[0][0]                \n",
            "__________________________________________________________________________________________________\n",
            "chars (TimeDistributed)         (None, None, 64)     0           char_highway[0][0]               \n",
            "__________________________________________________________________________________________________\n",
            "merged_embeddings (Concatenate) (None, None, 364)    0           token_embeddings[0][0]           \n",
            "                                                                 chars[0][0]                      \n",
            "__________________________________________________________________________________________________\n",
            "blstm (Bidirectional)           [(None, None, 32), ( 48768       merged_embeddings[0][0]          \n",
            "__________________________________________________________________________________________________\n",
            "state_h_concat (Concatenate)    (None, 32)           0           blstm[0][1]                      \n",
            "                                                                 blstm[0][3]                      \n",
            "__________________________________________________________________________________________________\n",
            "state_h (Lambda)                (None, 1, 32)        0           state_h_concat[0][0]             \n",
            "__________________________________________________________________________________________________\n",
            "attention_W1 (TimeDistributed)  (None, None, 16)     528         blstm[0][0]                      \n",
            "__________________________________________________________________________________________________\n",
            "attention_W2 (TimeDistributed)  (None, 1, 16)        528         state_h[0][0]                    \n",
            "__________________________________________________________________________________________________\n",
            "attention_W (Add)               (None, None, 16)     0           attention_W1[0][0]               \n",
            "                                                                 attention_W2[0][0]               \n",
            "__________________________________________________________________________________________________\n",
            "attention_scores (Lambda)       (None, None, 16)     0           attention_W[0][0]                \n",
            "__________________________________________________________________________________________________\n",
            "attention_V (TimeDistributed)   (None, None, 1)      17          attention_scores[0][0]           \n",
            "__________________________________________________________________________________________________\n",
            "attention_weights (Lambda)      (None, None, 1)      0           attention_V[0][0]                \n",
            "__________________________________________________________________________________________________\n",
            "company_input (InputLayer)      (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "context_vector (Multiply)       (None, None, 32)     0           attention_weights[0][0]          \n",
            "                                                                 blstm[0][0]                      \n",
            "__________________________________________________________________________________________________\n",
            "company_embeddings (Embedding)  (None, 1, 4)         176         company_input[0][0]              \n",
            "__________________________________________________________________________________________________\n",
            "context_agg (Lambda)            (None, 32)           0           context_vector[0][0]             \n",
            "__________________________________________________________________________________________________\n",
            "company (Lambda)                (None, 4)            0           company_embeddings[0][0]         \n",
            "__________________________________________________________________________________________________\n",
            "context (Dense)                 (None, 16)           528         context_agg[0][0]                \n",
            "__________________________________________________________________________________________________\n",
            "r_index (InputLayer)            (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "MFI (InputLayer)                (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "ForPE (InputLayer)              (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "SIP (InputLayer)                (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "features (Concatenate)          (None, 24)           0           company[0][0]                    \n",
            "                                                                 context[0][0]                    \n",
            "                                                                 r_index[0][0]                    \n",
            "                                                                 MFI[0][0]                        \n",
            "                                                                 ForPE[0][0]                      \n",
            "                                                                 SIP[0][0]                        \n",
            "__________________________________________________________________________________________________\n",
            "features_dropout (Dropout)      (None, 24)           0           features[0][0]                   \n",
            "__________________________________________________________________________________________________\n",
            "dense (Dense)                   (None, 16)           400         features_dropout[0][0]           \n",
            "__________________________________________________________________________________________________\n",
            "activation (PReLU)              (None, 16)           16          dense[0][0]                      \n",
            "__________________________________________________________________________________________________\n",
            "result (Dense)                  (None, 1)            17          activation[0][0]                 \n",
            "==================================================================================================\n",
            "Total params: 3,994,486\n",
            "Trainable params: 73,186\n",
            "Non-trainable params: 3,921,300\n",
            "__________________________________________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "IQ2iAKenmO5l",
        "colab_type": "code",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "outputId": "f4111b08-b974-46bd-c7a0-4101e538a6ab"
      },
      "source": [
        "number_of_epochs = 5\n",
        "lr_decay = 0.5\n",
        "K.set_value(model.optimizer.lr, 0.01)\n",
        "random.seed(1234)\n",
        "print(\"%d epochs\" % number_of_epochs)\n",
        "print()\n",
        "\n",
        "def iterate_minibatches(dataset):   \n",
        "    for sentence in dataset:\n",
        "        companies, tokens, chars, r_index, MFI, ForPE, SIP, label = sentence     \n",
        "        yield (np.asarray([companies]), np.asarray([tokens]), np.asarray([chars]), np.asarray([r_index]), \n",
        "               np.asarray([MFI]), np.asarray([ForPE]), np.asarray([SIP]), np.asarray([label]))\n",
        "\n",
        "def tag_dataset(dataset):\n",
        "    predicted_returns = []\n",
        "    true_returns = []\n",
        "    for company, tokens, chars, r_index, MFI, ForPE, SIP, label in dataset:\n",
        "        pred = model.predict_on_batch([np.asarray([company]), np.asarray([tokens]), np.asarray([chars]), np.asarray([r_index]),\n",
        "                                       np.asarray([MFI]), np.asarray([ForPE]), np.asarray([SIP])])[0]\n",
        "        predicted_returns.append(pred)\n",
        "        true_returns.append(label)\n",
        "    return predicted_returns, true_returns\n",
        "\n",
        "def compute_rmse(y_pred, y_true):\n",
        "    return np.sqrt(mean_squared_error(y_true = y_true, y_pred = y_pred))\n",
        "\n",
        "print(\"%d train sentences\" % len(train_data))\n",
        "print(\"%d test sentences\" % len(test_data))\n",
        "\n",
        "for epoch in range(number_of_epochs):    \n",
        "    print()\n",
        "    print(\"--------- Epoch %d -----------\" % epoch)\n",
        "    random.shuffle(train_data)\n",
        "    \n",
        "    start_time = time.time()    \n",
        "    for batch in iterate_minibatches(train_data):\n",
        "        companies, tokens, chars, r_index, MFI, ForPE, SIP, label = batch       \n",
        "        model.train_on_batch([companies, tokens, chars, r_index, MFI, ForPE, SIP], label)   \n",
        "    print(\"%.2f sec for training\" % (time.time() - start_time))\n",
        "    print()\n",
        "    \n",
        "    #Train Dataset       \n",
        "    start_time = time.time()  \n",
        "    print(\"================================== Train Data ==================================\")    \n",
        "    predicted, correct = tag_dataset(train_data)  \n",
        "    RMSE = compute_rmse(predicted, correct)\n",
        "    print(\"RMSE = \", RMSE)\n",
        "\n",
        "    #Test Dataset \n",
        "    print(\"================================== Test Data: ==================================\")\n",
        "    predicted, correct = tag_dataset(test_data)  \n",
        "    RMSE = compute_rmse(predicted, correct)\n",
        "    print(\"RMSE = \", RMSE)\n",
        "    print()\n",
        "    print(\"%.2f sec for evaluation\" % (time.time() - start_time))\n",
        "    \n",
        "    current_lr = K.get_value(model.optimizer.lr)\n",
        "    K.set_value(model.optimizer.lr, current_lr * (1.0 - lr_decay))"
      ],
      "execution_count": 19,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "5 epochs\n",
            "\n",
            "9235 train sentences\n",
            "486 test sentences\n",
            "\n",
            "--------- Epoch 0 -----------\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow_core/python/ops/math_grad.py:1424: where (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Use tf.where in 2.0, which has the same broadcast rule as np.where\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1033: The name tf.assign_add is deprecated. Please use tf.compat.v1.assign_add instead.\n",
            "\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/keras/backend/tensorflow_backend.py:1020: The name tf.assign is deprecated. Please use tf.compat.v1.assign instead.\n",
            "\n",
            "460.19 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.4031390264754185\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.43712723195815667\n",
            "\n",
            "96.83 sec for evaluation\n",
            "\n",
            "--------- Epoch 1 -----------\n",
            "448.02 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.2871028881540701\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.27889998968720203\n",
            "\n",
            "96.94 sec for evaluation\n",
            "\n",
            "--------- Epoch 2 -----------\n",
            "442.54 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.018154280161250543\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.018985967043018158\n",
            "\n",
            "95.31 sec for evaluation\n",
            "\n",
            "--------- Epoch 3 -----------\n",
            "433.64 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.017066556744985594\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.018777274529368275\n",
            "\n",
            "93.60 sec for evaluation\n",
            "\n",
            "--------- Epoch 4 -----------\n",
            "420.75 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.016041168537292413\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.01911653890115487\n",
            "\n",
            "92.07 sec for evaluation\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oIWTMP5smO5r",
        "colab_type": "text"
      },
      "source": [
        "## Baseline 2"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "F43nvf96mO5s",
        "colab_type": "code",
        "colab": {},
        "outputId": "77a96092-61f6-4692-b817-88e4338f98e7"
      },
      "source": [
        "dim_HIDDEN = 16\n",
        "dim_company = 4\n",
        "\n",
        "# Input layers and embeddings\n",
        "company_input = Input(dtype='int32', shape=(1,), name='company_input')\n",
        "company_embedding_layer = Embedding(input_dim=len(Company2idx), output_dim=dim_company,\n",
        "                                    trainable=True, name='company_embeddings')\n",
        "company = company_embedding_layer(company_input)\n",
        "company = Lambda(lambda x: K.squeeze(x, 1), name='company')(company)\n",
        "\n",
        "noise_vector = Input(dtype='float32', shape=(dim_HIDDEN,), name='noise_vector')\n",
        "\n",
        "r_index = Input(dtype='float32', shape=(1,), name='r_index')\n",
        "MFI = Input(dtype='float32', shape=(1,), name='MFI')\n",
        "ForPE = Input(dtype='float32', shape=(1,), name='ForPE')\n",
        "SIP = Input(dtype='float32', shape=(1,), name='SIP')\n",
        "\n",
        "# Combining vector of features\n",
        "features = concatenate([company, noise_vector, r_index, MFI, ForPE, SIP], name = 'features')\n",
        "features_dropout = Dropout(0.30, name = \"features_dropout\")(features)\n",
        "\n",
        "# Output regression\n",
        "dense = Dense(dim_HIDDEN, name='dense')(features_dropout)\n",
        "activation = PReLU(name='activation')(dense)\n",
        "result = Dense(1, name='result')(activation)\n",
        "\n",
        "# Compiling model\n",
        "model = Model(inputs=[company_input, noise_vector, r_index, MFI, ForPE, SIP], outputs=result)\n",
        "model.compile(loss='mse', optimizer=Adam())\n",
        "model.summary()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "__________________________________________________________________________________________________\n",
            "Layer (type)                    Output Shape         Param #     Connected to                     \n",
            "==================================================================================================\n",
            "company_input (InputLayer)      (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "company_embeddings (Embedding)  (None, 1, 4)         176         company_input[0][0]              \n",
            "__________________________________________________________________________________________________\n",
            "company (Lambda)                (None, 4)            0           company_embeddings[0][0]         \n",
            "__________________________________________________________________________________________________\n",
            "noise_vector (InputLayer)       (None, 16)           0                                            \n",
            "__________________________________________________________________________________________________\n",
            "r_index (InputLayer)            (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "MFI (InputLayer)                (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "ForPE (InputLayer)              (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "SIP (InputLayer)                (None, 1)            0                                            \n",
            "__________________________________________________________________________________________________\n",
            "features (Concatenate)          (None, 24)           0           company[0][0]                    \n",
            "                                                                 noise_vector[0][0]               \n",
            "                                                                 r_index[0][0]                    \n",
            "                                                                 MFI[0][0]                        \n",
            "                                                                 ForPE[0][0]                      \n",
            "                                                                 SIP[0][0]                        \n",
            "__________________________________________________________________________________________________\n",
            "features_dropout (Dropout)      (None, 24)           0           features[0][0]                   \n",
            "__________________________________________________________________________________________________\n",
            "dense (Dense)                   (None, 16)           400         features_dropout[0][0]           \n",
            "__________________________________________________________________________________________________\n",
            "activation (PReLU)              (None, 16)           16          dense[0][0]                      \n",
            "__________________________________________________________________________________________________\n",
            "result (Dense)                  (None, 1)            17          activation[0][0]                 \n",
            "==================================================================================================\n",
            "Total params: 609\n",
            "Trainable params: 609\n",
            "Non-trainable params: 0\n",
            "__________________________________________________________________________________________________\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "scrolled": false,
        "id": "CBJNROa0mO5z",
        "colab_type": "code",
        "colab": {},
        "outputId": "873758e3-52f3-47ed-974a-5916d2a79967"
      },
      "source": [
        "number_of_epochs = 5\n",
        "lr_decay = 0.5\n",
        "K.set_value(model.optimizer.lr, 0.01)\n",
        "random.seed(1234)\n",
        "print(\"%d epochs\" % number_of_epochs)\n",
        "print()\n",
        "\n",
        "random_train = np.random.normal(size=(len(train_dataset.index), dim_HIDDEN))\n",
        "random_test = np.random.normal(size=(len(test_dataset.index), dim_HIDDEN))\n",
        "\n",
        "def iterate_minibatches(dataset, random_vecs):\n",
        "    i = -1\n",
        "    for sentence in dataset:\n",
        "        i += 1\n",
        "        companies, _, _, r_index, MFI, ForPE, SIP, label = sentence     \n",
        "        yield (np.asarray([companies]), np.asarray([r_index]), \n",
        "               np.asarray([MFI]), np.asarray([ForPE]), np.asarray([SIP]), np.asarray([label]), np.asarray([random_vecs[i]]))\n",
        "\n",
        "def tag_dataset(dataset, random_vecs):\n",
        "    predicted_returns = []\n",
        "    true_returns = []\n",
        "    i = -1\n",
        "    for company, _, _, r_index, MFI, ForPE, SIP, label in dataset:\n",
        "        i += 1\n",
        "        pred = model.predict_on_batch([np.asarray([company]), np.asarray([random_vecs[i]]), np.asarray([r_index]),\n",
        "                                       np.asarray([MFI]), np.asarray([ForPE]), np.asarray([SIP])])[0]\n",
        "        predicted_returns.append(pred)\n",
        "        true_returns.append(label)\n",
        "    return predicted_returns, true_returns\n",
        "\n",
        "def compute_rmse(y_pred, y_true):\n",
        "    return np.sqrt(mean_squared_error(y_true = y_true, y_pred = y_pred))\n",
        "\n",
        "print(\"%d train observations\" % len(train_data))\n",
        "print(\"%d test observations\" % len(test_data))\n",
        "\n",
        "for epoch in range(number_of_epochs):    \n",
        "    print()\n",
        "    print(\"--------- Epoch %d -----------\" % epoch)\n",
        "    random.shuffle(train_data)\n",
        "    \n",
        "    start_time = time.time()    \n",
        "    for batch in iterate_minibatches(train_data, random_train):\n",
        "        companies, r_index, MFI, ForPE, SIP, label, random_vec = batch       \n",
        "        model.train_on_batch([companies, random_vec, r_index, MFI, ForPE, SIP], label)   \n",
        "    print(\"%.2f sec for training\" % (time.time() - start_time))\n",
        "    print()\n",
        "    \n",
        "    #Train Dataset       \n",
        "    start_time = time.time()  \n",
        "    print(\"================================== Train Data ==================================\")    \n",
        "    predicted, correct = tag_dataset(train_data, random_train)  \n",
        "    RMSE = compute_rmse(predicted, correct)\n",
        "    print(\"RMSE = \", RMSE)\n",
        "\n",
        "    #Test Dataset \n",
        "    print(\"================================== Test Data: ==================================\")\n",
        "    predicted, correct = tag_dataset(test_data, random_test)  \n",
        "    RMSE = compute_rmse(predicted, correct)\n",
        "    print(\"RMSE = \", RMSE)\n",
        "    print()\n",
        "    \n",
        "    print(\"%.2f sec for evaluation\" % (time.time() - start_time))\n",
        "    \n",
        "    current_lr = K.get_value(model.optimizer.lr)\n",
        "    K.set_value(model.optimizer.lr, current_lr * (1.0 - lr_decay))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "5 epochs\n",
            "\n",
            "9235 train observations\n",
            "486 test observations\n",
            "\n",
            "--------- Epoch 0 -----------\n",
            "12.09 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.0020940749952854418\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.024600661473915614\n",
            "\n",
            "7.61 sec for evaluation\n",
            "\n",
            "--------- Epoch 1 -----------\n",
            "8.58 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.00047133445018445136\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.025216087776783278\n",
            "\n",
            "6.58 sec for evaluation\n",
            "\n",
            "--------- Epoch 2 -----------\n",
            "8.71 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.0002665597918634258\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.024591328508430592\n",
            "\n",
            "6.56 sec for evaluation\n",
            "\n",
            "--------- Epoch 3 -----------\n",
            "8.83 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  6.831015763747323e-05\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.024664564952236127\n",
            "\n",
            "6.64 sec for evaluation\n",
            "\n",
            "--------- Epoch 4 -----------\n",
            "8.67 sec for training\n",
            "\n",
            "================================== Train Data ==================================\n",
            "RMSE =  0.002510423378195602\n",
            "================================== Test Data: ==================================\n",
            "RMSE =  0.02362537984107042\n",
            "\n",
            "6.40 sec for evaluation\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "U1oSQdpnmO56",
        "colab_type": "text"
      },
      "source": [
        "## Baseline 3"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U29lLb5TmO58",
        "colab_type": "code",
        "colab": {}
      },
      "source": [
        "def identity_tokenizer(text):\n",
        "    return text\n",
        "\n",
        "vectorizer = TfidfVectorizer(lowercase=False, input='content', stop_words='english',\n",
        "                             ngram_range=(1,1), tokenizer=identity_tokenizer)\n",
        "trSVD = TruncatedSVD(n_components=500, n_iter=100)\n",
        "\n",
        "tv = vectorizer.fit_transform(list(data['Headline_proc'].values))\n",
        "tr_tv = trSVD.fit_transform(tv)\n",
        "\n",
        "data['co_dummy'] = pd.get_dummies(data['Company']).astype(float).values.tolist()\n",
        "data['tv'] = tr_tv.tolist()"
      ],
      "execution_count": 0,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nigaJuAVmO6C",
        "colab_type": "code",
        "colab": {},
        "outputId": "8520b127-7587-407d-f497-1d5eba1c1aad"
      },
      "source": [
        "# Train-dev-test split\n",
        "\n",
        "train_dataset_v = data.sample(frac=0.95, random_state=1234)\n",
        "test_dataset_v = data.drop(train_dataset_v.index)\n",
        "\n",
        "print ('Train size:', len(train_dataset_v.index))\n",
        "print ('Test size:', len(test_dataset_v.index))"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train size: 9235\n",
            "Test size: 486\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BBwMmOlEmO6J",
        "colab_type": "code",
        "colab": {},
        "outputId": "55034757-6688-4780-f96c-be7299517f2a"
      },
      "source": [
        "svr_model = SVR()\n",
        "\n",
        "X = ((train_dataset_v[['r_index', 'MFI', 'ForPE', 'SIP']]\n",
        "     ).join(train_dataset_v['tv'].apply(pd.Series).add_prefix('tv_'))\n",
        "    ).join(train_dataset_v['co_dummy'].apply(pd.Series).add_prefix('co_'))\n",
        "y = train_dataset_v['r_stock']\n",
        "\n",
        "svr_model.fit(X, y)\n",
        "\n",
        "y_train = svr_model.predict(X)\n",
        "\n",
        "print(\"================================== Train Data: ==================================\")\n",
        "RMSE = compute_rmse(y_train, y)\n",
        "print(\"RMSE = \", RMSE)\n",
        "print()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "================================== Train Data: ==================================\n",
            "RMSE =  0.027232685386578026\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Fk1F-zbhmO6S",
        "colab_type": "code",
        "colab": {},
        "outputId": "f4f1354a-5a85-44b8-c4a7-0b0d9897137d"
      },
      "source": [
        "X_test = ((test_dataset_v[['r_index', 'MFI', 'ForPE', 'SIP']]\n",
        "     ).join(test_dataset_v['tv'].apply(pd.Series).add_prefix('tv_'))\n",
        "    ).join(test_dataset_v['co_dummy'].apply(pd.Series).add_prefix('co_'))\n",
        "y_true = test_dataset_v['r_stock']\n",
        "\n",
        "y_test = svr_model.predict(X_test)\n",
        "\n",
        "print(\"================================== Test Data: ==================================\")\n",
        "RMSE = compute_rmse(y_test, y_true)\n",
        "print(\"RMSE = \", RMSE)\n",
        "print()"
      ],
      "execution_count": 0,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "================================== Test Data: ==================================\n",
            "RMSE =  0.029775049558993457\n",
            "\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zg_FUwBSmO6Z",
        "colab_type": "text"
      },
      "source": [
        " "
      ]
    }
  ]
}