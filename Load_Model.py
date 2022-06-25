# # ********************************************************************
import pickle
import itertools
import random
from collections import Counter
import joblib
import keras
import pandas as pd
import numpy as np
import re
from keras import Sequential, optimizers
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Bidirectional, LSTM
from keras.utils.np_utils import to_categorical
from nltk import word_tokenize
from nltk.corpus import stopwords
from parsivar import Normalizer
# import emoji
import emojies
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

#%%

def clean(text):
    text = emojies.replace(text)
    text = re.sub("@[A-Za-z0-9]+", '', text)
    text = re.sub(r'http\S+', '', text)

    #### Convert Finiglish To Persian in Package Parsivar
    my_normilize = Normalizer(pinglish_conversion_needed=True)
    text = my_normilize.normalize(text)
    text_token = word_tokenize(text)
    tokens_without_sw = [word for word in text_token if not word in stopwords.words('persion') ]
    text = " ".join(tokens_without_sw)
    return text
Data_f = pd.read_excel('8tag.xlsx', index_col=False)
Data_f['clean'] = Data_f['caption'].apply(lambda x:clean(x))
#%%
with open(r'C:\Users\......\PycharmProjects\Sentiment_Analysis\CNN_BiLSTM_6_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
encode_Data_Trian = tokenizer.texts_to_sequences(Data_f['clean'])

Data_Trian_p = pad_sequences(encode_Data_Trian,maxlen=100,padding='post')

# Load model
filename = r'CNN_LSTM_2_branch.h5'
model = keras.models.load_model(filename)


#%%
list_a = []
max_len = 100
for item in Data_Trian_p:
    item = item.tolist()
    item = [item]
    output = model.predict(item)
    list_a.append(output)
print(list_a)

result_df = pd.DataFrame()
result_df['text'] = Data_f['caption'].copy()
result_df['my_model'] = list_a
result_df.to_excel('result1_df.xlsx', index=False)
#%%
abs_df = pd.read_excel('result1_df.xlsx')
list_b = abs_df['my_model']
list_c = list(map(lambda x: list(x[2:-2].split(" ")), list_b))
list_c = list(map(lambda x: list(filter(None, x)), list_c))

list_d = list(map(lambda x: [re.sub('\n','', i) for i in x], list_c))

list_e = list(map(lambda x: [float(i) for i in x], list_d))
list_f = list(map(lambda x: x.index(max(x)), list_e))
result_df['split_my_model'] = list_f
result_df.to_excel('result1_df.xlsx', index=False)

