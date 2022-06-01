import pandas as pd
import numpy as np

# text preprocessing
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

# plots and metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# feature extraction / vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# classifiers
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import pickle

import os

path = os.getcwd()
csv_files0 = os.path.join(path, "data_test.csv")
csv_files1 = os.path.join(path, "data_train.csv")

df_train = pd.read_csv(csv_files1)
df_test = pd.read_csv(csv_files0)


# df_train = pd.read_csv("C:\\Users\\ag970\\Downloads\\cureya\\ML Model\\ML Model\\data_train.csv")
# df_test = pd.read_csv("C:\\Users\\ag970\\Downloads\\cureya\\ML Model\\ML Model\\data_test.csv")

X_train = df_train.Text
X_test = df_test.Text

y_train = df_train.Emotion
y_test = df_test.Emotion

class_names = ['joy', 'sadness', 'anger', 'neutral', 'fear']
data = pd.concat([df_train, df_test])


def preprocess_and_tokenize(data):    

    data = re.sub("(<.*?>)", "", data)

    data = re.sub(r'http\S+', '', data)
    
    data= re.sub(r"(#[\d\w\.]+)", '', data)
    data= re.sub(r"(@[\d\w\.]+)", '', data)
    data = re.sub("(\\W|\\d)", " ", data)
    data = data.strip()
    
    data = word_tokenize(data)
    
    porter = PorterStemmer()
    stem_data = [porter.stem(word) for word in data]
        
    return stem_data

vect = TfidfVectorizer(tokenizer=preprocess_and_tokenize, sublinear_tf=True, norm='l2', ngram_range=(1, 2))

vect.fit_transform(data.Text)

X_train_vect = vect.transform(X_train)
X_test_vect = vect.transform(X_test)


svc = LinearSVC(tol=1e-05)
svc.fit(X_train_vect, y_train)

ysvm_pred = svc.predict(X_test_vect)

svm_model = Pipeline([
    ('tfidf', vect),
    ('clf', svc),
])


filename = "C:\\Users\\ag970\\Downloads\\cureya\\ML Model\\ML Model\\tfidf_svm.sav"
pickle.dump(svm_model, open(filename, 'wb'))

model = pickle.load(open(filename, 'rb'))

# message = "I don't even know how to start or even what i am doing. Everyone fears about something, like spiders,snakes, rats, ghosts, death or even loneliness. My biggest fear is to tell people what I feel. I never tell anyone how I truly feel about someone or something because I feel like I'm going to disappoint them or make them change their minds about me. Even though it doesn't seem like, I really care about what people think about me, and I'm not dumb, I know that a lot of people think what they think, and they are right in a certain way. I try to be mate and to not be so obvious. But it's hard. It's fuckig hard when you have to live above other people's expectations, when you can't say or do what your heart tells you to. This has never happened before I mean I sometimes looked at them in a way that it wasn't meant to be but you know it never got so far. You look at them and you think: oh maybe it's just my head or ideas of mine and it's not true, maybe is because they are so pretty and that kind of mix things up"

def emotion_teller(msg):
    return model.predict([msg])

# print(emotion_teller(message))
