import nltk
import numpy as np
import pandas as pd
import string
import re
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

columnNames =['Label', 'Tweets']
arabicTweets = pd.read_csv(r'/Users/nadahmed/Downloads/semester 5/AI/proj/arabictweets.tsv',sep='\t', names=columnNames)
pd.set_option('display.max_colwidth', 100)

def normalize_arabic(tweet):
    tweet = re.sub("آ", "أ", tweet)
    tweet = re.sub("ى", "ي", tweet)
    tweet = re.sub("ة", "ه", tweet)
    tweet = re.sub("گ", "ك", tweet)
    return tweet

def remove_punctuations(tweet):
    no_panctuation ="".join([c for c in tweet if c not in string.punctuation])
    return no_panctuation


def removeDuplicates(tweet): #duplicates letters
    chars = []
    prev = None
    for c in tweet:
        if prev != c:
            chars.append(c)
            prev = c

    return ''.join(chars)

def preprocess(tweet):

    tweet = re.sub(r'[0-9]+', '', tweet)
    tweet = re.sub(r'[A-Za-z]+', '', tweet)
    tweet = tweet.replace("_", " ")
    tweet = remove_punctuations(tweet)
    tweet = normalize_arabic(tweet)
    #tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI['en'])
    tweet = ' '.join(dict.fromkeys(tweet.split())) #duplicates words
    tweet = removeDuplicates(tweet)
    return tweet

def listToString(tweet):
    str1 = ' '.join(map(str, tweet))
    return str1

arabicTweets["Tweets"] = arabicTweets['Tweets'].apply(lambda x: preprocess(x))

tokenizer = RegexpTokenizer(r'\w+')
arabicTweets["Tweets"] = arabicTweets["Tweets"].apply(tokenizer.tokenize)

stopwords_list = set(stopwords.words('arabic'))
arabicTweets["Tweets"]=arabicTweets["Tweets"].apply(lambda x: [item for item in x if item not in stopwords_list])
arabicTweets["Tweets"] = arabicTweets['Tweets'].apply(lambda x: listToString(x))

print(len((arabicTweets['Tweets'])))
print(arabicTweets['Tweets'])

kf=KFold(n_splits=20,random_state=1,shuffle=True)
for train_index,test_index in kf.split(arabicTweets):
    x=arabicTweets['Tweets']
    y=arabicTweets['Label']
    x_train , x_test, y_train , y_test= x[train_index],x[test_index],y[train_index],y[test_index]


clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC()),])
clf.fit(x_train,y_train)

predict=clf.predict(x_test)
print("accuracy" , metrics.accuracy_score(y_test,predict))
print(classification_report(y_test,predict))


clf=Pipeline([('tfidf',TfidfVectorizer()),('clf',LogisticRegression(solver='liblinear',C=1,random_state=1)),])
clf.fit(x_train,y_train)

predict=clf.predict(x_test)
print("accuracy" , metrics.accuracy_score(y_test,predict))
print(classification_report(y_test,predict))
