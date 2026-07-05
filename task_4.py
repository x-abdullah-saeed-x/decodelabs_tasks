import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import unicodedata
from scipy.sparse import csr_matrix

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv("Tweets.csv")

x=df['text']
y=df['airline_sentiment']

x=x.apply(lambda text: re.sub(r'[^a-zA-Z ]', '', text))

x=x.apply(lambda text: text.lower())

x=x.apply(lambda text: unicodedata.normalize('NFKD', text))

x=x.apply(word_tokenize)

stop_words = set(stopwords.words('english'))
stop_words.discard('not')
stop_words.discard('no')
stop_words.discard('nor')
x = x.apply(lambda tokens: [word for word in tokens if word.isalnum() and word not in stop_words])

lemet=WordNetLemmatizer()
x = x.apply(lambda tokens: [lemet.lemmatize(word, pos='v') for word in tokens])

x = x.apply(lambda tokens: " ".join(tokens))

vectorizer=TfidfVectorizer()
x = vectorizer.fit_transform(x)

x=csr_matrix(x)

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

model=MultinomialNB(alpha=0.01)

model.fit(x_train, y_train)
y_pred=model.predict(x_test)

print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='viridis')
plt.title("Naive Baise for Sentiment Analysis")
plt.legend()
plt.show()
