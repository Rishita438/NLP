import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('nlp_data.csv')
print(df)

df['tokenized_text'] = df['text'].apply(word_tokenize)
print(df)

def Stop_Word_Removal(text, stop_words):
    filtered_words = []
    for w in text:
      if w.lower() not in stop_words:
        filtered_words.append(w)
    return filtered_words



def Stemming(text, stemmer):
  stemmed_words = []
  for word in text:
    stemmed_words.append(stemmer.stem(word))
  return stemmed_words


def Lemmatizing(text, lemmatizer):
  lemmatized_words = []
  for word in text:
    lemmatized_words.append(lemmatizer.lemmatize(word))
  return lemmatized_words


def Count(data_frame):
    word_freq = Counter() 
    for i,j in data_frame.iterrows():
        sl = j['filtered_text']
        for w in sl:
            word_freq[w] += 1

    return word_freq
 
stop_words = set(stopwords.words('english'))
df['filtered_text'] = df['tokenized_text'].apply(Stop_Word_Removal, args=(stop_words,))

stemmer = PorterStemmer()
df['stemmed_text'] = df['filtered_text'].apply(Stemming, args=(stemmer,))

lemmatizer = WordNetLemmatizer()
df['lemmatized_text'] = df['filtered_text'].apply(Lemmatizing, args=(lemmatizer,))

df['pos_tags'] = df['filtered_text'].apply(pos_tag)

word_freq = Count(df)
top_words = word_freq.most_common(15)
words, frequencies = zip(*top_words)

plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 15 Most Frequent Words')
plt.xticks(rotation=45, ha='right')
plt.show()