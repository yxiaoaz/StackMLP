import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk

def remove_stopwords(text):
    removed = []
    stop_words = list(stopwords.words("english"))
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        if tokens[i] not in stop_words:
            removed.append(tokens[i])
    return " ".join(removed)

def remove_extra_white_spaces(text):
    single_char_pattern = r'\s+[a-zA-Z]\s+'
    without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
    return without_sc

def lemmatizing(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    for i in range(len(tokens)):
        lemma_word = lemmatizer.lemmatize(tokens[i])
        tokens[i] = lemma_word
    return " ".join(tokens)


class preprocessing():
    def __init__(self,link):
        self.df=pd.read_csv(link,encoding='utf-8')

    def run(self):
        # transform label to {0,1}
        alphabet_to_num = {"NF": 0, "F": 1, }
        self.df["label"] = self.df["label"].map(lambda x: alphabet_to_num[x])

        # sort by timestamp
        self.df = self.df.sort_values("time", ascending=False)
        self.df = self.df.dropna()
        self.df["textual data"] = self.df["textual data"].apply(lambda x: remove_stopwords(x))
        self.df["textual data"] = self.df["textual data"].apply(lambda x: remove_extra_white_spaces(x))
        self.df.to_csv("Stemmed_All.csv", encoding='utf-8', index=False)
        print(self.df)
        return self.df


if __name__ == '__main__':
    pass

