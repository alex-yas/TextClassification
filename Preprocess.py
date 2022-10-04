import argparse
import pandas as pd
import re
import string
import datasets
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import csv


def remove_punctuation(text):
    removed = "".join([i for i in text if i not in string.punctuation])
    return removed


def tokenization(text):
    tokens = re.split(r'\W+', text)
    return tokens


def remove_stopwords(text):
    stops = stopwords.words('english') + ['was', 'has', 'could', 'said']
    output = [i for i in text if i not in stops]
    return output


def remove_numbers(text):
    output = []
    for word in text:
        number = False
        for symbol in word:
            if symbol.isdigit():
                number = True
                break
        if not number:
            output.append(word)
    return output


def remove_short_words(text):
    output = [i for i in text if len(i) > 2]
    return output


def lemming(text):
    lemmatizer = WordNetLemmatizer()
    lemm_text = [lemmatizer.lemmatize(word) for word in text]
    return lemm_text


def preprocessing(series):
    series = series.apply(remove_punctuation)
    series = series.str.lower()
    series = series.apply(tokenization)
    series = series.apply(remove_numbers)
    series = series.apply(remove_stopwords)
    series = series.apply(lemming)
    series = series.apply(remove_short_words)
    series = series.apply(lambda x: ' '.join(x))
    return series


def get_preprocessed_dataset(path_to_data):
    dataset = pd.read_csv(path_to_data)

    dataset['text'] = preprocessing(dataset['text'])
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path")
    parser.add_argument("--out_path")
    args = parser.parse_args()

    dataset = get_preprocessed_dataset(args.in_path)

    dataset.to_csv(args.out_path, index=False)
