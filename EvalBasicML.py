import argparse
from vectorizers import Word2Vec
import numpy as np
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import Preprocess
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--path_to_vectorizer")
    parser.add_argument("--path_to_classifier")
    parser.add_argument('--vectorizer', default='word2vec', choices=['word2vec'])
    parser.add_argument('--classifier', default='xgboost', choices=['xgboost'])
    args = parser.parse_args()

    dataset = Preprocess.get_preprocessed_dataset(args.data_path)

    clf = args.classifier
    vectorizer = args.vectorizer

    if vectorizer == 'word2vec':
        dataset = Word2Vec.get_vectorized_dataset(dataset, args.path_to_vectorizer)

    X = dataset.drop(columns=['label'])
    y = dataset['label']

    if clf == 'xgboost':
        classifier = pickle.load(open(args.path_to_classifier, 'rb'))

    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print(classification_report(y, y_pred))
