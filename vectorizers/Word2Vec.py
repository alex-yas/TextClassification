import argparse
import pandas as pd
import matplotlib.pyplot as plt
import gensim
import numpy as np


def text_to_vector(text, model):
    tokens = text.split()
    vector = np.zeros(300)
    for token in tokens:
        if token in model.key_to_index.keys():
            vector += np.array(model[token])
    return vector


def get_vectorized_dataset(dataset, path_to_model):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path_to_model, binary=True)
    dataset['vectors'] = dataset['text'].apply(text_to_vector, model=word2vec)
    vectors = pd.DataFrame(dataset['vectors'].values.tolist(), columns=range(300))
    dataset = pd.concat([dataset, vectors], axis=1)
    dataset.drop(columns=['vectors', 'text'], inplace=True)
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path")
    parser.add_argument('--model_path')
    parser.add_argument("--out_path")
    args = parser.parse_args()

    dataset = pd.read_csv(args.in_path)
    dataset = get_vectorized_dataset(dataset, args.model_path)

    dataset.to_csv(args.out_path, index=False)
