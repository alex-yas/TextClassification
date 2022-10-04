import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_path")
    parser.add_argument("--out_path")
    args = parser.parse_args()

    topics = os.listdir(args.in_path)
    data = {'label': [], 'text': []}
    for topic in topics:
        topic_path = os.path.join(args.in_path, topic)
        files = os.listdir(topic_path)
        for file in files:
            with open(os.path.join(topic_path, file), 'r') as in_file:
                text = in_file.read()
                data['label'].append(topic)
                data['text'].append(text)

    dataset = pd.DataFrame(data)

    encoder = LabelEncoder()
    dataset['label'] = encoder.fit_transform(dataset['label'])

    dataset.to_csv(os.path.join(args.out_path, 'full.csv'), index=False)

    train_data, test_data = train_test_split(dataset, stratify=dataset['label'], test_size=0.15)

    train_data.to_csv(os.path.join(args.out_path, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(args.out_path, 'test.csv'), index=False)