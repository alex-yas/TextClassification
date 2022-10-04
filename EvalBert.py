import argparse
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import datasets
import matplotlib.pyplot as plt
import torch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--load_path")
    args = parser.parse_args()

    data = datasets.load_dataset(args.data_path, data_files='test.csv')
    data = data['train']

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.load_path)
    model = DistilBertForSequenceClassification.from_pretrained(args.load_path, num_labels=5)

    data = data.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=20)

    y = data['label']
    data = data.remove_columns('label')

    y_pred = []
    for sample in data:
        inputs = tokenizer(sample['text'], padding=True, truncation=True, max_length=250, return_tensors="pt").to(device)
        outputs = model(inputs["input_ids"].to(device), inputs["attention_mask"].to(device))
        probs = outputs[0].softmax(1)
        y_pred.append(probs.argmax())

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print(classification_report(y, y_pred))

