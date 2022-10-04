import argparse
from transformers import TrainingArguments, Trainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os
import datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--save_path")
    args = parser.parse_args()

    train_data = datasets.load_dataset(args.data_path, data_files='train.csv', split='train[:85%]')
    test_data = datasets.load_dataset(args.data_path, data_files='test.csv')
    val_data = datasets.load_dataset(args.data_path, data_files='train.csv', split='train[15%:]')


    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path_or_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path_or_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_path_or_name, num_labels=5)

    train_data = train_data.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=20)
    test_data = test_data.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=20)
    val_data = val_data.map(lambda e: tokenizer(e['text'], padding=True, truncation=True), batched=True, batch_size=20)


    training_args = TrainingArguments(
        output_dir=os.path.join(args.save_path, 'outputs'),
        do_train=True,
        do_eval=True,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        logging_strategy='steps',
        logging_dir=os.path.join(args.save_path, 'outputs/log'),
        logging_steps=200,
        evaluation_strategy='steps',
        fp16=torch.cuda.is_available()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    results = trainer.train()

    trainer.save_model(args.save_path)
    tokenizer.save_pretrained(args.save_path)
