"""
Code for Problem 1 of HW 2.
"""
import pickle

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, BertTokenizerFast, BertForSequenceClassification, \
    Trainer, TrainingArguments
from train_model import preprocess_dataset
import numpy as np

def compute_accuracy(eval_pred):
        """
        Computes and returns accuracy metric.
        """
        metric = evaluate.load('accuracy')
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis = -1)
        return metric.compute(predictions = predictions, references = labels)

def init_tester(directory: str) -> Trainer:
    """
    Prolem 2b: Implement this function.

    Creates a Trainer object that will be used to test a fine-tuned
    model on the IMDb test set. The Trainer should fulfill the criteria
    listed in the problem set.

    :param directory: The directory where the model being tested is
        saved
    :return: A Trainer used for testing
    """
    model = AutoModelForSequenceClassification.from_pretrained(directory)

    trainable_params = count_trainable_parameters(model)
    print(f"Trainable Parameters: {trainable_params}")

    test_dataset = load_dataset("imdb", split = "test")
    test_args = TrainingArguments(output_dir = "./eval_results", do_train = False, do_eval = True)

    return Trainer(model = model,
                  args = test_args,
                  compute_metrics = compute_accuracy)

def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in the model.
    :param model: The model for which we want to count the parameters
    :return: The number of trainable parameters
    """
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_trainable_params

if __name__ == "__main__":  # Use this script to test your model
    model_name = "prajjwal1/bert-tiny"
    # with open("train_results_with_bitfit.p", "rb") as f:
    #     train_results = pickle.load(f)
    #     best_model_path = train_results["best_model_path"]  # Get saved model path

    # print(f"Loading model from {best_model_path}...")

    # Load IMDb dataset
    imdb = load_dataset("imdb")
    del imdb["train"]
    del imdb["unsupervised"]

    # Preprocess the dataset for the tester
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    imdb["test"] = preprocess_dataset(imdb["test"], tokenizer)

    # Set up tester
    tester = init_tester('/teamspace/studios/this_studio/nlu_s25/hw2/checkpoints_with_bitfit/run-1/checkpoint-5000')

    # Test
    results = tester.predict(imdb["test"])
    with open("test_results_with_bitfit.p", "wb") as f:
        pickle.dump(results, f)

