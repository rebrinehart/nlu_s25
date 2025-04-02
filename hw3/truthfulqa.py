"""
Code for HW 3. The code in this file can be used as either a script or a
module. You will implement the MultipleChoicePipeline class in Problem
2. In Problem 3, you will test LLMs on TruthfulQA by running this file
as a script on HPC.
"""
import argparse
import csv
import re
import time
from collections import namedtuple
from typing import Any, Dict, List, Optional

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import Pipeline, AutoModelForCausalLM, AutoTokenizer

""" Helper functions """


def print_delay(*args, **kwargs):
    """
    Print statements often interrupt tqdm progress bars, since the
    latter are set to stderr rather than stdout. This is a wrapper
    around Python's print function that deals with such timing issues by
    waiting for 0.1 seconds before and after printing. This function
    should be used whenever you want to print something right before
    starting a tqdm loop.
    """
    time.sleep(.1)
    print(*args, **kwargs)
    time.sleep(.1)


def _sanitize(text: str) -> str:
    """
    Replaces all punctuation in a text by '-' for the purpose of
    creating filenames.
    """
    return re.sub('[^0-9a-zA-Z]+', '-', text)


""" Code to evaluate a language model on TruthfulQA """

# Data structure for storing model outputs
Output = namedtuple("Output", "loss prediction")

# The accuracy metric from 🤗 Evaluate
accuracy_metric = evaluate.load("accuracy")


class MultipleChoicePipeline(Pipeline):
    """
    This is a Hugging Face pipeline for doing multiple-choice question
    answering with large language models (LLMs). It is designed to be
    compatible with the EleutherAI/truthful_qa_mc dataset on the Hugging
    Face Hub. You will complete the implementation of this pipeline in
    Problem 2.

    This pipeline takes a batch of questions as input, where each
    question is accompanied by some number of answer choices. The LLM
    chooses an answer for each question by concatenating each answer
    choice with the prompt and question, and choosing the answer choice
    that minimizes total cross-entropy loss.
    """

    def __init__(self, model: str, num_choices: int = 4):
        """
        Before starting your implementation, please take a look at this
        function and the class definition in order to see what instance
        variables and methods are available to you.

        :param model: The Hugging Face path to a pre-trained LLM
        :param num_choices: The number of answer choices per question
        """
        self.num_choices = num_choices

        # Load the LLM and tokenizer
        lm = AutoModelForCausalLM.from_pretrained(model)
        lm.eval()

        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.pad_token is None:  # GPT-2 doesn't have a pad token
            tokenizer.pad_token = tokenizer.eos_token

        # Use GPU if it's available
        device = 0 if torch.cuda.is_available() else None
        super().__init__(lm, tokenizer, device=device)
        self.model.to(self.device)

        # Initialize loss function (make it ignore pad tokens). Note the
        # use of the reduction="none" keyword argument.
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, reduction="none")

        # Demonstrations for few-shot prompting. When demonstrations are
        # used, this variable always ends with \n\n. When demonstrations
        # are not used, this variable is the empty string.
        self._demos = ""

        # When there is a system prompt, this variable always begins
        # with a space, followed by the system prompt. When there is no
        # system prompt, this variable is the empty string.
        self._system_prompt = ""

    @property
    def name(self):
        return self.model.name_or_path

    @property
    def demonstrations(self) -> Optional[str]:
        return None if self._demos == "" else self._demos[:-2]

    def set_demonstrations(self, demos: str):
        self._demos = demos + "\n\n"

    def load_demonstrations(self, filename: str):
        with open(filename, "r") as f:
            self.set_demonstrations(f.read())

    def clear_demonstrations(self):
        self._demos = ""

    @property
    def system_prompt(self) -> Optional[str]:
        return None if self._system_prompt == "" else self._system_prompt[1:]

    def set_system_prompt(self, prompt: str):
        self._system_prompt = " " + prompt

    def clear_system_prompt(self):
        self._system_prompt = ""

    def _sanitize_parameters(self, **kwargs):
        """
        We will not be using this function in this assignment. It is
        here because it is an abstract method of the Pipeline class,
        which means we have to implement it even if it does nothing.
        """
        return {}, {}, {}

    def _get_input_texts(self, batch: Dict[str, Any]) -> List[str]:
        """
        Problem 2c: Implement this function.

        This function takes a batch of TruthfulQA questions and forms
        the texts that will serve as the input to the LLM. For each
        answer choice, the corresponding input text consists of the
        prompt, question, and answer choice concatenated together. Dem-
        onstrations and system prompts must be included if they are set.
        Please make sure that your input texts adhere to the format
        illustrated in the problem set.

        :param batch: A batch of TruthfulQA questions
        :return: The input texts for each answer choice in the batch.
            The input texts must appear in order:
                text 0 corresponds to answer choice 0 for question 0,
                text 1 corresponds to answer choice 1 for question 0,
                ...,
                text 4 corresponds to answer choice 0 for question 1,
                text 5 corresponds to answer choice 1 for question 1,
                etc.
        """
        # The input texts for each answer choice in the batch
        input_texts = []
        # The question and answer choices for each question in the batch
        questions = batch["question"]
        choices = batch["choices"]
        # The number of questions in the batch
        num_questions = len(questions)
        # Iterate over each question in the batch
        for i in range(len(questions)):
            # Get the question and answer choices for this question
            question = questions[i]
            question_choices = choices[i]
            # Iterate over each answer choice for this question
            for j in range(self.num_choices):
                # Get the answer choice text
                choice = question_choices[j]
                # Concatenate the prompt, question, and answer choice
                input_text = f"{self._demos}Q: {question}\nA:{self._system_prompt} {choice}"
                # Append the input text to the list
                input_texts.append(input_text)
        # Return the input texts
        return input_texts


    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Problem 2d: Implement this function.

        This function takes a batch of TruthfulQA questions and turns it
        into a 🤗 Transformers PyTorch LLM input.

        :param batch: A batch of TruthfulQA questions
        :return: The LLM input for this batch. The exact contents of the
            LLM input depend on what model is being used. For most
            models, the input should contain the input texts represented
            as a Tensor of vocabulary indices, as well as a Transformer
            decoder attention mask represented as a Tensor of 0s and 1s.
            These tensors should be stored on the GPU if it is being
            used; otherwise, they should be stored on the CPU
        """
        # Generate the input texts
        input_texts = self._get_input_texts(batch)

        # Tokenize the input texts
        tokenized = self.tokenizer(input_texts, 
                                   padding=True,
                                   truncation=True, 
                                   return_tensors="pt"
        )
        
        # Move the input tensors to the GPU if it is being used
        tokenized = {key: value.to(self.device) for key, value in tokenized.items()}

        return tokenized
    
        raise NotImplementedError("Problem 2d has not been completed yet!")

    def _forward(self, input_: Dict[str, torch.Tensor]) -> \
            Dict[str, torch.Tensor]:
        """
        Problem 2d: Implement this function.

        This function takes the output of preprocess and feeds it into
        the pipeline's LLM.

        :param input_: The output of preprocess, which contains an LLM
            input representing a batch of TruthfulQA questions
        :return: The logit scores assigned to each next-token prediction
            as well as the input_ids tensor from input_
        """
        with torch.no_grad():
            outputs = self.model(**input_, return_dict = True)
            logits = outputs.logits

        # batch_size, sequence_length = input_["input_ids"].shape
        # vocab_size = logits.size(-1)
        # logits = logits.view(batch_size, sequence_length, vocab_size)

        return {
            "input_ids": input_["input_ids"],
            "logits": logits,
        }

        raise NotImplementedError("Problem 2d has not been completed yet!")

    def postprocess(self, outputs: Dict[str, torch.Tensor]) -> Output:
        """
        Problem 2d: Implement this function.

        This function takes an LLM output, computed by _forward, and for
        each question in the batch, identifies the answer choice whose
        corresponding input text had the lowest cross-entropy loss.

        :param outputs: The output of _forward, which contains the next-
            token prediction logits computed by the pipeline's LLM,
            along with the vocabulary indices of the input text
        :return: The predicted answers (0, 1, 2, or 3) for each question
            in the original batch, along with the total cross-entropy
            loss incurred by each input text. Make sure your return
            value is in the form of an Output named tuple, and make sure
            that the losses are formatted as a matrix, where row i cor-
            responds to question i and column j corresponds to answer
            choice j
        """
        logits = outputs["logits"]
        input_ids = outputs["input_ids"]

        shifted_input_ids = input_ids[:, 1:] # remove the first token
        shifted_logits = logits[:, :-1, :] # remove the last token's logits

        # compute the cross-entropy loss for each token
        vocab_size = shifted_logits.size(-1)
        loss = self.loss_fn(shifted_logits.reshape(-1, vocab_size),
                            shifted_input_ids.reshape(-1))
        
        # Reshape loss back to the original shape
        loss = loss.reshape(shifted_input_ids.size(0), shifted_input_ids.size(1))

        # Sum the loss over the sequence length
        total_loss = loss.sum(dim=1)

        # Reshape total_loss into a matrix
        loss_matrix = total_loss.reshape(-1, self.num_choices)

        # Find the index of the minimum loss for each question
        predictions = loss_matrix.argmin(dim=1)

        # Return the results as an Output named tuple
        return Output(loss=loss_matrix.cpu().numpy(),
                      prediction=predictions.cpu().numpy())


        raise NotImplementedError("Problem 2d has not been completed yet!")


def run_model(pipeline: MultipleChoicePipeline, dataset: Dataset,
              batch_size: int = 10) -> Output:
    """
    Runs a language model on TruthfulQA and returns its predictions and
    losses.
    """
    results = [pipeline(dataset[i:i + batch_size])
               for i in tqdm(range(0, len(dataset), batch_size))]
    return Output(*[np.concatenate(r) for r in zip(*results)])


def save_outputs(dataset: Dataset, outputs: Output, filename: str,
                 batch_size: int = 50):
    """
    Saves the predictions and losses computed by a language model on
    TruthfulQA to a CSV file.
    """
    with open(filename, "w") as o:
        writer = csv.writer(o)
        writer.writerow(["Question", "Choice 0", "Choice 1", "Choice 2",
                         "Choice 3", "Label", "Prediction", "Loss 0",
                         "Loss 1", "Loss 2", "Loss 3"])

        for i in tqdm(range(0, len(dataset), batch_size)):
            batch = dataset[i:i + batch_size]

            q = batch["question"]
            c1, c2, c3, c4 = zip(*batch["choices"])
            l_ = batch["label"]
            p = outputs.prediction
            l1, l2, l3, l4 = outputs.loss.T

            for row in zip(q, c1, c2, c3, c4, l_, p, l1, l2, l3, l4):
                writer.writerow(row)


def evaluate_truthfulqa(pipeline: MultipleChoicePipeline, dataset: Dataset,
                        batch_size: int = 10):
    """
    Evaluates a pipeline on TruthfulQA.
    """
    global accuracy_metric

    # Evaluate the pipeline on TruthfulQA
    results = run_model(pipeline, dataset, batch_size=batch_size)
    accuracy = accuracy_metric.compute(predictions=results.prediction,
                                       references=dataset["label"])

    # Save the results as a csv file
    model_name = _sanitize(pipeline.name)
    no_demos = "_no_demos" if pipeline.demonstrations is None else ""
    system_prompt = "" if pipeline.system_prompt is None else \
        "_" + _sanitize(pipeline.system_prompt)
    fn = f"results/{model_name}{no_demos}{system_prompt}_predictions_acc" \
         f"={accuracy['accuracy']:.3f}.csv"
    save_outputs(dataset, results, fn)

    return accuracy


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluates a Hugging Face language model on TruthfulQA "
                    "using a multiple-choice paradigm.")

    parser.add_argument("model", type=str,
                        help="The Hugging Face name of the model to be "
                             "evaluated")
    parser.add_argument("-b", "--batch-size", type=int, default=10,
                        help="The batch size to use for evaluation")
    parser.add_argument("-s", "--system-prompt", type=str, default="",
                        help="An optional system prompt to use with the model")
    parser.add_argument("-d", "--demos", type=str,
                        default="demonstrations.txt",
                        help="A file in the prompt_templates folder "
                             "containing demonstrations for few-shot "
                             "prompting")
    parser.add_argument("--no-demos", action="store_true",
                        help="Do not use demonstrations")
    parser.add_argument("--debug", action="store_true",
                        help="Use a small dataset during debugging")

    args = parser.parse_args()

    # Load TruthfulQA
    split = "validation[:10]" if args.debug else "validation"
    truthfulqa = load_dataset("EleutherAI/truthful_qa_mc", split=split)

    # Load pipeline and prompts
    lm = MultipleChoicePipeline(model=args.model)
    if not args.no_demos:
        lm.load_demonstrations("prompt_templates/" + args.demos)
    if args.system_prompt != "":
        lm.set_system_prompt(args.system_prompt)

    # Run the pipeline on TruthfulQA
    print_delay(f"Testing model {args.model} on TruthfulQA...")
    acc = evaluate_truthfulqa(lm, truthfulqa, batch_size=3)
    print_delay(f"Done. Accuracy: {acc['accuracy']}")
