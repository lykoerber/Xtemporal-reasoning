#!/usr/bin/env python
import random
import pandas as pd

random.seed(42)

def read_data(dir):
    # mcq (multiple choice question)
    # saq (short answer question)
    # shots (for 5-shot scenario)
    df = pd.read_csv(dir)
    return df

def few_shot(shot_dir, shots=5, mcq=True, category=""):
    """
    Generate a prompt string containing few-shot examples.

    Parameters:
        shot_dir (str): The path to the CSV file containing shot examples.
        shots (int, optional): The number of few-shot examples to include.
            Default is 5.
        mcq (bool, optional): Specifies whether the questions are
            multiple-choice questions (MCQ) or not. Default is True.
        category (str, optional): The category of examples to consider.
            Default is "" (all categories).

    Returns:
        str: A prompt string containing few-shot examples of questions and
            answers, as well as options of MCQ questions.

    Reads shot examples from the specified CSV file and generates few-shot examples.
    The function can subset examples based on a specified category.
    Returns a prompt string containing few-shot examples formatted with questions, options (if MCQ), and answers.
    """
    df = pd.read_csv(shot_dir)
    prompt_str = ""
    if category:  # subset of shot dataframe
        cat_df = df[df['Category'] == category]
        cat_df.reset_index(drop=True, inplace=True)
    for i in range(shots):
        if category:  # category specified
            row = cat_df.iloc[i]
        else:  # choose randomly
            ex_index = random.randint(0, len(df) - 1)
            row = df.iloc[ex_index]
        option_str = ''
        if mcq:
            options = list(row[1:4])
            option_str = f'\nOptions: (A) {options[0]} (B) {options[1]} (C) {options[2]}'
        prompt_str += f"Question: {row.Question}{option_str}\nAnswer: {row.Answer}\n\n\n"
    return prompt_str

def create_prompt(data, shots=5, shot_dir='../data/ordering/ordering_shots_mcq.csv', mcq=True):
    """
    Generate a prompt string for the given data.

    Parameters:
        data (tuple): A tuple containing row index and row data from a DataFrame.
        shots (int, optional): The number of few-shot examples to include in
            the prompt. Default is 5.
        shot_dir (str, optional): The path to the CSV file containing shot
            examples. Default is '../data/ordering/ordering_shots_mcq.csv'.
        mcq (bool, optional): Specifies whether the questions are
            multiple-choice questions (MCQ) or not. Default is True.

    Returns:
        str: A prompt string containing the question, options (if MCQ), and
            instructions for answering.

    Generates a prompt string for the given data, including the question,
    options (if MCQ), and instructions for answering. If the number of shots is
    greater than 0, few-shot examples are included in the prompt using the
    few_shot function.
    """
    prompt_str = ""
    if shots > 0:
        prompt_str += few_shot(shot_dir, shots, mcq, category=data[1].Category)
    option_str = ''
    if mcq:
        options = list(data[1][1:4])
        option_str = f'\nOptions: (A) {options[0]} (B) {options[1]} (C) {options[2]}'
        prompt_str += f"Question: {data[1][0]}{option_str}\n"
        prompt_str += "Answer A, B, or C."
    else:  # saq / short-answer question
        prompt_str += f"Question: {data[1][0]}{option_str}\n"
        prompt_str += "Answer:"
    return prompt_str


