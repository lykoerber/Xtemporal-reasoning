#!/usr/bin/env python
import random
import pandas as pd

random.seed(42)

def read_data(dir):
    # mcq (multiple choice question)
    # saq (short answer question)
    # shots (for 5-shot scenario)
    try:
        df = pd.read_csv(dir)
    except:
        df = pd.read_csv(dir, engine='python', encoding_errors='ignore')
    return df

def parse_row(row, dirname, mcq: bool=True):
    """parse different columns of a dataframe row"""
    # options
    options = []
    if mcq:
        option_cols = [c for c in list(row.index) if c.startswith('Option')]
        options = [row[c] for c in option_cols]
    # question
    question = ''
    if 'nli' in dirname:
        question += f"Premise: {row.Premise}\nHypothesis: {row.Hypothesis}\n"
    elif 'causality' in dirname:
        question += f"Premise: {row.Premise}\n"
    elif 'storytelling' in dirname:
        question += f"Story: {row.Story}\n"
    question += f"Question: {row.Question}"
    # category
    try:
        category = row.Category
    except:
        category = row.Source
    # answer
    answer = row.Answer
    return question, category, options, answer

def few_shot(shot_dir: str, shots: int=5, mcq: bool=True, category: str=""):
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
        try:  # filter by category
            cat_df = df[df['Category'] == category]
            cat_df.reset_index(drop=True, inplace=True)
        except:  # use source instead
            cat_df = df[df['Source'] == category]
            cat_df.reset_index(drop=True, inplace=True)
    for i in range(shots):
        if category:  # category specified
            row = cat_df.iloc[i]
        else:  # choose randomly
            ex_index = random.randint(0, len(df) - 1)
            row = df.iloc[ex_index]
        question, category, options, answer = parse_row(row, shot_dir, mcq)
        #print(question, category, options, answer)
        option_str = '\n'
        if mcq:
            option_str += f'Options: (A) {options[0]} (B) {options[1]}'
            if len(options) > 2:
                option_str += f' (C) {options[2]}'
                if len(options) > 3:
                    option_str += f' (D) {options[3]}'
            option_str += '\n'
        prompt_str += f"{question}{option_str}Answer: {answer}\n\n"
    prompt_str += '\n'
    return prompt_str

def create_prompt(data: tuple, shots: int=5, shot_dir: str='../data/ordering/ordering_shots_mcq.csv',
        mcq: bool=True):
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
        prompt_str += f'Here are {shots} examples of questions and answers.\n\n'
        try:
            cat = data[1].Category
        except:
            cat = data[1].Source
        prompt_str += few_shot(shot_dir, shots, mcq, category=cat)
        prompt_str += 'Please give a short answer to the following question.\n\n'
    question, category, options, answer = parse_row(data[1], shot_dir, mcq)
    option_str = ''
    if mcq:
        option_str += f'\nOptions: (A) {options[0]} (B) {options[1]}'
        if len(options) > 2:
            option_str += f' (C) {options[2]}'
            if len(options) > 3:
                option_str += f' (D) {options[3]}'
        option_str += '\n'
    prompt_str += f"{question}{option_str}"
    prompt_str += "Answer: "
    return prompt_str

if __name__ == '__main__':
    #s = few_shot(shot_dir='../data/nli/nli_shots_saq.csv', shots=5, mcq=False, category="SNLI")
    #print(s)
    data = 'arithmetic'
    ds = read_data(f'../data/{data}/{data}_mcq.csv')
    for d in ds.iterrows():
        s2 = create_prompt(d, shot_dir=f'../data/{data}/{data}_shots_mcq.csv', mcq=True)
        print(s2)
        break
    
