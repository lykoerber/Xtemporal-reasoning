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

def few_shot(shot_dir, shots=5, mcq=True):
    df = pd.read_csv(shot_dir)
    prompt_str = ""
    for i in range(shots):
        ex_index = random.randint(0, len(df) - 1)
        ex_row = df.iloc[ex_index]
        option_str = ''
        if mcq:
            options = list(ex_row[1:4])
            option_str = f'\nOptions: (A) {options[0]} (B) {options[1]} (C) {options[2]}'
        prompt_str += f"Question: {ex_row.Question}{option_str}\nAnswer: {ex_row.Answer}\n\n\n"
    return prompt_str

def create_prompt(data, shots=5, shot_dir='../data/ordering/ordering_shots_mcq.csv', mcq=True):
    prompt_str = ""
    if shots > 0:
        shot_prompt = few_shot(shot_dir, shots)
        prompt_str += shot_prompt
    option_str = ''
    if mcq:
        options = list(data[1][1:4])
        option_str = f'\nOptions: (A) {options[0]} (B) {options[1]} (C) {options[2]}'
    prompt_str += f"Question: {data[1][0]}{option_str}\n"
    prompt_str += "Answer A, B, or C."
    return prompt_str


