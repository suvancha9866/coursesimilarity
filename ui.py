from roberta import RoBERTa
from sbert import SBERT
import numpy as np
import pandas as pd
import re
#import time
import gradio as gr

df = pd.read_csv("course-catalog.csv")
df["Course"] = df['Subject'].astype(str) + " " + df['Number'].astype(str) 
df = df.drop_duplicates(subset=['Subject', 'Number'])
df = df[~df['Section Info'].str.startswith("Same as", na=False)]
df = df[~df["Description"].str.startswith("May be repeated", na=False)]
df = df[~df["Description"].str.startswith("Approved for S/U grading only.", na=False)]
df["Description"] = df["Description"].str.split("Approved for").str[0]
df["Description"] = df["Description"].str.split("May be repeated").str[0]
df["Description"] = df["Description"].str.split("Prerequisite:").str[0]
df["Description"] = df["Description"].str.split("Credit is not").str[0]
df["Description"] = df["Description"].str.split("Continuation of").str[0]
df["Description"] = df["Description"].str.replace(r"\b\d+\s*(or\s*\d+)?\s*(to\s*\d+)?\s*(undergraduate|graduate|professional)\s*hours\b", "", regex=True)
df = df[~df['Description'].str.match(r'^\s*$|^\.*$', na=True)]
df = df.drop_duplicates(subset=["Description"])

options = df["Course"].tolist()
similarity_dict = {}
def course(name, model):
    if (name, model) in similarity_dict:
        return similarity_dict[(name, model)]
    row = df[df["Course"] == name]
    file_content1 = row["Description"].iloc[0]
    scores = []
    #start_time = time.time()
    for i in range(len(df)):
        if df.iloc[i]["Course"] != name:
            if model == "SBERT":
                model1 = SBERT(file_content1, df.iloc[i]["Description"])
            else:
                model1 = RoBERTa(file_content1, df.iloc[i]["Description"])
            scores.append([df.iloc[i]["Subject"], df.iloc[i]["Number"], model1.similarity()])
    # timey = time.time() - start_time
    # print("--- %s seconds ---" % (timey))
    df_similarity = pd.DataFrame(scores, columns=["Subject", "Number", "Similarity"])
    df_similarity = df_similarity.sort_values(['Similarity'], ascending=False)
    df_similarity['Course Name'] = df_similarity['Subject'].astype(str) + " " + df_similarity['Number'].astype(str) 
    df_similarity["Similarity %"] = (df_similarity["Similarity"] * 100).round(2).astype(str) + "%"
    df_similarity_new = df_similarity[["Course Name", "Similarity %"]].head()
    similarity_dict[(name, model)] = df_similarity_new
    return df_similarity_new

with gr.Blocks() as demo:
    gr.Markdown("# UIUC: Find a Similar Course")
    with gr.Row():
        with gr.Column():
            dropdown = gr.Dropdown(choices=options, label="Choose a Course")
            radio = gr.Radio(["SBERT", "RoBERTa"], label="Choose a Model")
        with gr.Column():
            gr.Markdown('NOTE: Not all courses are listed.')
            gr.Markdown('SBERT: Faster ~ 45 minutes, Less Accurate.')
            gr.Markdown('RoBERTa: Slower ~ 85 minutes, More Accurate.')
        button = gr.Button("Submit", variant="primary")
    output = gr.Dataframe(headers=["Course Name", "Similarity %"])
    gr.Markdown('Please ignore the processing time. Times may vary')
    button.click(course, inputs=[dropdown, radio], outputs=output)

demo.launch()