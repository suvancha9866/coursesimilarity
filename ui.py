from models.roberta import RoBERTa
from models.sbert import SBERT
from models.bestsbert import SBERTquality
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
#df = df.head(10)

options = df["Course"].tolist()
similarity_dict = {}
def course(name, model):
    if (name, model) in similarity_dict:
        return similarity_dict[(name, model)]
    row = df[df["Course"] == name]
    file_content1 = row["Description"].iloc[0]
    if model == "Best Performance SBERT": # 106
        model1 = SBERT(file_content1, df["Description"])
    elif model == "Best Quality SBERT": # 518
        model1 = SBERTquality(file_content1, df["Description"])
    else: # 250
        model1 = RoBERTa(file_content1, df["Description"])
    scores = model1.similarity()

    df_similarity = pd.DataFrame({"Subject": df["Subject"].tolist(), "Number": df["Number"].tolist(), "Similarity": scores.tolist()[0]})
    df_similarity = df_similarity.sort_values(['Similarity'], ascending=False)
    df_similarity['Course Name'] = df_similarity['Subject'].astype(str) + " " + df_similarity['Number'].astype(str) 
    df_similarity["Similarity %"] = (df_similarity["Similarity"] * 100).round(2).astype(str) + "%"
    df_similarity_new = df_similarity[df_similarity["Course Name"] != name]
    df_similarity_new = df_similarity_new[["Course Name", "Similarity %"]].head(10)
    similarity_dict[(name, model)] = df_similarity_new
    return df_similarity_new

with gr.Blocks() as demo:
    gr.Markdown("# UIUC: Find a Similar Course")
    with gr.Row():
        with gr.Column():
            dropdown = gr.Dropdown(choices=options, label="Choose a Course")
            radio = gr.Radio(["Best Quality SBERT", "Best Performance SBERT", "RoBERTa"], label="Choose a Model")
        with gr.Column():
            gr.Markdown('NOTE: Not all courses are listed.')
            gr.Markdown('Best Quality SBERT ~ 550 seconds')
            gr.Markdown('Best Performance SBERT ~ 120 seconds')
            gr.Markdown('RoBERTa ~ 250 seconds')
        button = gr.Button("Submit", variant="primary")
    output = gr.Dataframe(headers=["Course Name", "Similarity %"])
    gr.Markdown('Please ignore the processing time. Times may vary')
    button.click(course, inputs=[dropdown, radio], outputs=output)

demo.launch()