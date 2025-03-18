from roberta import RoBERTa
import numpy as np
import pandas as pd
import re
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
df = df.head() # remember to delete this

options = df["Course"].tolist()
def course(name):
    row = df[df["Course"] == name]
    file_content1 = row["Description"].iloc[0]
    scores = []
    for i in range(len(df)):
        if df.iloc[i]["Course"] != name:
            roberta1 = RoBERTa(file_content1, df.iloc[i]["Description"])
            scores.append([df.iloc[i]["Subject"], df.iloc[i]["Number"], roberta1.similarity()])

    df_similarity = pd.DataFrame(scores, columns=['Subject', 'Number', 'Similarity'])
    df_similarity = df_similarity.sort_values(['Similarity'], ascending=False)
    df_similarity['Course Name'] = df_similarity['Subject'].astype(str) + " " + df_similarity['Number'].astype(str) 
    df_similarity["Similarity %"] = (df_similarity["Similarity"] * 100).round(2).astype(str) + "%"
    df_similarity_new = df_similarity[["Course Name", "Similarity"]].head()
    return df_similarity_new

with gr.Blocks() as demo:
    gr.Markdown("# Find a Similar Course")
    with gr.Row():
        with gr.Column():
            dropdown = gr.Dropdown(choices=options, label="Choose a Course", value="Course")
            gr.Markdown('NOTE: Not all courses are listed. Runtimes are high as of now.')
        button = gr.Button("Submit", variant="primary")
    output = gr.Dataframe(headers=["Course Name", "Similarity %"])
    button.click(course, inputs=dropdown, outputs=output)

demo.launch()