from models.roberta import RoBERTa
from models.sbert import SBERT
from models.bestsbert import SBERTquality
import numpy as np
import pandas as pd
import re
#import time
import gradio as gr
import csv

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
    if model == "Best Performance SBERT": # 106
        model1 = SBERT(file_content1, df["Description"])
    else: # 250
        model1 = RoBERTa(file_content1, df["Description"])
    scores = model1.similarity()

    df_similarity = pd.DataFrame({"Subject": df["Subject"].tolist(), "Number": df["Number"].tolist(), "Similarity": scores.tolist()[0]})
    df_similarity = df_similarity.sort_values(['Similarity'], ascending=False)
    df_similarity['Course Name'] = df_similarity['Subject'].astype(str) + " " + df_similarity['Number'].astype(str) 
    df_similarity["Similarity %"] = (df_similarity["Similarity"] * 100).round(2).astype(str) + "%"
    df_similarity_new = df_similarity[df_similarity["Course Name"] != name]
    df_similarity_new = df_similarity_new[["Course Name", "Similarity %"]].head(10)
    df_dict = df_similarity_new.set_index("Course Name")["Similarity %"].to_dict()
    similarity_dict[(name, model)] = df_dict
    return df_similarity_new

def update():
    with open("similarityscores.csv", "a", newline="") as f:
        w = csv.DictWriter(f, similarity_dict.keys())
        w.writeheader()
        w.writerow(similarity_dict)

update()

with gr.Blocks() as demo:
    gr.Markdown("# UIUC: Find a Similar Course")
    with gr.Tab("Project Description"):
        gr.Markdown("Welcome to UIUC: Find a Similar Course. This is a product developed by Suvan Chatakondu as a part of the Machine Learning/Artificial Intelligence Internship at ATLAS.")
        gr.Markdown("This is a solution to combat the problem of finding similar courses on campus. If you have ever seen a course on campus that seems to interest you, but maybe section times, prerequisites, or just availability in general don't work out with your schedule, you can use this product to find a similar course.")
        gr.Markdown("How do I use it? First, navigate to the Find Course tab. Input the course that you want to find similar courses to by using the drop down menu or typing the course in the specified format(DEP ###). \n NOTE: not all of the courses are listed. To improve efficiency, we took out some courses in the list. This includes courses that are crosslisted, so if you want to input a course that is crosslisted, please input the main department and number that course is listed under. \nThen you can choose one of the models. I recommend RoBERTa for better results, but you can use SBERT for faster results.")
        gr.Markdown("How long do I wait? The model shouldn't take more than 10 minutes. Sorry, it is not ideal, but we are going through all of the courses on campus and comparing the course descriptions. SBERT is faster than RoBERTa, but RoBERTa will give better results. Choose how you would like to.")
        gr.Markdown("How does it work? The algorithm that is working behind the scenes is a Natural Language Processing(NLP) algorithm. NLP is the way that a computer can read text, process it, and output a similarity score. Each model(SBERT/RoBERTa) has its own database and how it calculates semantics between words and outputs different similarity scores for the same texts.")
    with gr.Tab("Find Course"):
        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown(choices=options, label="Choose a Course")
                radio = gr.Radio(["SBERT", "RoBERTa"], label="Choose a Natural Language Processing Model")
            with gr.Column():
                gr.Markdown('NOTE: Not all courses are listed and Times May Vary')
                gr.Markdown('SBERT(Faster) ~ 120 seconds')
                gr.Markdown('RoBERTa(More accurate) ~ 250 seconds')
        button = gr.Button("Submit", variant="primary")
        output = gr.Dataframe(headers=["Course Name", "Similarity %"])
        gr.Markdown('Please ignore the processing time. Times may vary')
        button.click(course, inputs=[dropdown, radio], outputs=output)


demo.launch()