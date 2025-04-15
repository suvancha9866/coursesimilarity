from models.roberta import RoBERTa
from models.sbert import SBERT
import numpy as np
import pandas as pd
import re
import gradio as gr
from gemini_utils import get_common_topic, find_similar_courses_with_gemini
from uiuc import UIUCTheme

df = pd.read_csv("course-catalog.csv")
df["Course"] = df['Subject'].astype(str) + " " + df['Number'].astype(str)
df = df.drop_duplicates(subset=['Subject', 'Number'])
df = df[~df['Description'].str.startswith("Same as", na=False)]
df = df[~df["Description"].str.startswith("May be repeated", na=False)]
df = df[~df["Description"].str.startswith("Approved for S/U grading only.", na=False)]
df["Description"] = df["Description"].str.split("Approved for").str[0]
df["Description"] = df["Description"].str.split("May be repeated").str[0]
df["Description"] = df["Description"].str.split("Prerequisite:").str[0]
df["Description"] = df["Description"].str.split("Credit is not").str[0]
df["Description"] = df["Description"].str.split("Continuation of").str[0]
df["Description"] = df["Description"].str.replace(
    r"\b\d+\s*(or\s*\d+)?\s*(to\s*\d+)?\s*(undergraduate|graduate|professional)\s*hours\b", "", regex=True)
df = df[~df['Description'].str.match(r'^\s*$|^\.*$', na=True)]
df = df.drop_duplicates(subset=["Description"])
#df = df.head()

options = df["Course"].tolist()
similarity_dict = {}
def course(name, model):
    if (name, model) in similarity_dict:
        similar_courses_data = similarity_dict[(name, model)]
        df_similarity = pd.DataFrame(list(similar_courses_data.items()), columns=["Course Name", "Similarity %"])
        return df_similarity, list(similar_courses_data.keys())
    row = df[df["Course"] == name]
    file_content1 = row["Description"].iloc[0]
    if model == "SBERT":
        model1 = SBERT(file_content1, df["Description"])
    elif model == "RoBERTa":
        model1 = RoBERTa(file_content1, df["Description"])
    else:
        df_similarity = pd.DataFrame({"Course Name": [name], "Similarity %": ["Pick a Model"]})
        return df_similarity, []
    scores = model1.similarity()
    df_similarity = pd.DataFrame({
        "Subject": df["Subject"].tolist(),
        "Number": df["Number"].tolist(),
        "Similarity": scores.tolist()[0]
    })
    df_similarity = df_similarity.sort_values(['Similarity'], ascending=False)
    df_similarity['Course Name'] = df_similarity['Subject'].astype(str) + " " + df_similarity['Number'].astype(str)
    df_similarity["Similarity %"] = (df_similarity["Similarity"] * 100).round(2).astype(str) + "%"
    df_similarity_new = df_similarity[df_similarity["Course Name"] != name]
    df_similarity_new = df_similarity_new[["Course Name", "Similarity %"]].head(5)
    df_dict = df_similarity_new.set_index("Course Name")["Similarity %"].to_dict()
    similarity_dict[(name, model)] = df_dict
    return df_similarity_new, list(df_dict.keys())

def get_common_topic_from_state(course_name, model_choice):
    if not course_name or not model_choice:
        return "Please select a course and model first."
    key = (course_name, model_choice)
    if key not in similarity_dict:
        return "Course similarity data not found. Run similarity search first."
    similar_courses_data = similarity_dict[key]
    similar_course_names = list(similar_courses_data.keys())
    descriptions = []
    for course_name in similar_course_names:
        try:
            description = df[df["Course"] == course_name]["Description"].iloc[0]
            descriptions.append(description)
        except IndexError:
            continue
    if not descriptions:
        return "No valid course descriptions found."
    return get_common_topic(descriptions, similar_course_names)

def find_more_courses_based_on_topic(course_name, model_choice):
    if not course_name or not model_choice:
        return "Please select a course and model first."
    
    key = (course_name, model_choice)
    if key not in similarity_dict:
        return "Course similarity data not found. Run similarity search first."
    
    similar_courses_data = similarity_dict[key]
    similar_course_names = list(similar_courses_data.keys())
    descriptions = [df[df["Course"] == course_name]["Description"].iloc[0] for course_name in similar_course_names]

    return find_similar_courses_with_gemini(descriptions, similar_course_names, df)

uiuc = UIUCTheme()
with gr.Blocks(theme = uiuc) as demo:
    gr.Markdown("# UIUC: Find a Similar Course")

    with gr.Tab("Project Description"):
        gr.Markdown("""
            # 🎓 UIUC: Find a Similar Course

            Created by **Suvan Chatakondu** as part of the **Machine Learning & AI Internship at ATLAS**.

            This application helps students at the **University of Illinois Urbana-Champaign (UIUC)** discover **alternative or similar courses** by analyzing and comparing course **descriptions** using advanced **Natural Language Processing (NLP)** techniques.
                    
            Dataset (Based on Spring 2025 Catalog) : https://waf.cs.illinois.edu/discovery/course-catalog.csv

            ---

            ### 💡 Why Use This?

            Sometimes, the perfect course doesn't work out — due to scheduling conflicts, prerequisites, or enrollment caps. This app helps you:

            - **Find 5 most similar courses** to any course at UIUC
            - **Understand shared topics** across courses using Google's **Gemini AI**
            - **Discover additional recommendations** using AI reasoning beyond traditional similarity

            ---

            ### ⚙️ How It Works

            The system uses two NLP models to compare course descriptions:

            - 🏃‍♂️ **SBERT** (Faster): Good for quick comparisons (~2 mins)
            - 🧠 **RoBERTa** (More accurate): Deeper contextual similarity (~4 mins)

            After selecting a course and model, the app outputs the top 5 most similar courses based on **textual similarity of descriptions**.

            You can also:

            - 🔮 **Ask Gemini**: Use Google's Gemini AI to extract the **common topic** shared by the top 5 results.
            - 🔍 **Find More Courses**: Let Gemini suggest **additional relevant courses** beyond the top 5 using language understanding and reasoning.

            ---

            ### 🚀 How to Use

            1. Go to the **“Find Courses”** tab.
            2. Select a course from the dropdown (e.g., `CS 225`).
            3. Pick a model: `SBERT` (fast) or `RoBERTa` (accurate).
            4. Click **“Find Similar Courses”** to generate your recommendations.
            5. Click **“Ask Gemini”** to see what they have in common.
            6. Optionally, click **“Find Even More Courses”** to explore deeper AI-driven suggestions.

            ---

            ### 📌 Notes

            - Not all courses are listed
                - Cross-listed courses are listed on the main department for the course
            - Processing times may vary based on description length, model used, and other factors. Gradio will be wrong for it.
            - Similarity is based solely on course **description text**, not credits, prerequisites, or schedules.
            - **Gemini Limits**:
                - 15 Requests per Minute
                - 1,500 Requests per Day

            ---

            Try it out and discover a whole new set of courses that might be a perfect fit! 🎯
            """)
    with gr.Tab("Find Courses"):
        gr.Markdown("## 🧾 Discover Similar Courses")
        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown(choices=options, label="Choose a Course")
                radio = gr.Radio(["SBERT", "RoBERTa"], label="Choose a Model")
            with gr.Column():
                gr.Markdown('SBERT(Faster) ~ 200 seconds')
                gr.Markdown('RoBERTa(More accurate) ~ 350 seconds')
        find_button = gr.Button("Find Similar Courses", variant="primary")
        output_similarity = gr.Dataframe(headers=["Course Name", "Similarity %"])
        similar_course_names_output = gr.State([])
        find_button.click(
            fn=course,
            inputs=[dropdown, radio],
            outputs=[output_similarity, similar_course_names_output],
        )
        gr.Markdown("---" \
        "")
        gr.Markdown("## 🔮 Common Topic Amongst Courses")
        analyze_button = gr.Button("Ask Gemini", variant="primary")
        common_topic_output = gr.Textbox(label="Topics:")
        analyze_button.click(
            fn=get_common_topic_from_state,
            inputs=[dropdown, radio],
            outputs=[common_topic_output]
        )
        gr.Markdown("---" \
        "")
        gr.Markdown("## 🔍 Find Even More Courses")
        find_more_courses_button = gr.Button("Find Even More Courses with Gemini")
        more_courses_output = gr.Textbox(label="Similar Courses:")

        find_more_courses_button.click(
            fn=find_more_courses_based_on_topic,
            inputs=[dropdown, radio],
            outputs=[more_courses_output]
        )

demo.title = "UIUC: Find a Similar Course"
demo.launch(favicon_path='uiuc.png')

