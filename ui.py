from models.roberta import RoBERTa
from models.sbert import SBERT
import numpy as np
import pandas as pd
import re
import gradio as gr
from gemini_utils import get_common_topic
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

uiuc = UIUCTheme()
with gr.Blocks(theme = uiuc) as demo:
    gr.Markdown("# UIUC: Find a Similar Course")

    with gr.Tab("Project Description"):
        gr.Markdown("""
                # 🎓 UIUC: Find a Similar Course

                Welcome to **UIUC: Find a Similar Course**, developed by Suvan Chatakondu as part of the **Machine Learning & AI Internship** at **ATLAS**.

                Ever found a course that sounds interesting, but the **timing**, **availability**, or **prerequisites** don’t align? This tool helps you discover **similar courses** at **UIUC** based on course descriptions, so you can explore more options that fit your needs.

                ---

                ### 🔍 How to Use This Tool

                1. **Go to the “Find Course” tab** above.
                2. **Choose a course** using the dropdown. Type the course code in this format: `DEPT ###` (e.g., `CS 225`).
                    - *Note:* Not all courses are listed for performance optimization. Cross-listed courses are grouped under their primary department.
                3. **Pick a model**:
                    - 🏃‍♂️ **SBERT** — faster (about ~2 minutes)
                    - 🧠 **RoBERTa** — more accurate, but slower (about ~4 minutes)
                4. Once similar courses are displayed, you can click **"Get Common Topic"** to see the main theme or subject area shared by those similar courses.

                ---

                ### ⚙️ How It Works

                This tool leverages **Natural Language Processing (NLP)** models to analyze and compare course descriptions, identifying other courses that share similar content. The two models used are:

                - **SBERT** and **RoBERTa**: Two cutting-edge NLP models that process course descriptions to determine how closely courses are related. 
                - Both models rank other courses based on similarity, displaying the top matches in a neat table.

                **SBERT** is faster (2 minutes) but less resource-intensive.  
                **RoBERTa** is more accurate, though it takes a bit longer (4 minutes).

                ---

                ### 🌐 Gemini Feature: Common Topic Generator

                After finding similar courses, the tool can go a step further using **Gemini API**, which analyzes the course descriptions and extracts the **common themes** shared by the most similar courses. This feature identifies the overarching **topics** or **concepts** that these courses explore, giving you a clearer understanding of the subject area.

                - **What does Gemini do?**
                    - It takes the descriptions of similar courses and identifies key patterns, common keywords, and shared topics.
                    - It helps you understand the **big picture** — what all those courses are fundamentally about, which can guide you in making more informed course decisions.
                
                Example: If you're comparing courses related to **Machine Learning**, Gemini might identify that the common topics are **Neural Networks**, **Deep Learning**, **Supervised Learning**, etc.

                **Use Case**: After you find similar courses in the “Find Course” tab, you can click **“Get Common Topic”** to see what **all the similar courses** have in common, making it easier for you to decide if the subject area fits your interests.

                ---

                ### 📌 Notes

                - Processing times may vary depending on the model and dataset size — thanks for your patience!
                - Feel free to explore different courses and models as many times as you'd like.
                - Gemini functionality provides a higher-level understanding of course content by focusing on shared themes.
        """)

    with gr.Tab("Find Course"):
        gr.Markdown("## 🧾 Discover Similar Courses")
        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown(choices=options, label="Choose a Course")
                radio = gr.Radio(["SBERT", "RoBERTa"], label="Choose a Model")
            with gr.Column():
                gr.Markdown('NOTE: Not all courses are listed and Times May Vary')
                gr.Markdown('SBERT(Faster) ~ 200 seconds')
                gr.Markdown('RoBERTa(More accurate) ~ 350 seconds')
        find_button = gr.Button("Find Similar Courses", variant="secondary")
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

demo.launch()
