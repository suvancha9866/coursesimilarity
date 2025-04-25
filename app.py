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
#df = df.head()

options = df["Course"].tolist()
similarity_dict = {}
def course(name, model):
    if (name, model) in similarity_dict:
        similar_courses_data = similarity_dict[(name, model)]
        df_similarity = pd.DataFrame(list(similar_courses_data.items()), columns=["Course Name", "Similarity %", "Description"])
        return df_similarity, list(similar_courses_data.keys())
    row = df[df["Course"] == name]
    file_content1 = row["Description"].iloc[0]
    if model == "SBERT":
        model1 = SBERT(file_content1, df["Description"])
    elif model == "RoBERTa":
        model1 = RoBERTa(file_content1, df["Description"])
    else:
        df_similarity = pd.DataFrame({"Course Name": [name], "Similarity %": ["Pick a Model"], "Description": ["N/A"],})
        return df_similarity, []
    scores = model1.similarity()
    df_similarity = pd.DataFrame({
        "Subject": df["Subject"].tolist(),
        "Number": df["Number"].tolist(),
        "Similarity": scores.tolist()[0],
        "Description": df["Description"].tolist()
    })
    df_similarity = df_similarity.sort_values(['Similarity'], ascending=False)
    df_similarity['Course Name'] = df_similarity['Subject'].astype(str) + " " + df_similarity['Number'].astype(str)
    df_similarity["Similarity %"] = (df_similarity["Similarity"] * 100).round(2).astype(str) + "%"
    df_similarity_new = df_similarity[df_similarity["Course Name"] != name]
    df_similarity_new = df_similarity_new[["Course Name", "Similarity %", "Description"]].head(5)
    df_dict = df_similarity_new.set_index("Course Name")["Similarity %"].to_dict()
    similarity_dict[(name, model)] = df_dict
    return df_similarity_new, list(df_dict.keys())

def course2(name, model):
    if (name, model) in similarity_dict:
        similar_courses_data = similarity_dict[(name, model)]
        df_similarity = pd.DataFrame(list(similar_courses_data.items()), columns=["Course Name", "Similarity %", "Description"])
        return df_similarity, list(similar_courses_data.keys())
    if model == "SBERT":
        model1 = SBERT(name, df["Description"])
    elif model == "RoBERTa":
        model1 = RoBERTa(name, df["Description"])
    else:
        df_similarity = pd.DataFrame({"Course Name": [name], "Similarity %": ["Pick a Model"], "Description": ["N/A"],})
        return df_similarity, []
    scores = model1.similarity()
    df_similarity = pd.DataFrame({
        "Subject": df["Subject"].tolist(),
        "Number": df["Number"].tolist(),
        "Similarity": scores.tolist()[0],
        "Description": df["Description"].tolist()
    })
    df_similarity = df_similarity.sort_values(['Similarity'], ascending=False)
    df_similarity['Course Name'] = df_similarity['Subject'].astype(str) + " " + df_similarity['Number'].astype(str)
    df_similarity["Similarity %"] = (df_similarity["Similarity"] * 100).round(2).astype(str) + "%"
    df_similarity_new = df_similarity[df_similarity["Course Name"] != name]
    df_similarity_new = df_similarity_new[["Course Name", "Similarity %", "Description"]].head(5)
    df_dict = df_similarity_new.set_index("Course Name")["Similarity %"].to_dict()
    similarity_dict[(name, model)] = df_dict
    return df_similarity_new, list(df_dict.keys())

def morecourse(similar_course_names, firstname, model, common_topic_output):
    descriptions = []
    for course_name in similar_course_names:
        try:
            desc = df[df["Course"] == course_name]["Description"].iloc[0]
            descriptions.append(desc)
        except IndexError:
            continue
    descriptions.append(common_topic_output)
    if not descriptions:
        return pd.DataFrame({"Course Name": [], "Similarity %": [], "Description": []}), []
    if model == "SBERT":
        model1 = SBERT(descriptions, df["Description"])
    elif model == "RoBERTa":
        model1 = RoBERTa(descriptions, df["Description"])
    else:
        return pd.DataFrame({"Course Name": ["Invalid"], "Similarity %": ["Pick a Model"], "Description": ["N/A"]}), []
    scores = model1.similarity()
    df_similarity = pd.DataFrame({
        "Subject": df["Subject"].tolist(),
        "Number": df["Number"].tolist(),
        "Similarity": scores.tolist()[0],
        "Description": df["Description"].tolist()
    })
    df_similarity["Course Name"] = df_similarity["Subject"].astype(str) + " " + df_similarity["Number"].astype(str)
    df_similarity["Similarity %"] = (df_similarity["Similarity"] * 100).round(2).astype(str) + "%"
    df_similarity = df_similarity[~df_similarity["Course Name"].isin(similar_course_names + [firstname])]
    df_similarity_new = df_similarity.sort_values("Similarity", ascending=False).head(5)
    df_similarity_new = df_similarity_new[["Course Name", "Similarity %", "Description"]]
    return df_similarity_new

def get_course_description(course_code):
    try:
        return df[df["Course"] == course_code]["Description"].iloc[0]
    except IndexError:
        return "Description not found."

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

                    Developed by **Suvan Chatakondu** as part of the **Machine Learning & Artificial Intelligence Internship at ATLAS**.

                    This application empowers students at the **University of Illinois Urbana-Champaign (UIUC)** to explore **alternative or related courses** by analyzing course descriptions through **cutting-edge Natural Language Processing (NLP)** models and **Google's Gemini AI**.

                    Sometimes, your ideal course doesn’t fit your schedule, has prerequisites, or fills up too fast. This tool helps you:

                    - 🔎 **Find 5 similar courses** based on course content
                    - 💬 **Use Gemini AI** to uncover shared themes across those courses
                    - 🌐 **Discover even more courses** by combining topic analysis with semantic similarity
                    - 🧠 Choose between **Course Number Dropdown** or **Keyword Input** to start your search

                    📚 Dataset: [Spring 2025 Course Catalog](https://waf.cs.illinois.edu/discovery/course-catalog.csv)

                    ---

                    ### 🚀 How to Use

                    1. Navigate to the **"Find Courses with Course Number"** tab to select a course, or the **"Find Courses with Keywords"** tab to search by topic.
                    2. Choose your NLP model: `SBERT` (fast) or `RoBERTa` (more accurate).
                    3. Click **"Find Similar Courses"** to generate your top 5 matches.
                    4. Click **"Ask Gemini"** to identify the **common topic** among them.
                    5. Click **"Discover Even More Courses"** to receive 5 additional recommendations based on both **topic** and **similarity**.

                    ---

                    ### ⚙️ How It Works

                    - 🏃‍♂️ **SBERT**: Delivers faster, lightweight comparisons (~2 mins)
                    - 🧠 **RoBERTa**: Performs deeper contextual matching (~4 mins)

                    **Gemini Integration**:
                    - Sends the top 5 course descriptions to Gemini
                    - Receives a human-readable summary of the shared theme
                    - Uses that theme + similarity to recommend even more courses

                    ---

                    ### 📌 Notes

                    - ⚠️ Gradio's displayed **processing time estimates** may not reflect actual runtime
                    - Matching is based solely on **course descriptions** — not credits, prerequisites, or scheduling
                    - Some cross-listed or special topic courses may be excluded
                    - **Gemini API Limits**:
                    - 15 requests/minute
                    - 1,500 requests/day

                    ---

                    🎯 Try it out and discover a whole new set of courses tailored to your academic and career goals!

                    ---

                    *Project Description generated with ideas by ChatGPT*
                    Feedback Form: https://docs.google.com/forms/d/e/1FAIpQLSdgd_CsPrjLDvnufnaFGiQampZYOCnhxZscbMypGYCmqQDsnQ/viewform?usp=dialog

        """)
    with gr.Tab("Find Courses with Course Number"):
        gr.Markdown("## 🧾 Discover Similar Courses using Course Numbers")
        with gr.Row():
            with gr.Column():
                dropdown = gr.Dropdown(choices=options, label="Choose a Course")
            with gr.Column():
                radio = gr.Radio(["SBERT", "RoBERTa"], label="Choose a Model")
        desc = gr.Textbox(label="Course Description", lines=3, interactive=False)
        dropdown.change(
            fn=get_course_description,
            inputs=dropdown,
            outputs=desc
        )
        find_button = gr.Button("Find Similar Courses", variant="primary")
        with gr.Column(elem_classes="dataframe-wrap"):
            output_similarity = gr.Dataframe(headers=["Course Name", "Similarity %", "Description"], wrap = True)
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
        gr.Markdown("## 🧠 Discover Even More Courses")
        button2 = gr.Button("Discover", variant="primary")
        output2 = gr.Dataframe(headers=["Course Name", "Similarity %", "Description"])
        button2.click(
            fn=morecourse,
            inputs=[similar_course_names_output, dropdown, radio, common_topic_output],
            outputs=output2
        )
    with gr.Tab("Find Courses with Keywords"):
        gr.Markdown("## 🧾 Discover Similar Courses using Keywords")
        with gr.Row():
            with gr.Column():
                dropdown = gr.Textbox(label="Input Keywords to look for")
            with gr.Column():
                radio = gr.Radio(["SBERT", "RoBERTa"], label="Choose a Model")
        find_button = gr.Button("Find Similar Courses", variant="primary")
        with gr.Column(elem_classes="dataframe-wrap"):
            output_similarity = gr.Dataframe(headers=["Course Name", "Similarity %", "Description"], wrap = True)
        similar_course_names_output = gr.State([])
        find_button.click(
            fn=course2,
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
        gr.Markdown("## 🧠 Discover Even More Courses")
        button2 = gr.Button("Discover", variant="primary")
        output2 = gr.Dataframe(headers=["Course Name", "Similarity %", "Description"])
        button2.click(
            fn=morecourse,
            inputs=[similar_course_names_output, dropdown, radio, common_topic_output],
            outputs=output2
        )


demo.title = "UIUC: Find a Similar Course"
demo.launch(favicon_path='uiuc.png')