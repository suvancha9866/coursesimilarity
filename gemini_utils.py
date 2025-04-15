from models.sbert import SBERT
from models.roberta import RoBERTa
from config import GEMINI_API_KEY
import google.generativeai as genai

genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

def get_common_topic(course_descriptions, course_names):
    if not course_descriptions:
        return "No course descriptions provided."
    context = "\n".join([f"- {desc}" for desc, name in zip(course_descriptions, course_names)])

    prompt = f"""Give me a topic that is common amongst all of these course descriptions:
    {context}
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error calling Gemini API: {e}"

def find_similar_courses_with_gemini(course_descriptions, course_names, df):
    if not course_descriptions:
        return "No course descriptions provided."
    context = "\n".join([f"- {desc} (from {name})" for desc, name in zip(course_descriptions, course_names)])
    prompt = f"""
    Given these course descriptions, please find and list other courses from the dataset below that are most similar to them:

    {context}

    --- 

    Here are the available courses in the dataset:
    {df[['Course', 'Description']].to_string(index=False)}
    
    Please return a list of the top 5 courses (with course name and description) that match the provided descriptions the most.
    """
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error calling Gemini API: {e}"