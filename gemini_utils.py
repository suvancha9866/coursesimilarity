import google.generativeai as genai
from config import GEMINI_API_KEY
import pandas as pd

# Initialize Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

def get_common_topic(course_descriptions, course_names):
    if not course_descriptions:
        return "No course descriptions provided."
    context = "\n".join([f"- {desc} (from {name})" for desc, name in zip(course_descriptions, course_names)])
    prompt = f"""Give me a topic that is common amongst all of these course descriptions:
    {context}
    """
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"