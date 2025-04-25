from models.sbert import SBERT
from models.roberta import RoBERTa
#from config import GEMINI_API_KEY
import google.generativeai as genai
import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-2.0-flash')

def get_common_topic(course_descriptions, course_names):
    if not course_descriptions:
        return "No course descriptions provided."
    context = "\n".join([f"- {desc}" for desc, name in zip(course_descriptions, course_names)])

    prompt = f"""Give me a topic that is common amongst all of these course descriptions:
    {context}
    Only give me the topic. Please do not add any extra words because I want to send the topic as an input for something else. Thank you!!
    """
    
    try:
        response = model_gemini.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error calling Gemini API: {e}"