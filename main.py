import google.generativeai as genai
import os
from dotenv import load_dotenv
import pandas as pd


load_dotenv()


GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("What is the meaning of life?", stream=True)

for chunk in response:
    print(chunk.text)

    # palm.configure(api_key=PALM_API_KEY)
    # palm_gen = llm(provider="palm",api_key=PALM_API_KEY)
    #
    # models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    # model = models[0].name
    # print(model)