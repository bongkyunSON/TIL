from fastapi import FastAPI
from pydantic import BaseModel
import openai

openai.api_key = "<YOUR_OPENAI_API_KEY>"

def summarize(text):
    system_instruction = "assistant는 user의 입력을 bullet point로 3줄 요약해준다."

    messages = [{"role": "system", "content": system_instruction},
                {"role": "user", "content": text} 
                ]

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    result = response['choices'][0]['message']['content']
    return result


app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/summarize")
def post_summarize(input_text: InputText):
    summary = summarize(input_text.text)
    return {"summary": summary}
