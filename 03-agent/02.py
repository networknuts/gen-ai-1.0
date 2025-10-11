from dotenv import load_dotenv
from openai import OpenAI

import os, json, requests

load_dotenv()
client = OpenAI()

# We define a function but never expose a calling protocol to the model.
def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        return f"The weather in {city} is {r.text}."
    else:
        return "Something went wrong."
    
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's questions. "
    "Do not fabricate real-time data. Output JSON with keys: step, content."
    )

messages = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    query = input("> ")
    messages.append({"role": "user", "content": query})

    resp = client.chat.completions.create(
    model="gpt-4.1",
    response_format={"type": "json_object"},
    messages=messages,
    )

    msg = resp.choices[0].message.content

    messages.append({"role": "assistant", "content": msg})

    data = json.loads(msg)
    
    print(" :", data.get("content"))
