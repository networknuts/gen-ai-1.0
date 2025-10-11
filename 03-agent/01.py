from dotenv import load_dotenv
from openai import OpenAI
import os, json
load_dotenv()
client = OpenAI()
SYSTEM_PROMPT = (
    "You are an honest assistant. You cannot access the internet or real-time tools. "
    "If the user asks for live data (like current weather), say you do not have that capability. "
    "Only answer with known, timeless information. Output JSON with keys: step, content."
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
