# filename: script_c.py
from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests

load_dotenv()
client = OpenAI()

# --- Tools ---
def get_weather(city: str):
    url = f"https://wttr.in/{city}?format=%C+%t"
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        return f"The weather in {city} is {r.text}."
    return "Something went wrong."


# --- Tool Registry ---
available_tools = {
    "get_weather": get_weather,
}


# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are an assistant that works in four phases: start/plan/action/observe/output.\n"
    "Given the user's query and these tools, you must: (1) plan the steps, (2) if needed, select a tool,\n"
    "(3) perform an action by emitting JSON telling which tool and input, (4) wait for observation,\n"
    "then (5) produce the final output.\n\n"
    "Rules:\n"
    "- Use the JSON schema strictly: {step, content, function, input}.\n"
    "- One step at a time; after an action, wait for an observation.\n"
    "- Be careful and concise.\n\n"
    "Available Tools:\n"
    "- get_weather(city: str) → returns current weather via wttr.in\n"
)


# --- Main Loop ---
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

while True:
    query = input("> ")
    messages.append({"role": "user", "content": query})

    while True:
        resp = client.chat.completions.create(
            model="gpt-4.1",
            response_format={"type": "json_object"},
            messages=messages,
        )

        raw = resp.choices[0].message.content
        messages.append({"role": "assistant", "content": raw})
        data = json.loads(raw)
        step = data.get("step")

        if step == "plan":
            print("🧠:", data.get("content"))
            continue

        if step == "action":
            tool = data.get("function")
            tool_input = data.get("input")
            print(f"🛠️  Calling {tool} with input: {tool_input}")

            if tool not in available_tools:
                messages.append({
                    "role": "user",
                    "content": json.dumps({
                        "step": "observe",
                        "output": f"Unknown tool: {tool}"
                    })
                })
                continue

            try:
                output = available_tools[tool](tool_input)
            except Exception as e:
                output = f"Tool error: {e}"

            messages.append({
                "role": "user",
                "content": json.dumps({"step": "observe", "output": output})
            })
            continue

        if step == "output":
            print("🤖:", data.get("content"))
            break

        # Fallback for unexpected steps
        print("ℹ️ ", data.get("content"))
