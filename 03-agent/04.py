# filename: script_d.py
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime
import json
import os
import requests

load_dotenv()
client = OpenAI()

# --- Tools ---
def get_weather(city: str):
    """Fetch simple weather info using wttr.in."""
    url = f"https://wttr.in/{city}?format=%C+%t"
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        return f"The weather in {city} is {r.text}."
    return "Something went wrong."


def run_command(cmd: str):
    """Execute a simple shell command using os.system."""
    try:
        result = os.system(cmd)
        return f"Command executed with exit code {result}"
    except Exception as e:
        return f"Command error: {e}"


# --- Tool Registry ---
available_tools = {
    "get_weather": get_weather,
    "run_command": run_command,
}


# --- System Prompt ---
SYSTEM_PROMPT = (
    "You are an assistant that operates via plan → action → observe → output.\n"
    "Choose tools only when needed, be explicit about inputs, and wait for observations before final answers.\n\n"
    "JSON schema (strict): {step, content, function, input}.\n\n"
    "Available Tools:\n"
    "- get_weather(city: str) → returns current weather via wttr.in\n"
    "- run_command(cmd: str) → executes a local shell command and returns stdout/stderr (demo only).\n\n"
    "Safety:\n"
    "- Prefer read-only commands (e.g., ls, pwd, whoami).\n"
    "- If a command seems risky or destructive, plan to ask the user first instead of executing.\n"
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
                obs = f"Unknown tool: {tool}"
            else:
                try:
                    obs = available_tools[tool](tool_input)
                except Exception as e:
                    obs = f"Tool error: {e}"

            messages.append({
                "role": "user",
                "content": json.dumps({"step": "observe", "output": obs})
            })
            continue

        if step == "output":
            print("🤖:", data.get("content"))
            break

        # Fallback for unexpected steps
        print("ℹ️ ", data.get("content"))
