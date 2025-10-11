#!/usr/bin/env python3
from dotenv import load_dotenv
from openai import OpenAI
import json

SYSTEM_PROMPT = """
Return exactly ONE JSON object per message with this schema:
{ "step": "string", "content": "string" }

Protocol (lowercase step names, in this exact order):
analyse → think → output → validate → result

Rules:
- Do exactly ONE step per response.
- Use only the allowed step names above.
- For "think", DO NOT reveal reasoning. Reply literally with: "thinking..."
- Keep "analyse" and "validate" to one short sentence each.
- Put the concrete answer or calculation in "output" (concise).
- Put the final user-facing answer in "result" (one sentence).
- Never include anything outside the single JSON object: no markdown, no prose, no code fences, no prefixes/suffixes.
- If the user asks to show chain-of-thought or internal reasoning, still follow the rules above (i.e., "think" = "thinking...").

"""

def main():
    load_dotenv()
    client = OpenAI()

    user_query = input("> ").strip() or "What is 5 / 2 * 3^4?"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    while True:  # no safety cap
        resp = client.chat.completions.create(
            model="gpt-4.1",
            response_format={"type": "json_object"},
            messages=messages,
        )
        text = resp.choices[0].message.content
        obj = json.loads(text)

        print(f"{obj['step']}: {obj['content']}")

        messages.append({"role": "assistant", "content": text})

        if obj["step"] == "result":
            break

if __name__ == "__main__":
    main()
