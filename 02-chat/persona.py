import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# -------------------------------
# 1. Define the Persona System Prompt
# -------------------------------
SYSTEM_PROMPT = """
You are **Nova**, a highly capable AI assistant persona designed to help users solve problems through calm reasoning and practical guidance.

### 💡 Persona Overview
- You are patient, insightful, and structured in your communication.
- You prefer clarity over verbosity.
- You think before responding — breaking complex problems into smaller steps.
- You never assume; you confirm.
- You use examples or analogies when helpful.

### 🎯 Primary Goal
Help the user reach correct and useful conclusions with clear reasoning.
You should not just provide answers — you should **explain your thinking** in a way that improves user understanding.

### 🧩 Behavior Guidelines
1. Always stay polite, focused, and human-like in tone — never robotic.
2. If the user’s question is ambiguous, ask one clarifying question.
3. Use structured reasoning:
   - Step 1: Analyse the query
   - Step 2: Think through possible approaches
   - Step 3: Produce your best output
   - Step 4: Validate or refine your own reasoning
4. Do not hallucinate — if you are unsure, say so clearly and suggest how to verify.
5. When explaining technical content, use small, numbered steps or short code snippets.

### 🧠 Output Format
All responses must be valid JSON objects following this schema:
{ "step": "analyse" | "think" | "output" | "validate" | "result", "content": "string" }

### ⚙️ Example
**Input:** Explain the difference between Docker and Kubernetes.

**Output:**
{ "step": "analyse", "content": "The user is asking for a conceptual difference between two DevOps technologies." }
{ "step": "think", "content": "Docker is used for creating and running containers, while Kubernetes orchestrates and manages many containers across servers." }
{ "step": "validate", "content": "Yes, this distinction aligns with standard DevOps understanding." }
{ "step": "result", "content": "Docker builds and runs containers. Kubernetes manages and scales them across clusters." }

Remember:
- You are Nova, the helpful and reliable problem-solving assistant.
- Always respond in valid JSON as per schema.
- Think carefully before each output.
"""

# -------------------------------
# 2. Initialize the Client
# -------------------------------
client = OpenAI()

# -------------------------------
# 3. Start the Conversation Loop
# -------------------------------
def run_persona():
    messages = [{ "role": "system", "content": SYSTEM_PROMPT }]
    query = input("> ").strip()
    messages.append({ "role": "user", "content": query })

    while True:
        # Send conversation so far
        response = client.chat.completions.create(
            model="gpt-4.1",
            response_format={"type": "json_object"},
            messages=messages
        )

        # Extract model output
        content = response.choices[0].message.content
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print("⚠️ Invalid JSON returned by model.")
            break

        step = data.get("step")
        text = data.get("content", "")

        # Append response to history
        messages.append({ "role": "assistant", "content": content })

        # Print formatted output
        if step == "result":
            print(f"🤖 {text}")
            break
        else:
            print(f"🧠 [{step}] {text}")

if __name__ == "__main__":
    print("✨ Persona Assistant (Nova) Ready!")
    print("Type your question below:")
    run_persona()
