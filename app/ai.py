import os
import httpx
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the new Google GenAI client

# ---------- OpenAI ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

async def call_openai(system_prompt: str, question: str):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=30,
            )

        if response.status_code != 200:
            return {
                "answer": "",
                "error": response.text,
            }

        data = response.json()
        return  data["choices"][0]["message"]["content"].strip(),

    except Exception as e:
        return {
            "answer": "",
            "error": str(e),
        }



# ---------- Gemini ----------
async def call_gemini(system_prompt: str, question: str):
    prompt = f"{system_prompt}\n\nQuestion:\n{question}"

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                url,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"text": prompt}
                            ]
                        }
                    ],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": 400
                    }
                }
            )

            response.raise_for_status()
            data = response.json()

            return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return {
            "model": "Gemini",
            "answer": "",
            "error": str(e),
        }

