import os
import asyncio
import httpx
from google import genai
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize the new Google GenAI client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

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
    # Try multiple model names in order of preference
    model_names = [
        'gemini-2.0-flash-exp',  # Latest experimental model
        'gemini-1.5-flash',  # Stable flash model
        'gemini-1.5-pro',  # Pro model
        'gemini-pro',  # Older stable model
    ]
    
    prompt = f"{system_prompt}\n\nQuestion:\n{question}"
    
    # Try new SDK approach with different models
    for model_name in model_names:
        try:
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model=model_name,
                contents=prompt,
                config={
                    "temperature": 0.3,
                    "max_output_tokens": 500
                }
            )
            
            # Extract text from response
            if hasattr(response, 'text'):
                answer = response.text.strip()
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                answer = response.candidates[0].content.parts[0].text.strip()
            else:
                continue
            

            return  answer,
        except Exception as e:
            # Log error for debugging but continue to next model
            continue  # Try next model
    
    # If SDK fails, try REST API with different models
    rest_models = [
        ('gemini-2.0-flash-exp', 'v1beta'),
        ('gemini-1.5-flash', 'v1'),
        ('gemini-1.5-pro', 'v1'),
        ('gemini-pro', 'v1'),
        ('gemini-3-flash-preview', 'v1beta'),
        ('gemini-3-pro-preview', 'v1beta'),
    ]
    
    for model_name, api_version in rest_models:
        try:
            async with httpx.AsyncClient() as client:
                url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent"
                response = await client.post(
                    url,
                    params={"key": GEMINI_API_KEY},
                    json={
                        "contents": [
                            {
                                "parts": [
                                    {
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0.3,
                            "maxOutputTokens": 500
                        }
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["candidates"][0]["content"]["parts"][0]["text"]

                    return  answer.strip(),
        except Exception:
            continue  # Try next model
    
    # If all models fail, return error
    return {
        "model": "Gemini",
        "answer": "",
        "error": "All Gemini models failed. Please check your API key and available models.",
    }
