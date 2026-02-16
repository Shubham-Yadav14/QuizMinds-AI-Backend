from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from dotenv import load_dotenv

from app.schemas import QuizRequest
from app.prompts import build_system_prompt
from app.ai import call_openai, call_gemini

load_dotenv()

app = FastAPI(title="Quiz AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/answerQuestion")
async def answer_quiz(req: QuizRequest):
    if not req.subject or not req.question:
        raise HTTPException(status_code=400, detail="Subject and question are required")

    system_prompt = build_system_prompt(req.subject)

    gemini_res, openai_res = await asyncio.gather(
        call_gemini(system_prompt, req.question),
        call_openai(system_prompt, req.question),
    )

    return {
        "subject": req.subject,
        "gemini": gemini_res,
        "openai": openai_res,
    }
