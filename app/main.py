from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from dotenv import load_dotenv

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
import time
from collections import deque, defaultdict

from app.schemas import QuizRequest
from app.prompts import build_system_prompt
from app.ai import call_openai, call_gemini
from fastapi.responses import StreamingResponse
import json

load_dotenv()

app = FastAPI(title="Quiz AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests=9, window_seconds=60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(deque)
        self.lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        # determine client IP (trust X-Forwarded-For if present)
        xff = request.headers.get("x-forwarded-for")
        if xff:
            ip = xff.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"

        now = time.time()
        async with self.lock:
            dq = self.requests[ip]
            # drop timestamps outside the window
            while dq and dq[0] <= now - self.window:
                dq.popleft()
            # if adding this request would reach 10 within the window, block
            if len(dq) >= self.max_requests:
                return JSONResponse({"detail": "Too many requests - rate limit exceeded"}, status_code=429)
            dq.append(now)

        response = await call_next(request)
        return response


# Add rate limiting middleware: blocks the 10th request within 60 seconds
app.add_middleware(RateLimitMiddleware, max_requests=9, window_seconds=60)


@app.get("/")
async def server_health():
    return {
        "health":"server running"
    }

@app.post("/answerQuiz")
async def answer_quiz(req: QuizRequest):
    if not req.subject or not req.question:
        raise HTTPException(status_code=400, detail="Subject and question are required")

    system_prompt = build_system_prompt(req.subject)
    # Start both model calls concurrently and stream results as they finish.
    async def event_stream():
        tasks = {
            asyncio.create_task(call_gemini(system_prompt, req.question)): "gemini",
            asyncio.create_task(call_openai(system_prompt, req.question)): "openai",
        }

        pending = set(tasks.keys())

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                model = tasks[d]
                try:
                    res = await d
                except Exception as e:
                    payload = {"model": model, "answer": "", "error": str(e)}
                else:
                    # normalize tuple returns from ai functions
                    if isinstance(res, tuple):
                        val = res[0] if len(res) > 0 else ""
                    else:
                        val = res

                    if isinstance(val, dict):
                        payload = val
                        payload.setdefault("model", model)
                    else:
                        payload = {"model": model, "answer": val}

                # Server-Sent Events format: event + data
                yield f"event: {model}\n"
                yield f"data: {json.dumps(payload)}\n\n"

        # send a final event to indicate completion
        yield "event: done\n"
        yield "data: {}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
    