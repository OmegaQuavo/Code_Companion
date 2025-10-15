from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# Allow frontend requests (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Regular chat (non-streaming)
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert programmer assistant that explains, fixes, and optimizes code clearly.",
            },
            {"role": "user", "content": user_input},
        ],
    )

    reply = response.choices[0].message.content
    return {"reply": reply}


# Streaming responses (word-by-word)
@app.get("/stream")
async def stream_response(prompt: str):
    def generate():
        for chunk in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            if hasattr(chunk, "choices"):
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content

    return StreamingResponse(generate(), media_type="text/plain")
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
