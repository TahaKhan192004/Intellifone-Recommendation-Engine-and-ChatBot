from fastapi import FastAPI, UploadFile, File, Form,HTTPException
from typing import List, Optional
from fastapi.responses import FileResponse
import requests
import os
import shutil
import uuid
from pydantic import BaseModel
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# --- Import your modules ---
from models import UsedMobile
from fastapi.responses import StreamingResponse
from RecommendationEngine.recommendation_service import get_recommendations,stream_recommendations
from models import ChatRequest, ChatResponse, ChatHistoryResponse
from ChatBot.chatbot import generate_reply,generate_stream_reply
from ChatBot.crud import (
    create_conversation,
    get_chat_history,
    get_chat_history_formatted,
    get_user_conversations,
    save_message
)

app = FastAPI(title="IntelliFone AI Backend")



##### New Streaming Versions
##### New Streaming Versions - api 1
@app.get("/recommend-stream/")
async def recommend_phones_stream(max_price: float, priority: str):
    generator = stream_recommendations(max_price, priority)
    return StreamingResponse(generator, media_type="text/plain")
from fastapi.responses import StreamingResponse
@app.post("/chat-stream")
async def chat_stream(req: ChatRequest):

    conversation_id = req.conversation_id

    if not conversation_id:
        conversation_id = create_conversation(req.user_id, req.message)

    history = get_chat_history(conversation_id)

    generator = generate_stream_reply(history, req.message)

    async def stream_and_save():
        full_response = ""

        async for chunk in generator:
            full_response += chunk
            yield chunk

        # Save AFTER streaming completes
        save_message(conversation_id, req.user_id, "user", req.message)
        save_message(conversation_id, req.user_id, "assistant", full_response)

    return StreamingResponse(stream_and_save(), media_type="text/plain")
# @app.post("/chat-stream")
# ##### New Streaming Versions - api 2
# async def chat_stream(req: ChatRequest):

#     conversation_id = req.conversation_id

#     if not conversation_id:
#         conversation_id = create_conversation(
#             req.user_id, req.message
#         )

#     history = get_chat_history(conversation_id)

#     generator = await generate_stream_reply(history, req.message)

#     async def stream_and_save():
#         full_response = ""

#         async for chunk in generator:
#             full_response += chunk
#             yield chunk

#         # Save AFTER streaming completes
#         save_message(conversation_id, req.user_id, "user", req.message)
#         save_message(conversation_id, req.user_id, "assistant", full_response)

#     return StreamingResponse(stream_and_save(), media_type="text/plain")

# ============================================================
#  ENDPOINT 5 — PHONE RECOMMENDATIONS (Old Version)
# ============================================================
@app.get("/recommend/")
async def recommend_phones(max_price: float, priority: str):
    return get_recommendations(max_price, priority)
# ============================================================
#  ENDPOINT 6 — CHATBOT INTERFACE (Old Version)
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    conversation_id = req.conversation_id

    if not conversation_id:
        conversation_id = create_conversation(
            req.user_id, req.message
        )

    history = get_chat_history(conversation_id)

    reply = generate_reply(history, req.message)

    save_message(conversation_id, req.user_id, "user", req.message)
    save_message(conversation_id, req.user_id, "assistant", reply)

    return {
        "conversation_id": conversation_id,
        "reply": reply
    }
# ============================================================
#  ENDPOINT 7 — get all messages in a conversation
@app.get("/chat/{conversation_id}", response_model=ChatHistoryResponse)
async def get_chat(conversation_id: str):
    history = get_chat_history_formatted(conversation_id)
    return history
@app.get("/conversations/{user_id}")
async def get_conversations(user_id: str):
    convs = get_user_conversations(user_id)
    return {"conversations": convs}