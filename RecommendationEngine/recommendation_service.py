# recommendation_service.py

from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pydantic import BaseModel, Field
from typing import AsyncGenerator, Dict, Any

load_dotenv()

# ----------------------------
# Database Setup
# ----------------------------
MONGO_URI = os.getenv("MONGO_CONNECTION_STRING")
client = MongoClient(MONGO_URI)
db = client["MobileDB"]
recommended_collection = db["phones"]

# ----------------------------
# LLM Setup (Streaming Enabled)
# ----------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    streaming=True,
    temperature=0.3
)

# ----------------------------
# Input Schema
# ----------------------------
class PhoneRecommendationInput(BaseModel):
    max_price: float = Field(description="Maximum price budget")
    priority: str = Field(description="User priority (gaming, camera, battery, etc.)")


# ----------------------------
# Helper: Fetch Phones
# ----------------------------
def fetch_candidate_phones(max_price: float):
    """
    Fetch phones within ±5000 range of budget.
    """
    phones = list(
        recommended_collection.find({
            "$and": [
                {"price_range": {"$lte": max_price + 5000}},
                {"price_range": {"$gte": max_price - 5000}}
            ]
        })
    )
    return phones


# ----------------------------
# Helper: Build Prompt
# ----------------------------
def build_prompt(max_price: float, priority: str, phones: list) -> str:
    candidates = []

    for idx, phone in enumerate(phones, 1):
        name = phone.get("phone_name", "Unknown Phone")
        desc = phone.get("description", "No description available")
        price = phone.get("price_range", "Price not available")

        candidates.append(f"{idx}. {name} – {desc} – Rs {price}")

    prompt = f"""
You are a mobile phone expert.

User requirement:
- Budget: Rs {int(max_price)}
- Priority: {priority}

Candidate phones:
{chr(10).join(candidates)}

Instructions:
- Rank phones based on priority
- Clearly explain why each phone fits or does not fit
- Keep response clean and structured
- Use "->" for headings
- Use short paragraphs
- Avoid markdown symbols like * or **
- Use Rs format like: Rs 70,000

Provide a clean ranked recommendation list.
"""

    return prompt


# ----------------------------
# Streaming Recommendation
# ----------------------------
# async def stream_recommendations(max_price: float, priority: str) -> AsyncGenerator[str, None]:
#     """
#     Stream recommendations token by token.
#     """

#     phones = fetch_candidate_phones(max_price)

#     if not phones:
#         async def empty():
#             yield "No phones found in this price range."
#         return empty()

#     prompt = build_prompt(max_price, priority, phones)

#     async def generator():
#         full_response = ""

#         async for chunk in model.astream(prompt):
#             if chunk.content:
#                 full_response += chunk.content
#                 print(f"Streaming chunk: {chunk.content}")  # Debug log
#                 yield chunk.content

#         # (Optional) You can log or store full_response here if needed

#     return generator()
async def stream_recommendations(max_price: float, priority: str):
    phones = fetch_candidate_phones(max_price)

    if not phones:
        yield "No phones found in this price range."
        return

    prompt = build_prompt(max_price, priority, phones)

    async for chunk in model.astream(prompt):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content

# ----------------------------
# Non-Streaming Fallback
# ----------------------------
def get_recommendations(max_price: float, priority: str) -> Dict[str, Any]:
    """
    Standard (non-streaming) recommendation response.
    """

    phones = fetch_candidate_phones(max_price)

    if not phones:
        return {"recommendations": "No phones found in this price range."}

    prompt = build_prompt(max_price, priority, phones)

    response = model.invoke(prompt)

    return {
        "recommendations": response.content
    }