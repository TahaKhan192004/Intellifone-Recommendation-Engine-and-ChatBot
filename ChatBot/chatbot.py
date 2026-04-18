from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import re
from typing import AsyncGenerator
from RecommendationEngine.recommendation_service import stream_recommendations

# ----------------------------
# System Prompt
# ----------------------------
SYSTEM_PROMPT = """
You are a mobile phone expert assistant.

You ONLY answer questions related to:
- smartphones
- mobile hardware or software
- buying/selling used phones
- mobile diagnostics and pricing

If the question is unrelated, politely refuse.

Formatting rules:
- Use -> for headings
- Use short paragraphs
- Use clean spacing
- Avoid markdown symbols like * or **
- Use simple readable formatting
"""

# ----------------------------
# LLM (Streaming Enabled)
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    streaming=True
)

# ----------------------------
# Extract Budget & Priority
# ----------------------------
def extract_budget_and_priority(message: str):
    budget_match = re.search(r'(\d{2,6})', message.replace(",", ""))
    max_price = float(budget_match.group(1)) if budget_match else 70000

    priorities = {
        "gaming": ["gaming", "performance", "fps"],
        "camera": ["camera", "photography", "selfie"],
        "battery": ["battery", "backup", "mah"],
        "general": ["all round", "balanced", "daily use"]
    }

    priority = "general"

    for key, words in priorities.items():
        if any(word in message.lower() for word in words):
            priority = key
            break

    return max_price, priority


# ----------------------------
# Intent Detection
# ----------------------------
def is_recommendation_query(message: str) -> bool:
    keywords = [
        "recommend",
        "recommendation",
        "suggest",
        "suggestion",
        "best phone",
        "which phone",
        "buy",
        "purchase"
    ]

    message = message.lower()
    return any(keyword in message for keyword in keywords)


# ----------------------------
# Build Messages
# ----------------------------
def build_messages(chat_history, user_message):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))
    return messages


# ----------------------------
# Streaming Chat Generator
# ----------------------------
async def generate_stream_reply(chat_history, user_message):

    # 🔀 Recommendation Flow
    if is_recommendation_query(user_message):
        max_price, priority = extract_budget_and_priority(user_message)

        async for chunk in stream_recommendations(max_price, priority):
            yield chunk
        return

    # 🤖 Normal Chat Flow
    messages = build_messages(chat_history, user_message)

    async for chunk in llm.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content


# ----------------------------
# Non-Streaming Fallback (Optional)
# ----------------------------
def generate_reply(chat_history, user_message):
    if is_recommendation_query(user_message):
        from RecommendationEngine.recommendation_service import get_recommendations
        max_price, priority = extract_budget_and_priority(user_message)

        rec_response = get_recommendations(
            max_price=max_price,
            priority=priority
        )

        return rec_response["recommendations"]

    llm_sync = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    messages = build_messages(chat_history, user_message)

    response = llm_sync.invoke(messages)
    return response.content