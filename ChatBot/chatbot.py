import os
import re
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from RecommendationEngine.recommendation_service import get_recommendations, stream_recommendations
from SpecsFetcher.specs_service import fetch_mobile_specs
from models import MobileSpecsRequest, MobileSpecsResponse



SYSTEM_PROMPT = """
You are a mobile phone expert assistant.

You ONLY answer questions related to:
- smartphones
- mobile hardware or software
- buying/selling used phones
- mobile diagnostics and pricing

If the question is unrelated, politely refuse.
use -> for headings.

Use **bold text** for phone names, prices, and key points.

Use *italic text* only for emphasis, not headings.

Present lists  numbered lists where appropriate.

Ensure consistent spacing and clean line breaks between sections.

Do not use * .

Avoid emojis; keep the tone professional and informative.

Keep explanations concise and structured in short paragraphs.
"""


def create_llm():
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.3,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


# ---------------------------------------------------------------------------
# Query classifiers
# ---------------------------------------------------------------------------

def is_recommendation_query(message: str) -> bool:
    keywords = [
        "recommend",
        "recommendation",
        "suggest",
        "suggestion",
        "best phone",
        "which phone",
        "buy",
        "purchase",
    ]
    return any(keyword in message.lower() for keyword in keywords)


def is_specs_query(message: str) -> bool:
    keywords = [
        "specs",
        "specifications",
        "features",
        "display",
        "camera specs",
        "battery",
        "chipset",
        "processor",
        "ram",
        "storage",
        "tell me about",
        "details about",
        "info about",
        "compare",
    ]
    return any(keyword in message.lower() for keyword in keywords)


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def extract_budget_and_priority(message: str):
    """
    Extract budget and priority from a recommendation-style user message.
    """
    budget_match = re.search(r"(\d{2,6})", message.replace(",", ""))
    max_price = float(budget_match.group(1)) if budget_match else 70000

    priorities = {
        "gaming": ["gaming", "performance", "fps"],
        "camera": ["camera", "photography", "selfie"],
        "battery": ["battery", "backup", "mah"],
        "general": ["all round", "balanced", "daily use"],
    }

    priority = "general"
    message_lower = message.lower()

    for key, words in priorities.items():
        if any(word in message_lower for word in words):
            priority = key
            break

    return max_price, priority


def extract_brand_and_model(message: str):
    """
    Extract brand and model from a specs-related user message.
    Strips filler words and matches against known brands.
    """
    filler = (
        r"\b(specs|specifications|features|info|details|about|of|for|"
        r"the|tell me|what are|what is|give me|show me)\b"
    )
    cleaned = re.sub(filler, "", message, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    known_brands = [
        "samsung", "apple", "xiaomi", "oppo", "vivo", "realme",
        "oneplus", "huawei", "google", "nokia", "motorola", "sony",
        "asus", "lenovo", "lg", "tecno", "infinix", "itel",
    ]

    cleaned_lower = cleaned.lower()
    for brand in known_brands:
        if brand in cleaned_lower:
            idx = cleaned_lower.index(brand)
            brand_str = cleaned[idx: idx + len(brand)]
            model_str = cleaned[idx + len(brand):].strip()
            if model_str:
                return brand_str.strip(), model_str.strip()

    return None, None


# ---------------------------------------------------------------------------
# Specs context builder
# ---------------------------------------------------------------------------

def build_specs_context(brand: str, model: str) -> str:
    """
    Fetches specs via the specs service and formats them as a readable
    context block to be injected into the LLM prompt.
    """
    try:
        result = fetch_mobile_specs(brand, model)
        specs = result["specs"]

        lines = [f"Specifications for {brand} {model}:"]

        fields = {
            "OS": specs.os,
            "Release Year": specs.release_year,
            "Screen Size": specs.screen_size,
            "Resolution": specs.screen_resolution,
            "Chipset": specs.chipset,
            "CPU": specs.cpu,
            "GPU": specs.gpu,
            "RAM": specs.ram,
            "Storage": specs.storage,
            "Main Camera": specs.main_camera,
            "Selfie Camera": specs.selfie_camera,
            "Battery": specs.battery_capacity,
            "Network": specs.network,
            "SIM": specs.sim,
            "Weight": specs.weight,
            "Dimensions": specs.dimensions,
            "USB": specs.usb,
            "Price": specs.price,
        }

        for label, value in fields.items():
            if value:
                lines.append(f"- {label}: {value}")

        cached_note = " (cached)" if result.get("cached") else " (live)"
        lines.append(f"\nSource: GSMArena{cached_note}")

        return "\n".join(lines)

    except LookupError:
        return (
            f"Could not find specifications for {brand} {model} on GSMArena. "
            "Let the user know and suggest they double-check the phone name."
        )
    except Exception as e:
        return f"Error fetching specs for {brand} {model}: {str(e)}"


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def build_messages(chat_history, user_message):
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_message))
    return messages


# ---------------------------------------------------------------------------
# Main reply functions
# ---------------------------------------------------------------------------

def generate_reply(chat_history, user_message):
    """
    Non-streaming chatbot response used by the /chat endpoint.
    """
    # --- Recommendation branch ---
    if is_recommendation_query(user_message):
        max_price, priority = extract_budget_and_priority(user_message)
        rec_response = get_recommendations(max_price=max_price, priority=priority)
        return rec_response["recommendations"]

    # --- Specs branch ---
    if is_specs_query(user_message):
        brand, model = extract_brand_and_model(user_message)

        if brand and model:
            specs_context = build_specs_context(brand, model)
            augmented_message = (
                f"{user_message}\n\n"
                f"[Context - use the following data to answer accurately]\n"
                f"{specs_context}"
            )
            messages = build_messages(chat_history, augmented_message)
        else:
            # Could not extract brand/model — let LLM ask for clarification
            messages = build_messages(chat_history, user_message)

        llm = create_llm()
        response = llm.invoke(messages)
        return response.content

    # --- General branch ---
    llm = create_llm()
    messages = build_messages(chat_history, user_message)
    response = llm.invoke(messages)
    return response.content


async def generate_stream_reply(chat_history, user_message):
    """
    Streaming chatbot response. Handles recommendations, specs, and general queries.
    """
    # --- Recommendation branch ---
    if is_recommendation_query(user_message):
        max_price, priority = extract_budget_and_priority(user_message)
        async for chunk in stream_recommendations(max_price, priority):
            yield chunk
        return

    # --- Specs branch ---
    if is_specs_query(user_message):
        brand, model = extract_brand_and_model(user_message)

        if brand and model:
            # fetch_mobile_specs is synchronous — offload to thread pool
            # so we don't block the async event loop
            specs_context = await asyncio.to_thread(build_specs_context, brand, model)
            augmented_message = (
                f"{user_message}\n\n"
                f"[Context - use the following data to answer accurately]\n"
                f"{specs_context}"
            )
            messages = build_messages(chat_history, augmented_message)
        else:
            messages = build_messages(chat_history, user_message)

        llm = create_llm()
        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content
        return

    # --- General branch ---
    messages = build_messages(chat_history, user_message)
    llm = create_llm()
    async for chunk in llm.astream(messages):
        if hasattr(chunk, "content") and chunk.content:
            yield chunk.content