import logging
import json
import os
import datetime
import uuid
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from pydantic import BaseModel

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("shopping-agent")
load_dotenv(".env.local")

CATALOG_FILE = "acp_catalog.json"
ORDERS_FILE = "orders_acp.json"

# --- Commerce Logic Layer (The "Merchant API") ---

class ListProductsArgs(BaseModel):
    category: Optional[str] = None
    max_price: Optional[float] = None
    color: Optional[str] = None
    search_query: Optional[str] = None

def load_catalog():
    if not os.path.exists(CATALOG_FILE):
        return []
    with open(CATALOG_FILE, "r") as f:
        return json.load(f)

def load_orders():
    if not os.path.exists(ORDERS_FILE):
        return []
    try:
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_order_to_db(order):
    orders = load_orders()
    orders.append(order)
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)

CATALOG = load_catalog()

# --- Agent Definition ---

class ShoppingAssistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are a helpful Voice Shopping Assistant powered by the Agentic Commerce Protocol.
            
            **Your Role:**
            Help the user browse the catalog and place orders. You act as the bridge between the user's intent and the merchant's API.
            
            **Capabilities:**
            1. **Browse**: If the user asks for products (e.g., "Show me hoodies", "Any mugs under 500?"), use the `list_products` tool. 
               - Summarize the results concisely (Name + Price).
            2. **Order**: If the user wants to buy something, use the `create_order` tool.
               - You MUST identify the specific `product_id` from the catalog first. If the user is vague ("I'll take the hoodie"), ask for clarification (e.g., "The black one or the blue one?").
            3. **History**: If the user asks "What did I just buy?", use `get_last_order`.
            
            **Tone:**
            - Professional, efficient, and polite. 
            - When confirming an order, explicitly state the total amount and currency.
            """
        )

    @function_tool
    async def list_products(
        self, 
        ctx: RunContext, 
        args: ListProductsArgs,
    ):
        """
        Search and filter products in the catalog. Works with acp_catalog.json entries
        (id, name, price (int), currency, category, color, optional size).
        """
        category = args.category
        max_price = args.max_price
        color = args.color
        search_query = args.search_query

        def matches_term(item: dict, term: str) -> bool:
            t = term.lower()
            if t in item.get("name", "").lower():
                return True
            if t in item.get("category", "").lower():
                return True
            if t in item.get("id", "").lower():
                return True
            if t in item.get("color", "").lower():
                return True
            if t in str(item.get("size", "")).lower():
                return True
            tags = item.get("tags", [])
            if tags and any(t in str(tag).lower() for tag in tags):
                return True
            return False

        results = []
        for item in CATALOG:
            # Ensure required shape
            item_price = float(item.get("price", 0))
            item_currency = item.get("currency", "")

            # Category: if provided, allow matching against category OR name/id/tags
            if category and not matches_term(item, category):
                continue

            # max_price: compare numerically
            if max_price is not None and item_price > float(max_price):
                continue

            # color exact-ish match
            if color and color.lower() != item.get("color", "").lower():
                continue

            # general search query: match across name/id/category/color/size/tags
            if search_query and not matches_term(item, search_query):
                continue

            results.append(item)

        if not results:
            return "No products found matching those criteria."

        # Present consistent summary using fields present in acp_catalog.json
        summary = []
        for i in results:
            price = i.get("price", "")
            currency = i.get("currency", "")
            size = f" | Size: {i['size']}" if "size" in i else ""
            summary.append(f"ID: {i.get('id','')} | Name: {i.get('name','')} | Price: {price} {currency}{size}")

        return "\n".join(summary)

    @function_tool
    async def create_order(
        self, 
        ctx: RunContext, 
        product_id: str, 
        quantity: int = 1
    ):
        """
        Place an order for a specific product ID. Returns the order details.

        Args:
            product_id: The specific ID of the product to buy.
            quantity: The quantity to purchase. Defaults to 1.
        """
        # 1. Lookup Product
        product = next((p for p in CATALOG if p["id"] == product_id), None)
        if not product:
            return f"Error: Product ID '{product_id}' not found."

        # 2. Create Order Object
        total_price = product["price"] * quantity
        order_id = f"ORD-{uuid.uuid4().hex[:8]}"
        
        order = {
            "id": order_id,
            "items": [
                {
                    "product_id": product["id"],
                    "name": product["name"],
                    "quantity": quantity,
                    "unit_price": product["price"]
                }
            ],
            "total": total_price,
            "currency": product["currency"],
            "created_at": datetime.datetime.now().isoformat()
        }

        # 3. Persist
        save_order_to_db(order)

        return f"Order Created Successfully! Order ID: {order_id}. Total: {total_price} {product['currency']}."

    @function_tool
    async def get_last_order(self, ctx: RunContext):
        """
        Retrieve details of the most recently placed order.
        """
        orders = load_orders()
        if not orders:
            return "No recent orders found."
        
        last = orders[-1]
        item_summary = ", ".join([f"{i['quantity']}x {i['name']}" for i in last['items']])
        return f"Last Order ({last['created_at']}): {item_summary}. Total: {last['total']} {last['currency']}."

# --- Entrypoint ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-alicia", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=ShoppingAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))