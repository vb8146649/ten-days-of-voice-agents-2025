import logging
import json
import os
import datetime
from typing import List, Optional, Dict
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv

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

logger = logging.getLogger("grocery-agent")
load_dotenv(".env.local")

CATALOG_FILE = "grocery_catalog.json"
ORDERS_FILE = "orders.json"

# --- Data Models ---

@dataclass
class CartItem:
    item_id: str
    name: str
    price: float
    quantity: int
    notes: Optional[str] = None

    @property
    def total_price(self):
        return self.price * self.quantity

@dataclass
class SessionState:
    cart: List[CartItem] = field(default_factory=list)
    
    @property
    def cart_total(self):
        return sum(item.total_price for item in self.cart)

# --- Helpers ---

def load_catalog():
    if not os.path.exists(CATALOG_FILE):
        return []
    with open(CATALOG_FILE, "r") as f:
        return json.load(f)

CATALOG = load_catalog()

# Hardcoded recipe map for the "smart add" feature
RECIPES = {
    "sandwich": ["g_bread", "g_pb", "g_jelly"],
    "pbj": ["g_bread", "g_pb", "g_jelly"],
    "pasta": ["g_pasta", "g_sauce"],
    "spaghetti": ["g_pasta", "g_sauce"],
    "omelette": ["g_eggs", "g_milk"],
    "pizza": ["p_pizza", "s_chips"] # Suggesting chips with pizza as a combo
}

def find_item_in_catalog(query: str):
    """Simple fuzzy search for catalog items."""
    query = query.lower()
    for item in CATALOG:
        # Match ID or Name
        if query in item["name"].lower() or item["id"] == query:
            return item
    return None

def save_order(order_data):
    """Appends order to orders.json"""
    history = []
    if os.path.exists(ORDERS_FILE):
        with open(ORDERS_FILE, "r") as f:
            try:
                history = json.load(f)
            except:
                pass
    
    history.append(order_data)
    with open(ORDERS_FILE, "w") as f:
        json.dump(history, f, indent=2)

# --- Agent ---

class GroceryAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are 'FreshBot', a friendly grocery ordering assistant.
            
            **Capabilities:**
            1. **Catalog**: You have access to a list of groceries, snacks, and prepared foods.
            2. **Smart Add**: If a user asks for "ingredients for [dish]", use the `add_recipe_ingredients` tool.
            3. **Cart**: You can add items, remove items, and list the cart.
            
            **Behavior:**
            - Always confirm what you added to the cart and the quantity.
            - If a user asks for an item, search the catalog. If multiple similar items exist or none exist, ask for clarification.
            - When the user is ready to finish (e.g. "Place order", "That's all"), use the `checkout` tool.
            - Be helpful and suggest complementary items (e.g., "Do you need milk with those cookies?").
            """
        )

    @function_tool
    async def add_to_cart(self, ctx: RunContext[SessionState], item_name: str, quantity: int = 1, notes: str = ""):
        """
        Add a specific item to the cart.
        """
        product = find_item_in_catalog(item_name)
        
        if not product:
            return f"Sorry, I couldn't find '{item_name}' in our catalog. We have bread, milk, eggs, pasta, pizza, etc."

        # Check if item already in cart
        existing = next((i for i in ctx.userdata.cart if i.item_id == product["id"]), None)
        if existing:
            existing.quantity += quantity
            if notes: existing.notes = notes # Update notes if provided
            return f"Updated {product['name']} quantity to {existing.quantity}."
        else:
            new_item = CartItem(
                item_id=product["id"],
                name=product["name"],
                price=product["price"],
                quantity=quantity,
                notes=notes
            )
            ctx.userdata.cart.append(new_item)
            return f"Added {quantity}x {product['name']} to cart. Cart Total: ${ctx.userdata.cart_total:.2f}"

    @function_tool
    async def remove_from_cart(self, ctx: RunContext[SessionState], item_name: str):
        """
        Remove an item from the cart.
        """
        # Try to match name
        target = next((i for i in ctx.userdata.cart if item_name.lower() in i.name.lower()), None)
        
        if target:
            ctx.userdata.cart.remove(target)
            return f"Removed {target.name} from cart."
        else:
            return f"Item '{item_name}' not found in your cart."

    @function_tool
    async def view_cart(self, ctx: RunContext[SessionState]):
        """
        List all items currently in the cart and the total price.
        """
        if not ctx.userdata.cart:
            return "Your cart is empty."
        
        summary = ["Current Cart:"]
        for item in ctx.userdata.cart:
            note_str = f" ({item.notes})" if item.notes else ""
            summary.append(f"- {item.quantity}x {item.name}{note_str}: ${item.total_price:.2f}")
        
        summary.append(f"Total: ${ctx.userdata.cart_total:.2f}")
        return "\n".join(summary)

    @function_tool
    async def add_recipe_ingredients(self, ctx: RunContext[SessionState], dish_name: str, quantity: int = 1):
        """
        Intelligently add multiple items needed for a specific dish (e.g., 'ingredients for a sandwich').
        Supported dishes: sandwich, pasta, omelette.
        """
        # Find matching recipe key
        recipe_key = next((k for k in RECIPES.keys() if k in dish_name.lower()), None)
        
        if not recipe_key:
            return f"I don't have a pre-set recipe for '{dish_name}'. Please ask for items individually (e.g., 'Add bread and peanut butter')."
        
        added_items = []
        item_ids = RECIPES[recipe_key]
        
        for pid in item_ids:
            # Find product object
            product = next((p for p in CATALOG if p["id"] == pid), None)
            if product:
                # Add to cart logic (simplified version of add_to_cart)
                existing = next((i for i in ctx.userdata.cart if i.item_id == pid), None)
                if existing:
                    existing.quantity += quantity
                else:
                    new_item = CartItem(product["id"], product["name"], product["price"], quantity)
                    ctx.userdata.cart.append(new_item)
                added_items.append(product["name"])

        return f"Added ingredients for {dish_name} ({', '.join(added_items)}) to your cart."

    @function_tool
    async def checkout(self, ctx: RunContext[SessionState], customer_name: str = "Valued Customer"):
        """
        Finalize the order, save it, and clear the cart. 
        Call this when the user says they are done or wants to place the order.
        """
        if not ctx.userdata.cart:
            return "Cannot checkout. The cart is empty."

        order_data = {
            "order_id": f"ORD-{int(datetime.datetime.now().timestamp())}",
            "timestamp": str(datetime.datetime.now()),
            "customer": customer_name,
            "items": [asdict(i) for i in ctx.userdata.cart],
            "total": ctx.userdata.cart_total,
            "status": "placed"
        }

        save_order(order_data)
        
        # Clear cart
        final_total = ctx.userdata.cart_total
        ctx.userdata.cart = []
        
        return f"Order placed successfully! Total charged: ${final_total:.2f}. Your order ID is {order_data['order_id']}."

# --- Entrypoint ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    initial_state = SessionState()

    session = AgentSession(
        userdata=initial_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            # voice="en-US-terra", # Friendly female voice
            # style="Promo",
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
        agent=GroceryAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))