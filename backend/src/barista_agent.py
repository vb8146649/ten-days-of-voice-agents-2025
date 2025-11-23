import logging
import json
import os
from typing import List, Optional, Annotated
from dataclasses import dataclass, field
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
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# 1. Define the Order State Structure (The Data)
@dataclass
class OrderState:
    drink_type: Optional[str] = None
    size: Optional[str] = None
    milk: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    name: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        # Check if all required fields are filled (extras is optional/defaults to empty)
        return all([self.drink_type, self.size, self.milk, self.name])

    def to_dict(self):
        return {
            "drinkType": self.drink_type,
            "size": self.size,
            "milk": self.milk,
            "extras": self.extras,
            "name": self.name
        }

# 2. Define the Session Userdata (Holds the State)
@dataclass
class SessionState:
    order: OrderState

# 3. The Agent Logic
class BaristaAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly, high-energy barista at 'Java Jive' coffee shop.
            Your goal is to take a customer's order efficiently and warmly.
            
            You maintain an order ticket that REQUIRES the following 5 fields:
            1. Drink Type (e.g., Latte, Cappuccino, Cold Brew)
            2. Size (Small, Medium, Large)
            3. Milk Choice (Whole, Oat, Almond, Skim, or "None")
            4. Extras (Syrups, sugars, etc. - If they say 'no', record 'None')
            5. Customer Name
            
            **Behavioral Rules:**
            - Ask one clarifying question at a time if fields are missing.
            - If the user gives multiple details (e.g., "Large Oat Latte"), capture all of them at once using the tool.
            - Once all 5 fields are captured, READ THE FULL ORDER BACK to the customer.
            - Only after they confirm the read-back, call `submit_order`.
            
            Be conversational. If they ask for a recommendation, suggest the "Java Jive Special" (a Medium Oat Milk Latte with Vanilla).
            """,
        )

    @function_tool
    async def update_order_details(
        self, 
        ctx: RunContext[SessionState], 
        drink_type: Optional[str] = None, 
        size: Optional[str] = None, 
        milk: Optional[str] = None, 
        extras: Optional[List[str]] = None, 
        name: Optional[str] = None
    ):
        """
        Update the order ticket with new details provided by the customer. 
        Only provide arguments for fields the customer has explicitly mentioned.
        """
        current_order = ctx.userdata.order

        # Update fields if provided
        if drink_type: current_order.drink_type = drink_type
        if size: current_order.size = size
        if milk: current_order.milk = milk
        if extras is not None: current_order.extras = extras
        if name: current_order.name = name

        # Check what is missing
        missing_fields = []
        if not current_order.drink_type: missing_fields.append("Drink Type")
        if not current_order.size: missing_fields.append("Size")
        if not current_order.milk: missing_fields.append("Milk")
        if not current_order.name: missing_fields.append("Customer Name")
        # We treat empty extras as valid if the rest is filled, but strictly we want them to confirm "no extras"
        
        state_dump = json.dumps(current_order.to_dict(), indent=2)
        
        if missing_fields:
            return f"Order Updated.\nCurrent State: {state_dump}\nMISSING: {', '.join(missing_fields)}. Please ask for these."
        else:
            return f"Order Complete! Details: {state_dump}. Please read this back to the user for confirmation."

    @function_tool
    async def submit_order(self, ctx: RunContext[SessionState]):
        """
        Finalize the order. Call this ONLY after the user has confirmed the full order details.
        """
        order = ctx.userdata.order
        
        if not order.is_complete:
            return "Order is incomplete. Please check missing fields before submitting."

        # Create a filename
        safe_name = "".join(x for x in (order.name or "customer") if x.isalnum())
        filename = f"order_{safe_name}.json"

        try:
            with open(filename, "w") as f:
                json.dump(order.to_dict(), f, indent=2)
            return f"Order successfully submitted and saved to {filename}. Thank the customer and end the interaction."
        except Exception as e:
            return f"Error saving file: {e}"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize the State (Learned from Drive-Thru resource)
    initial_state = SessionState(order=OrderState())

    session = AgentSession(
        # Pass the initialized state to the session
        userdata=initial_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew", 
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
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))