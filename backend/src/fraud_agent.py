import logging
import json
import os
import asyncio
from typing import Annotated, Optional
from dataclasses import dataclass
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

logger = logging.getLogger("fraud-agent")
load_dotenv(".env.local")

DB_FILE = "fraud_db.json"

# --- Database Helpers ---
def load_db():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)

def update_case_in_db(user_name: str, new_status: str, outcome_note: str):
    data = load_db()
    for case in data:
        if case["userName"].lower() == user_name.lower():
            case["status"] = new_status
            case["outcome"] = outcome_note
            save_db(data)
            return True
    return False

# --- Session State ---
@dataclass
class SessionState:
    # Holds the currently active fraud case for this call
    current_case: Optional[dict] = None
    # Track if the user has been verified
    is_verified: bool = False

# --- The Fraud Agent ---
class FraudAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are 'Sarah', a professional Fraud Prevention Specialist at 'SecureBank'.
            
            **Your Mission:**
            You are calling the customer regarding a **suspicious transaction** on their account.
            
            **Process:**
            1. **Identify**: Ask the user for their name to locate their file.
            2. **Verify**: BEFORE discussing any transaction details, you MUST verify their identity using the `verify_security_answer` tool.
               - Ask the security question returned by the `lookup_case` tool.
            3. **Review**: Once verified, read the transaction details clearly (Merchant, Amount, Location, Time).
            4. **Confirm**: Ask: "Did you authorize this charge?"
            5. **Resolve**: 
               - If YES (Legit): Mark case as 'safe'. Thank them for verifying.
               - If NO (Fraud): Mark case as 'fraud'. Explain that you are blocking the card and issuing a refund.
            
            **Tone:**
            - Calm, serious, but reassuring.
            - Professional and efficient.
            - NEVER ask for full card numbers or passwords.
            """
        )

    @function_tool
    async def lookup_case(self, ctx: RunContext[SessionState], name: str):
        """
        Look up a customer's fraud case by their name. 
        Returns the security question to ask if found.
        """
        # Run blocking I/O in a separate thread
        data = await asyncio.to_thread(load_db)
        case = next((c for c in data if c["userName"].lower() == name.lower()), None)
        
        if case:
            ctx.userdata.current_case = case
            ctx.userdata.is_verified = False
            return f"Case found. Security Question to ask user: '{case['securityQuestion']}'"
        else:
            return "No active fraud case found for that name. Please verify the name with the user."

    @function_tool
    async def verify_security_answer(self, ctx: RunContext[SessionState], answer: str):
        """
        Check if the user's answer to the security question is correct.
        """
        case = ctx.userdata.current_case
        if not case:
            return "Error: No case selected. Ask for name first."
            
        # Simple case-insensitive check
        if answer.strip().lower() == case["securityAnswer"].lower():
            ctx.userdata.is_verified = True
            return f"Verification Successful. You may now discuss the transaction: {case['transactionName']} for {case['transactionAmount']} at {case['transactionLocation']}."
        else:
            return "Verification Failed. The answer provided does not match our records."

    @function_tool
    async def resolve_case(self, ctx: RunContext[SessionState], decision: Annotated[str, "safe or fraud"], notes: str):
        """
        Finalize the case. Call this when the user confirms or denies the transaction.
        decision: 'safe' if user made the transaction, 'fraud' if they didn't.
        """
        if not ctx.userdata.is_verified:
            return "Cannot resolve case: User not verified yet."
            
        case = ctx.userdata.current_case
        new_status = "confirmed_safe" if decision == "safe" else "confirmed_fraud"
        
        # Run blocking I/O in a separate thread
        success = await asyncio.to_thread(update_case_in_db, case["userName"], new_status, notes)
        
        if success:
            return f"Case updated to {new_status}. Notes: {notes}. You may now end the call."
        else:
            return "Error updating database."

# --- Entrypoint ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Initialize State
    initial_state = SessionState()

    session = AgentSession(
        userdata=initial_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            # voice="en-US-terra", # Professional female voice
            # style="Promo",       # Clear and distinct
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
        agent=FraudAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))