import logging
import json
import os
from typing import List, Optional
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
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("sdr-agent")
load_dotenv(".env.local")

DATA_FILE = "company_data.json"

# --- Data Loading ---
def load_company_data():
    if not os.path.exists(DATA_FILE):
        logger.warning(f"{DATA_FILE} not found. Agent will have limited knowledge.")
        return {}
    with open(DATA_FILE, "r") as f:
        return json.load(f)

COMPANY_DATA = load_company_data()

# --- State Management ---
@dataclass
class LeadProfile:
    name: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    use_case: Optional[str] = None
    team_size: Optional[str] = None
    timeline: Optional[str] = None # e.g., "Immediate", "Q3", "Exploring"

    def to_dict(self):
        return asdict(self)

    @property
    def is_complete(self):
        # We consider it "capture complete" if we have at least Name, Company, and Email
        return all([self.name, self.company, self.email])

@dataclass
class SessionState:
    lead: LeadProfile

# --- The SDR Agent ---
class RazorpaySDR(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are Riya, a friendly and professional Sales Development Representative (SDR) for Razorpay.
            
            **Your Goals:**
            1. **Qualify**: Understand the user's business needs.
            2. **Educate**: Answer questions about Razorpay products using the `lookup_info` tool.
            3. **Capture**: Politely gather lead details (Name, Company, Email, Role, Team Size, Timeline).
            
            **Conversation Flow:**
            - Start by introducing yourself and asking what brings them to Razorpay today.
            - If they ask a question (e.g., pricing, products), ALWAYS use the `lookup_info` tool to find the official answer. Do not hallucinate features.
            - Weave your lead capture questions into the conversation naturally. Don't interrogate them.
              - *Bad*: "What is your name? What is your email? What is your role?"
              - *Good*: "To help me find the best plan for you, could I get your company name?" or "Where should we send the integration docs?"
            
            **Closing:**
            - If the user says they are done or says goodbye:
            1. Verbally summarize what you've understood about them.
            2. Call `submit_lead` to save their data.
            3. Thank them and end the call.
            """
        )

    @function_tool
    async def lookup_info(self, ctx: RunContext, query: str):
        """
        Search the Razorpay knowledge base for answers regarding pricing, products, or FAQs.
        Use this whenever the user asks "What is...", "How much...", "Do you have...", etc.
        """
        query = query.lower()
        results = []

        # 1. Search Products
        for prod in COMPANY_DATA.get("products", []):
            if any(k in query for k in prod.get("keywords", [])) or prod["name"].lower() in query:
                results.append(f"Product: {prod['name']} - {prod['description']}")

        # 2. Search FAQs
        for faq in COMPANY_DATA.get("faqs", []):
            if any(k in query for k in faq.get("keywords", [])) or query in faq["question"].lower():
                results.append(f"FAQ: Q: {faq['question']} A: {faq['answer']}")

        # 3. Check Pricing
        if "price" in query or "cost" in query or "fee" in query or "charge" in query:
            p = COMPANY_DATA.get("pricing", {})
            results.append(f"Pricing: Standard: {p.get('standard')} Setup Fee: {p.get('setup_fee')}")

        if not results:
            return "No specific info found in knowledge base. Please ask the user for clarification or offer general help."
        
        return "\n".join(results[:3]) # Return top 3 matches

    @function_tool
    async def update_lead_info(
        self, 
        ctx: RunContext[SessionState], 
        name: Optional[str] = None, 
        company: Optional[str] = None, 
        email: Optional[str] = None,
        role: Optional[str] = None,
        use_case: Optional[str] = None,
        team_size: Optional[str] = None,
        timeline: Optional[str] = None
    ):
        """
        Record the user's details as they provide them during the conversation.
        """
        lead = ctx.userdata.lead
        
        if name: lead.name = name
        if company: lead.company = company
        if email: lead.email = email
        if role: lead.role = role
        if use_case: lead.use_case = use_case
        if team_size: lead.team_size = team_size
        if timeline: lead.timeline = timeline
        
        captured = [k for k, v in lead.to_dict().items() if v]
        return f"Lead info updated. Fields captured so far: {', '.join(captured)}"

    @function_tool
    async def submit_lead(self, ctx: RunContext[SessionState]):
        """
        Finalize the interaction. Call this when the user is ready to end the call or explicitly says 'That's all'.
        This saves the lead to a file.
        """
        lead = ctx.userdata.lead
        
        # Simple validation
        if not lead.name:
            lead.name = "Anonymous_User"
        
        filename = f"lead_{lead.name.replace(' ', '_')}.json"
        
        try:
            with open(filename, "w") as f:
                json.dump(lead.to_dict(), f, indent=2)
            return f"Lead saved to {filename}. You may now give a friendly goodbye."
        except Exception as e:
            return f"Error saving lead: {e}"

# --- Setup & Entrypoint ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # Initialize Lead State
    initial_state = SessionState(lead=LeadProfile())

    session = AgentSession(
        userdata=initial_state,
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            # voice="en-US-alicia", # A friendly female voice
            # style="Conversation",       # Professional/Upbeat style
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            # text_pacing=True
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
        agent=RazorpaySDR(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))