import logging
import json
import os
from typing import Annotated, Literal, Optional
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

logger = logging.getLogger("tutor-agent")
load_dotenv(".env.local")

# --- Content Loading ---
CONTENT_FILE = "day4_tutor_content.json"

def load_content():
    if not os.path.exists(CONTENT_FILE):
        logger.warning("Content file not found!")
        return []
    with open(CONTENT_FILE, "r") as f:
        return json.load(f)

COURSE_CONTENT = load_content()
TOPIC_IDS = [t["id"] for t in COURSE_CONTENT]

def get_topic(topic_id: str):
    if not topic_id:
        return None
    # Case-insensitive search
    return next((t for t in COURSE_CONTENT if t["id"].lower() == topic_id.lower()), None)

# --- TTS Helper ---
def get_tts(voice_id: str):
    return murf.TTS(
        voice=voice_id,
        style="Conversation",
        tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
        text_pacing=True
    )

# --- Base Tutor Agent (Shared Tools) ---
class BaseTutorAgent(Agent):
    def __init__(self, current_topic_id: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.current_topic_id = current_topic_id

    @function_tool
    async def switch_mode(
        self, 
        ctx: RunContext, 
        mode: Annotated[Literal["learn", "quiz", "teach_back"], "The learning mode to switch to."],
        topic_id: Annotated[Optional[str], "The ID of the topic to focus on (variables, loops, functions)."] = None
    ):
        """
        Switch the learning mode or topic. 
        Use this when the user asks to change what they are doing (e.g., "Let's quiz me", "I want to explain it", "Teach me about loops").
        """
        # Default to current topic if not specified
        target_topic = topic_id if topic_id else self.current_topic_id
        
        # If still no topic (e.g. from greeting), default to the first one
        if not target_topic:
            target_topic = TOPIC_IDS[0]

        logger.info(f"Switching to mode: {mode} for topic: {target_topic}")
        
        if mode == "learn":
            return LearnAgent(target_topic), f"Switching to Learn Mode for {target_topic}."
        elif mode == "quiz":
            return QuizAgent(target_topic), f"Switching to Quiz Mode for {target_topic}."
        elif mode == "teach_back":
            return TeachBackAgent(target_topic), f"Switching to Teach-Back Mode for {target_topic}."
        
        return "Mode not recognized."

# --- Specific Agents ---

class LearnAgent(BaseTutorAgent):
    def __init__(self, topic_id: str):
        topic = get_topic(topic_id)
        # Safety check if topic is None
        title = topic['title'] if topic else topic_id
        summary = topic['summary'] if topic else "Content not found."
        
        super().__init__(
            current_topic_id=topic_id,
            instructions=f"""
                You are the 'Learn' module. Your voice is Matthew.
                Current Topic: {title}
                Summary: {summary}
                
                Your goal: Explain the concept clearly to the user.
                - Use the summary provided.
                - Elaborate with simple examples.
                - After explaining, ask if they want to switch to 'Quiz' mode or 'Teach-Back' mode to test their knowledge.
            """
        )

    async def on_enter(self, **kwargs):
        # Hot-swap the TTS voice using internal attribute to bypass setter restriction
        if hasattr(self, 'session'):
             self.session._tts = get_tts("en-US-matthew")
             topic = get_topic(self.current_topic_id)
             summary = topic['summary'] if topic else "I couldn't find the details for this topic."
             await self.session.generate_reply(instructions=f"Hello! I'm Matthew. Let's learn about {self.current_topic_id}. {summary}")

class QuizAgent(BaseTutorAgent):
    def __init__(self, topic_id: str):
        topic = get_topic(topic_id)
        # Safety check
        title = topic['title'] if topic else topic_id
        sample_q = topic['sample_question'] if topic else "No question found."

        super().__init__(
            current_topic_id=topic_id,
            instructions=f"""
                You are the 'Quiz' module. Your voice is Alicia.
                Current Topic: {title}
                Sample Question: {sample_q}
                
                Your goal: Test the user's knowledge.
                - Ask the sample question or variations of it.
                - Evaluate their answer.
                - If they get it right, suggest switching to 'Teach-Back' mode.
            """
        )

    async def on_enter(self, **kwargs):
        # Hot-swap the TTS voice
        if hasattr(self, 'session'):
            self.session._tts = get_tts("en-US-alicia")
            await self.session.generate_reply(instructions=f"Hi, I'm Alicia! Ready for a quiz on {self.current_topic_id}?")

class TeachBackAgent(BaseTutorAgent):
    def __init__(self, topic_id: str):
        topic = get_topic(topic_id)
        title = topic['title'] if topic else topic_id

        super().__init__(
            current_topic_id=topic_id,
            instructions=f"""
                You are the 'Teach-Back' module. Your voice is Ken.
                Current Topic: {title}
                
                Your goal: Act as a curious student. Ask the USER to explain the concept to YOU.
                - Say: "Okay, I'm ready to learn. How would you explain {title} to me?"
                - Listen to their explanation.
                - Give feedback: "That made sense!" or "I'm confused about X."
                - Grade them qualitatively (Great, Good, Needs Improvement).
            """
        )

    async def on_enter(self, **kwargs):
        # Hot-swap the TTS voice
        if hasattr(self, 'session'):
            self.session._tts = get_tts("en-US-ken")
            await self.session.generate_reply(instructions=f"Hey, I'm Ken. I hear you're the expert on {self.current_topic_id} now. Teach me!")

class GreetingAgent(BaseTutorAgent):
    def __init__(self):
        topic_list = ", ".join([t["title"] for t in COURSE_CONTENT])
        super().__init__(
            current_topic_id=None,
            instructions=f"""
                You are the receptionist for the Active Recall Tutor.
                Available Topics: {topic_list}.
                
                Your goal:
                1. Greet the user.
                2. List the available topics.
                3. Ask them to choose a topic and a mode (Learn, Quiz, or Teach-Back).
                4. Use the `switch_mode` tool to connect them to the right agent.
            """
        )

    async def on_enter(self, **kwargs):
        # Hot-swap the TTS voice
        if hasattr(self, 'session'):
            self.session._tts = get_tts("en-US-matthew") # Default voice
            await self.session.generate_reply(instructions="Welcome to the Active Recall Coach. Please choose a topic and a mode to start.")

# --- Entrypoint ---

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    # We start with a default configuration
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=get_tts("en-US-matthew"), # Start with Matthew
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

    # Start with the Greeting Agent
    await session.start(
        agent=GreetingAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))