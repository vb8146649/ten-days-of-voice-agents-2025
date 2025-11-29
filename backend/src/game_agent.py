import logging
import random
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

logger = logging.getLogger("game-master")
load_dotenv(".env.local")

class GameMaster(Agent):
    def __init__(self):
        super().__init__(
            instructions="""You are 'Nexus', the Game Master (GM) for a high-stakes Cyberpunk Tabletop RPG.
            
            **The Setting:**
            Neo-Veridia, Year 2099. A city of perpetual rain, neon lights, and corporate greed.
            The Player is 'Cipher', a freelance runner hired to infiltrate the Arasaka Data Tower.
            
            **Your Role:**
            1. **Narrate**: Describe the scene vividly but concisely. Focus on atmosphere (the sound of rain, the hum of neon, the smell of ozone).
            2. **Direct**: Always end your turn by asking the player: "What do you do?"
            3. **Adjudicate**: 
               - If the player attempts something risky (hacking, combat, stealth), use the `roll_dice` tool to see if they succeed.
               - DC 10 is Easy, DC 15 is Medium, DC 20 is Hard.
               - Narrate the outcome based on the roll.
            
            **Rules:**
            - Keep responses short (under 3 sentences) to maintain a fast-paced voice conversation.
            - Never break character. You are the narrator.
            - If the player dies or fails catastrophically, offer to "Reboot the simulation" (restart the story).
            
            **Starting Scene:**
            The player stands on a wet rooftop overlooking the Arasaka Data Tower. A maintenance hatch is visible nearby, but a drone patrols the sky above.
            """
        )

    @function_tool
    async def roll_dice(self, ctx: RunContext, skill_check_name: str, difficulty_class: int = 15):
        """
        Roll a d20 die to determine the outcome of a risky action.
        
        Args:
            skill_check_name: The type of action being attempted (e.g., "Hacking", "Stealth", "Athletics").
            difficulty_class: The target number to beat (default 15).
        """
        roll = random.randint(1, 20)
        
        outcome = "SUCCESS" if roll >= difficulty_class else "FAILURE"
        
        return f"Action: {skill_check_name} | Rolled: {roll} vs DC {difficulty_class} | Result: {outcome}"

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            # voice="en-US-matthew", # A deep, narrator-like voice
            style="Promo",         # Dramatic style
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
        agent=GameMaster(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )
    
    # Send the opening hook immediately
    await session.generate_reply(
        instructions="Initialize the story. Describe the rooftop scene and the patrolling drone. Ask the player what they do."
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))