"""LiveKit Voice Pipeline Agent for Axiom Prop Trading Rules."""

import logging
from dotenv import load_dotenv

from langchain_core.messages import ToolMessage
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    TurnHandlingOptions,
    inference,
)
from livekit.plugins import langchain as lk_langchain

from prop_trading_rules_voice_agent.config import settings
from prop_trading_rules_voice_agent.tools import axiom_agent

_ = load_dotenv()
logger = logging.getLogger("axiom-voice-agent")


def _is_tool_token(item: object) -> bool:
    """Identifies tool message tokens emitted from LangChain to prevent them from reaching TTS."""
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage)


class SpeechOnlyGraph:
    """Wraps the LangChain agent graph so intermediate tool execution output is never spoken."""

    def __init__(self, graph: object) -> None:
        self._graph = graph

    def astream(self, *args: object, **kwargs: object):
        inner = self._graph.astream(*args, **kwargs)

        async def _filtered():
            async for item in inner:
                if not _is_tool_token(item):
                    yield item

        return _filtered()


class AxiomVoiceAgent(Agent):
    """Axiom Prop Trading voice agent powered by the LangChain trading rules graph."""

    def __init__(self) -> None:
        super().__init__(
            instructions="You are the Axiom Prop Trading voice assistant. Answer questions using verified trading rules.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(axiom_agent)),
        )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext) -> None:
    """Main WebRTC voice session handler."""
    logger.info("Connecting to LiveKit room: %s", ctx.room.name)

    session = AgentSession(
        stt=inference.STT(
            model=settings.stt_model,
            language=settings.stt_language,
        ),
        tts=inference.TTS(
            model=settings.tts_model,
            voice=settings.tts_voice,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "mode": "fixed",
                "min_delay": 0.5,
                "max_delay": 6.0,
            },
        ),
    )

    # Start the agent session attached to the LiveKit WebRTC room
    await session.start(agent=AxiomVoiceAgent(), room=ctx.room)

    # Initial spoken greeting to the trader
    await session.generate_reply(
        user_input=(
            "Greet the trader warmly as the Axiom Prop Trading voice assistant "
            "and ask how you can help them with account rules or guidelines today."
        )
    )


def main() -> None:
    """Entry point to start the LiveKit agent worker."""
    agents.cli.run_app(server)


if __name__ == "__main__":
    main()

