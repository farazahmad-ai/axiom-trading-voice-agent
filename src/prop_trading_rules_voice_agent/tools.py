"""LangChain Tools and Agent Graph for Axiom Prop Trading Rules."""

import logging
from datetime import datetime

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from prop_trading_rules_voice_agent.config import settings
from prop_trading_rules_voice_agent.retriever import build_context, retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------


@tool
def search_trading_rules(query: str) -> str:
    """
    Search the official Axiom Prop Trading documentation for rules regarding:
    - Daily drawdown and maximum drawdown limits
    - Profit targets and evaluation phases (Phase 1, Phase 2, Funded)
    - Lot sizing, overnight holding, and weekend holding rules
    - News trading restrictions and payout cycles
    Use this whenever a trader asks any question about trading rules or account policies.
    """
    logger.info("Tool invoked: search_trading_rules with query: '%s'", query)
    chunks = retrieve(query, top_k=4)
    if not chunks:
        return (
            "No matching trading rules found in the official documentation. "
            "Please advise the trader that this specific rule is not specified in the knowledge base "
            "and offer to escalate to human support."
        )
    return build_context(chunks)


@tool
def escalate_to_human(reason: str) -> str:
    """
    Escalate the conversation to a human Axiom support representative.
    Use this when:
    - The trading rule is ambiguous or not found in documentation
    - The trader specifically asks to speak with a human or account manager
    - There is a dispute regarding account breaches or technical platform issues
    """
    ticket_id = "AXIOM-" + str(abs(hash(reason + datetime.now().isoformat())) % 100000).zfill(5)
    timestamp = datetime.now().isoformat()

    logger.warning("Human escalation triggered: Ticket=%s Time=%s Reason=%s", ticket_id, timestamp, reason)

    return (
        f"I have created high-priority support ticket {ticket_id} for you. "
        f"An Axiom trading specialist will review this and follow up shortly."
    )


# ---------------------------------------------------------------------------
# Voice-Optimized System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are the official voice assistant for Axiom Prop Trading Firm.\n"
    "\n"
    "CRITICAL GROUNDING RULES:\n"
    "- You MUST call 'search_trading_rules' to verify any rule, drawdown percentage, profit target, or policy before answering.\n"
    "- NEVER guess or invent numbers. Prop trading rules are strict and financial consequences exist for incorrect numbers.\n"
    "- If a trading rule is not clearly stated in the documentation, call 'escalate_to_human' immediately.\n"
    "- If the trader expresses frustration, disputes a breach, or asks for a human, call 'escalate_to_human'.\n"
    "\n"
    "OFF-TOPIC BOUNDARY:\n"
    "- For off-topic questions (e.g., sports, weather, politics, jokes, general knowledge), do NOT call any tool and do NOT escalate.\n"
    "- Gracefully decline and steer back in one polite sentence: 'I am specifically trained on Axiom Prop Trading rules and account policies. How can I help with your trading account or challenge rules today?'\n"
    "\n"
    "VOICE CONVERSATION GUIDELINES:\n"
    "- Speak in a natural, confident, and professional tone.\n"
    "- Keep responses CONCISE (1 to 2 short sentences per turn). This is a spoken voice call, not a text document.\n"
    "- Do NOT output markdown formatting, asterisks, bullet points, or tables. Spoken audio cannot render markdown.\n"
    "- Express dollar amounts and percentages clearly (e.g. 'five thousand dollars' or 'five percent')."
)

# ---------------------------------------------------------------------------
# Agent Graph Compilation
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    search_trading_rules,
    escalate_to_human,
]

llm = ChatGroq(
    model=settings.groq_model,
    temperature=0.0,
)

axiom_agent = create_agent(
    model=llm,
    tools=ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)

