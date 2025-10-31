# chatagent/agent.py
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from langchain_core.messages import SystemMessage, AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache

from chatagent.config import AgentState
from chatagent.summarizer import OllamaSummarizer
from chatagent.prompts import ASSISTANT_SYSTEM_PROMPT
from core.time import build_chat_clock_vars, resolve_timezone
from config import get_settings
from core.contracts import AgentMixin, ModelFactory
from utils.messages import TokenEstimator, coerce_message_content

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass



@dataclass
class ChatAgentConfig:
    messages_to_keep: int = 5
    max_tokens_before_summary: int = 4000
    model_name: str | None = None
    base_url: str | None = None
    temperature: float = 0.5
    num_ctx: int | None = None
    # allow callers to choose streaming-by-default semantics if they want
    default_stream: bool = False
    # If None, the agent will read the timezone from global settings
    tz_name: str | None = None


class ChatAgent(AgentMixin):
    config: ChatAgentConfig

    def __init__(self, config: ChatAgentConfig | None = None):
        super().__init__()
        self.config = config or ChatAgentConfig()

        # Load settings from environment if not provided in config
        settings = get_settings()
        if self.config.model_name is None:
            self.config.model_name = settings.model_name
        if self.config.base_url is None:
            self.config.base_url = settings.base_url
        # Make timezone configurable via global settings if not explicitly set
        if self.config.tz_name is None:
            # `Settings` provides a `timezone` attribute (IANA name)
            self.config.tz_name = getattr(settings, "timezone", "America/Sao_Paulo")

        self.summarizer = OllamaSummarizer(
            model_id=self.config.model_name or get_settings().model_name,
            base_url=self.config.base_url,
            k_tail=self.config.messages_to_keep,
        )

        # Token estimator (pure, cheap)
        self._tokenizer = TokenEstimator()

        # Timezone for clock/timestamps (standardized resolution)
        tz_in = self.config.tz_name or getattr(settings, "timezone", "America/Sao_Paulo")
        tz, norm, fell_back = resolve_timezone(tz_in)
        self.tz = tz
        self.config.tz_name = norm
        if fell_back:
            logger.warning("chatagent.timezone_fallback", extra={"requested_tz": tz_in, "used": norm})

        self.build()

    # ---------------------------
    # Node builders (async-first)
    # ---------------------------

    def _build_generate_node(self):
        model_name = self.config.model_name or ""
        if not model_name:
            raise ValueError("model_name must be set ...")
        
        # Create model once per compiled graph via factory (provider-safe)
        model = ModelFactory.create_chat_model(
            model_name=model_name,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
        )

        # System message rendered via shared prompt abstraction

        async def _generate(state: AgentState):
            # Observability context
            run_id = (state.get("metadata") or {}).get("run_id", "unknown")
            msgs = state.get("messages", [])
            
            # Standardized time context for prompts
            tc = build_chat_clock_vars(self.tz)
            clock = tc["clock"]
            
            sys_text = ASSISTANT_SYSTEM_PROMPT.format_system(
                clock=clock, summary=state.get("summary") or ""
            ) or ""
            sys = SystemMessage(content=sys_text)

            # Token estimate (input)
            input_tokens = self._tokenizer.count_messages([sys, *msgs])

            t0 = time.perf_counter()
            try:
                if state.get("stream"):
                    chunks: list[str] = []
                    async for chunk in model.astream([sys, *msgs]):
                        chunks.append(coerce_message_content(chunk))
                    out_text = "".join(chunks)
                    resp = AIMessage(content=out_text)
                else:
                    resp = await model.ainvoke([sys, *msgs])
                    out_text = coerce_message_content(resp)
            finally:
                dt = time.perf_counter() - t0

            # Token estimate (output)
            output_tokens = self._tokenizer.count_text(out_text)
            step_total = input_tokens + output_tokens

            # Keep stats in state (state is the source of truth)
            stats = state.get("stats", {})
            stats["last_step_tokens"] = step_total
            stats["window_tokens"] = stats.get("window_tokens", 0) + step_total
            state["stats"] = stats

            logger.info(
                "chatagent.generate_complete",
                extra={
                    "run_id": run_id,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "step_total": step_total,
                    "window_total": stats["window_tokens"],
                    "latency_ms": dt * 1000.0,
                },
            )

            new_history = (state.get("history", []) + ([msgs[-1]] if msgs else []) + [resp])
            return {
                "messages": [resp],
                "history": new_history,
                "stats": stats,
            }

        return _generate

    def _build_summarize_node(self):
        async def _summarize(state: AgentState):
            run_id = (state.get("metadata") or {}).get("run_id", "unknown")
            t0 = time.perf_counter()
            # If summarizer is sync, offload to thread to keep event loop clean
            result = await asyncio.to_thread(self.summarizer.summarize, state)
            dt = time.perf_counter() - t0

            # Reset window after summarization
            stats = state.get("stats", {})
            stats["window_tokens"] = 0
            state["stats"] = stats

            logger.info(
                "chatagent.summary_complete",
                extra={"run_id": run_id, "latency_ms": dt * 1000.0},
            )
            return result

        return _summarize

    def _build_should_summarize(self):
        async def should_summarize(state: AgentState) -> dict:
            stats = state.get("stats", {})
            window = stats.get("window_tokens", 0)
            if window >= self.config.max_tokens_before_summary:
                logger.info(
                    "chatagent.trigger_summarize",
                    extra={"run_id": (state.get("metadata") or {}).get("run_id", "unknown"),
                           "window_tokens": window},
                )
                return {"summarize_decision": "yes"}
            return {"summarize_decision": "no"}

        return should_summarize

    def _build_route_decision(self):
        def route_decision(state: AgentState):
            return "summarize" if state.get("summarize_decision") == "yes" else END
        return route_decision

    def _build_graph(self):
        g = StateGraph(AgentState)

        gen = self._build_generate_node()
        sum_ = self._build_summarize_node()
        should_summarize = self._build_should_summarize()

        g.add_node("generate", gen, cache_policy=CachePolicy(ttl=200))
        g.add_node("summarize", sum_, cache_policy=CachePolicy(ttl=200))
        g.add_node("should_summarize", should_summarize)

        g.add_edge(START, "generate")
        g.add_edge("generate", "should_summarize")
        g.add_conditional_edges(
            "should_summarize",
            self._build_route_decision(),
            {"summarize": "summarize", END: END},
        )
        g.add_edge("summarize", END)

        cache_backend = InMemoryCache()

        return g.compile(name="ChatAgent", cache=cache_backend)


# ============================================================================
# LangGraph Server Exports
# ============================================================================

def _create_default_agent():
    """Create default chat agent for LangGraph server."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    config = ChatAgentConfig(
        model_name=os.getenv("MODEL_NAME") or os.getenv("CHAT_MODEL_NAME"),
        base_url=os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL"),
        messages_to_keep=int(os.getenv("CHAT_MESSAGES_TO_KEEP", "5")),
        max_tokens_before_summary=int(os.getenv("CHAT_MAX_TOKENS_BEFORE_SUMMARY", "4000")),
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0.5")),
    )
    agent = ChatAgent(config)
    # keep lazy build by default; server will build on first request
    return agent.ensure_built()

# Export for LangGraph server
chatagent = _create_default_agent()
