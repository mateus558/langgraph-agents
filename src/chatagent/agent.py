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

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from chatagent.config import AgentState
from chatagent.summarizer import OllamaSummarizer
from chatagent.prompts import ASSISTANT_SYSTEM_PROMPT
from core.time import build_chat_clock_vars, resolve_timezone
from config import get_settings
from core.contracts import AgentMixin, ModelFactory
from utils.messages import TokenEstimator

from chatagent.weather.tool import weather_tool

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
    default_stream: bool = False
    tz_name: str | None = None


class ChatAgent(AgentMixin):
    config: ChatAgentConfig

    def __init__(self, config: ChatAgentConfig | None = None):
        super().__init__()
        self.config = config or ChatAgentConfig()

        settings = get_settings()
        if self.config.model_name is None:
            self.config.model_name = settings.model_name
        if self.config.base_url is None:
            self.config.base_url = settings.base_url
        if self.config.tz_name is None:
            self.config.tz_name = getattr(settings, "timezone", "America/Sao_Paulo")

        self.summarizer = OllamaSummarizer(
            model_id=self.config.model_name or get_settings().model_name,
            base_url=self.config.base_url,
            k_tail=self.config.messages_to_keep,
        )

        self._tokenizer = TokenEstimator()

        tz_in = self.config.tz_name or getattr(settings, "timezone", "America/Sao_Paulo")
        tz, norm, fell_back = resolve_timezone(tz_in)
        self.tz = tz
        self.config.tz_name = norm
        if fell_back:
            logger.warning("chatagent.timezone_fallback", extra={"requested_tz": tz_in, "used": norm})

        self.build()

    def _build_langchain_agent(self):
        """
        Build a LangChain v1 agent with dynamic system prompt middleware.
        The agent encapsulates the model + tool-calling loop. 
        """
        model = ModelFactory.create_chat_model(
            model_name=self.config.model_name or "",
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
        )
        tools = [weather_tool]

        @dynamic_prompt
        def clock_prompt(request: ModelRequest) -> str:
            """Inject dynamic system prompt using runtime context."""
            clock = request.runtime.context.get("clock", "") # type: ignore
            summary = request.runtime.context.get("summary", "") # type: ignore
            return ASSISTANT_SYSTEM_PROMPT.format_system(clock=clock, summary=summary) or ""

        # Attach middleware; no static system_prompt here.
        agent = create_agent(
            model=model,
            tools=tools,
            middleware=[clock_prompt],
        )
        return agent

    def _build_generate_node(self):
        agent = self._build_langchain_agent()

        async def _generate(state: AgentState):
            """Generate node with optional token streaming.

            When state["stream"] (or default_stream) is True, this function will
            yield incremental AIMessageChunk updates to the "messages" channel.
            At the end, it returns the final assembled AIMessage and stats.
            """
            run_id = (state.get("metadata") or {}).get("run_id", "unknown")
            msgs = state.get("messages", [])

            # Build runtime context for the middleware
            clock = build_chat_clock_vars(self.tz)["clock"]
            context = {"clock": clock, "summary": state.get("summary") or ""}

            # Token estimate (input) - approximate
            input_tokens = self._tokenizer.count_messages(msgs)

            do_stream = bool(state.get("stream", self.config.default_stream))

            t0 = time.perf_counter()
            out_text = ""
            try:
                if do_stream:
                    # Stream from the underlying runnable and forward message chunks
                    # Note: Many LC runnables support astream; if not, we fall back.
                    async for ev in agent.astream({"messages": msgs}, context=context):  # type: ignore[attr-defined]
                        # The langchain agent returns a dict with messages when completed,
                        # but during streaming we expect AIMessageChunk events.
                        # Normalize a few common shapes here.
                        content = getattr(ev, "content", None)
                        if isinstance(content, str):
                            # Likely an AIMessageChunk or similar message-like object
                            out_text += content
                            yield {"messages": [ev]}
                        elif isinstance(ev, dict) and "messages" in ev:
                            # Some implementations stream structured events; forward if they are chunks
                            for m in ev["messages"]:
                                m_content = getattr(m, "content", None)
                                if isinstance(m_content, str):
                                    out_text += m_content
                                    yield {"messages": [m]}
                        # Ignore other non-text events in this simplified bridge

                    # After stream completes, build final response from accumulated text
                    resp = AIMessage(content=out_text)
                else:
                    # Non-streaming path: single final invocation
                    result = await agent.ainvoke({"messages": msgs}, context=context)  # type: ignore[attr-defined]
                    final_msgs = result.get("messages") if isinstance(result, dict) else None
                    if final_msgs:
                        last = final_msgs[-1]
                        resp = last if isinstance(last, AIMessage) else AIMessage(content=str(last))
                    else:
                        # Fallback: coerce result to text
                        resp = AIMessage(content=str(result))
                    out_text = str(resp.content)
            finally:
                dt = time.perf_counter() - t0

            output_tokens = self._tokenizer.count_text(out_text)
            step_total = input_tokens + output_tokens

            stats = state.get("stats", {})
            stats["last_step_tokens"] = step_total
            stats["window_tokens"] = stats.get("window_tokens", 0) + step_total
            state["stats"] = stats

            logger.info(
                "chatagent.agent_generate_complete",
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
            # In async generator nodes, yield the final update and return without value
            yield {"messages": [resp], "history": new_history, "stats": stats}
            return

        return _generate

    def _build_summarize_node(self):
        async def _summarize(state: AgentState):
            run_id = (state.get("metadata") or {}).get("run_id", "unknown")
            t0 = time.perf_counter()
            result = await asyncio.to_thread(self.summarizer.summarize, state)
            dt = time.perf_counter() - t0

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
                    extra={
                        "run_id": (state.get("metadata") or {}).get("run_id", "unknown"),
                        "window_tokens": window,
                    },
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
    return agent.ensure_built()

# Export for LangGraph server
chatagent = _create_default_agent()
