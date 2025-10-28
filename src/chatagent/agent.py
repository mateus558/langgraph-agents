import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langgraph.graph import END, START, StateGraph

from chatagent.config import AgentState
from chatagent.summarizer import OllamaSummarizer
from config import get_settings
from core.contracts import AgentProtocol
from utils.messages import TokenEstimator, coerce_message_content

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    messages_to_keep: int = 5
    max_tokens_before_summary: int = 4000
    model_name: str | None = None
    base_url: str | None = None
    temperature: float = 0.5
    num_ctx: int = 131072


class ChatAgent(AgentProtocol):
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

        # Load settings from environment if not provided in config
        settings = get_settings()
        if self.config.model_name is None:
            self.config.model_name = settings.model_name
        if self.config.base_url is None:
            self.config.base_url = settings.base_url

        self.summarizer = OllamaSummarizer(
            model_id=self.config.model_name or get_settings().model_name,
            base_url=self.config.base_url,
            k_tail=self.config.messages_to_keep,
        )

        # Token counters for sliding window
        self.token_count = 0  # tokens accumulated since last summary
        self.last_token_count = 0  # last snapshot for logging
        self._tokenizer = TokenEstimator()

        self.graph = self._build_graph()

    def invoke(self, state: AgentState) -> Any:
        return self.graph.invoke(state)

    def get_mermaid(self) -> str:
        return str(self.graph.get_graph().draw_mermaid())

    def _build_generate_node(self):
        # Initialize model with config values or environment defaults
        model_provider = "ollama" if self.config.base_url else "openai"
        kwargs = {"num_ctx": self.config.num_ctx} if self.config.num_ctx else {}

        model = init_chat_model(
            self.config.model_name,
            model_provider=model_provider,
            temperature=self.config.temperature,
            base_url=self.config.base_url,
            kwargs=kwargs,
        )

        sys_tmpl = (
            "You are a helpful AI assistant. Be friendly and professional. "
            "Avoid long responses unless needed and reply in the user's language.\n"
            "Conversation Summary: {summary}\n"
        )

        def _generate(state: AgentState):
            msgs = state.get("messages", [])
            sys = SystemMessage(content=sys_tmpl.format(summary=state.get("summary") or ""))

            # Estimate input tokens
            input_tokens = self._tokenizer.count_messages([sys, *msgs])

            t0 = time.perf_counter()
            resp = model.invoke([sys, *msgs])
            dt = time.perf_counter() - t0

            # Estimate output tokens
            out_text = coerce_message_content(resp)
            output_tokens = self._tokenizer.count_text(out_text)

            # Update counters
            step_total = input_tokens + output_tokens
            self.last_token_count = step_total
            self.token_count += step_total

            logger.info(
                "Estimated tokens - input: %d, output: %d, step_total: %d, window_total: %d",
                input_tokens,
                output_tokens,
                step_total,
                self.token_count,
            )
            logger.debug("generate_model.invoke took %.3fs", dt)

            # Partial update: append only the response; update history explicitly
            new_history = (state.get("history", []) + ([msgs[-1]] if msgs else []) + [resp])
            return {
                "messages": [resp],
                "history": new_history,
            }

        return _generate

    def _build_summarize_node(self):
        def _summarize(state: AgentState):
            result = self.summarizer.summarize(state)
            # Reset the window after summarization
            logger.info("Summary done. Resetting token window (prev=%d).", self.token_count)
            self.token_count = 0
            return result

        return _summarize

    def _build_should_summarize(self):
        def should_summarize(state: AgentState) -> dict:
            if self.token_count >= self.config.max_tokens_before_summary:
                logger.info("Triggering summarization due to token limit (window_total >= max).")
                return {"summarize_decision": "yes"}
            return {"summarize_decision": "no"}

        return should_summarize

    def _build_route_decision(self):
        def route_decision(state: AgentState):
            if state["summarize_decision"] == "yes":
                return "summarize"
            else:
                return END

        return route_decision

    def _build_graph(self):
        g = StateGraph(AgentState)

        gen = self._build_generate_node()
        sum_ = self._build_summarize_node()
        should_summarize = self._build_should_summarize()

        g.add_node("generate", gen)
        g.add_node("summarize", sum_)
        g.add_node("should_summarize", should_summarize)

        g.add_edge(START, "generate")
        g.add_edge("generate", "should_summarize")
        g.add_conditional_edges(
            "should_summarize", self._build_route_decision(), {"summarize": "summarize", END: END}
        )
        g.add_edge("summarize", END)

        return g.compile()


# ============================================================================
# LangGraph Server Exports
# ============================================================================
# These module-level exports are required for LangGraph server deployment.
# The server discovers graphs via langgraph.json configuration.
# Configuration is loaded from environment variables at runtime.
# ============================================================================


def _create_default_agent():
    """Create default chat agent for LangGraph server.

    This function is called at module import time to create the graph
    that the LangGraph server will expose. Configuration is loaded from
    environment variables.

    Returns:
        Compiled LangGraph graph ready for deployment.
    """
    import os

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available

    # Load all configuration from environment
    config = AgentConfig(
        model_name=os.getenv("MODEL_NAME") or os.getenv("CHAT_MODEL_NAME"),
        base_url=os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL"),
        messages_to_keep=int(os.getenv("CHAT_MESSAGES_TO_KEEP", "5")),
        max_tokens_before_summary=int(os.getenv("CHAT_MAX_TOKENS_BEFORE_SUMMARY", "4000")),
        temperature=float(os.getenv("CHAT_TEMPERATURE", "0.5")),
        num_ctx=int(os.getenv("CHAT_NUM_CTX", "131072")),
    )
    agent = ChatAgent(config)
    return agent.graph


# Export for LangGraph server (referenced in langgraph.json)
chatagent = _create_default_agent()
