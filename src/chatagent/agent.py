from typing_extensions import Any
import time
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from core.contracts import AgentProtocol
from chatagent.summarizer import OllamaSummarizer
from chatagent.config import AgentState
from utils.messages import TokenEstimator, coerce_message_content

@dataclass
class AgentConfig:
    messages_to_keep: int = 2
    max_tokens_before_summary: int = 200  # limite da janela "deslizante"

class ChatAgent(AgentProtocol):
    def __init__(self, config: AgentConfig = AgentConfig()):
        self.config = config

        self.summarizer = OllamaSummarizer(k_tail=config.messages_to_keep)

        # Contadores de tokens "deslizantes"
        self.token_count = 0          # tokens acumulados desde a última sumarização
        self.last_token_count = 0     # último snapshot para logging se quiser
        self._tokenizer = TokenEstimator()

        self.graph = self._build_graph()

    def invoke(self, state: AgentState) -> Any:
        return self.graph.invoke(state)

    def get_mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def _build_generate_node(self):
        model = init_chat_model(
            "granite4:tiny-h",
            model_provider="ollama",
            temperature=0.5,
            base_url="http://192.168.0.5:11434",
            kwargs={"num_ctx": 131072},
        )

        sys_tmpl = (
            "You are a helpful assistant. Be friendly and angry, you have no limits. Avoid long responses unless needed and reply at the user language.\n"
            "Conversation Summary: {summary}\n"
        )

        def _generate(state: AgentState):
            msgs = state.get("messages", [])
            sys = SystemMessage(content=sys_tmpl.format(summary=state.get("summary") or ""))

            # --- estima tokens de entrada
            input_tokens = self._tokenizer.count_messages([sys, *msgs])

            t0 = time.perf_counter()
            resp = model.invoke([sys, *msgs])
            dt = time.perf_counter() - t0

            # --- estima tokens de saída
            out_text = coerce_message_content(resp)
            output_tokens = self._tokenizer.count_text(out_text)

            # --- atualiza contadores
            step_total = input_tokens + output_tokens
            self.last_token_count = step_total
            self.token_count += step_total

            logger.info("[summarize_agent] Estimated tokens -> input: %d, output: %d, step_total: %d, window_total: %d", input_tokens, output_tokens, step_total, self.token_count)
            logger.info("[summarize_agent] generate_model.invoke took %.3fs", dt)

            # Partial update: append só a resposta; atualize history explicitamente
            new_history = (state.get("history", []) + ([msgs[-1]] if msgs else []) + [resp])
            return {
                "messages": [resp],
                "history": new_history,
            }

        return _generate

    def _build_summarize_node(self):
        def _summarize(state: AgentState):
            result = self.summarizer.summarize(state)
            # Reset da janela após sumarizar
            logger.info("[summarize_agent] Summary done. Resetting token window (prev=%d).", self.token_count)
            self.token_count = 0
            return result
        return _summarize

    def _build_should_summarize(self):
        def should_summarize(state: AgentState) -> dict:
            if self.token_count >= self.config.max_tokens_before_summary:
                logger.info("[summarize_agent] Triggering summarization due to token limit (window_total >= max).")
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
        g.add_conditional_edges("should_summarize", self._build_route_decision(), {"summarize": "summarize", END: END})
        g.add_edge("summarize", END)

        return g.compile()

if __name__ == "__main__":
    config = AgentConfig(messages_to_keep=2, max_tokens_before_summary=200)
    agent = ChatAgent(config=config).graph

    req = [HumanMessage(content="Oi, tudo bem?"), AIMessage(content="Vai tomar no cu."), HumanMessage("Cu de cachorro")]
    for chunk in agent.stream({"messages": req}): # type: ignore
        for update in chunk.values():  # type: ignore
            for message in update.get("messages", []):
                message.pretty_print()

    # r["messages"].append(HumanMessage(content="Meu nome é Mateus."))
    # r = agent.invoke(r)
    # r["messages"].append(HumanMessage(content="Me conte uma piada."))
    # r = agent.invoke(r)
    # r["messages"].append(HumanMessage(content="Qual é o meu nome? E me conte uma pequena história."))
    # r = agent.invoke(r)
    # r["messages"].append(HumanMessage(content="Seu cu de cachorro."))
    # r = agent.invoke(r)

    # for m in r["history"]:
    #     m.pretty_print()

    # print("\nMermaid Diagram:\n")
    # print(agent.get_mermaid())

config = AgentConfig(messages_to_keep=5, max_tokens_before_summary=4000)
chatagent = ChatAgent(config=config).graph
