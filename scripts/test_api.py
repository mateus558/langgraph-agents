from langgraph_sdk import get_client, get_sync_client
from langgraph.pregel.remote import RemoteGraph

url = "http://192.168.30.100:8123"
client = get_client(url=url)
sync_client = get_sync_client(url=url)

# Using graph name (uses default assistant)
graph_name = "chatagent"
remote_graph = RemoteGraph(graph_name, client=client, sync_client=sync_client)

input = {
    "messages": [{"role": "user", "content": "what's the weather in sf"}],
    "stream": True,
}

buffer = ""
for chunk in remote_graph.stream(input, stream_mode="messages"):
    # Expect chunks like {"messages": [{"content": "..."}, ...]}
    try:
        msgs = None
        if isinstance(chunk, dict):
            msgs = chunk.get("messages")
        elif hasattr(chunk, "get"):
            # Some SDKs return a Mapping-like object
            msgs = chunk.get("messages")  # type: ignore[attr-defined]

        if msgs:
            for m in msgs:
                content = None
                if isinstance(m, dict):
                    content = m.get("content")
                else:
                    content = getattr(m, "content", None)

                if isinstance(content, str) and content:
                    # Print only the new suffix to avoid reprinting the final full message
                    new_part = content[len(buffer):] if content.startswith(buffer) else content
                    if new_part:
                        print(new_part, end="", flush=True)
                        buffer += new_part
        else:
            # Fallback: print a compact representation for unexpected event shapes
            print(str(chunk))
    except Exception:
        print(str(chunk))

print()  # newline after stream completes

