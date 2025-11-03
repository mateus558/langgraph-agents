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

for chunk in remote_graph.stream(input, stream_mode="values"):
    print(chunk)

