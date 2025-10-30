from pathlib import Path
from langchain_community.utilities import SearxSearchWrapper

from websearch.agent import _create_default_agent
from websearch.config import SearchState

def load_queries() -> list[str]:
    queries = []
    with open("data/websearch-eval-queries.txt", "r") as f:
        queries = [line.strip().rstrip('\n') for line in f.readlines()]

    return queries

def search(query: str, k: int):
    search_call = SearxSearchWrapper(searx_host="http://192.168.30.100:8095")

    return (query, search_call.results(query=query, num_results=k))


async def main_async():
    agent = _create_default_agent()
    queries = load_queries()

    results = []
    for query in queries:
        state = SearchState(query=query, categories=None, results=None, summary=None, lang=None, query_en=None)
        res = await agent.ainvoke(state)
        results.append(res)

    print(results)


def main():
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()