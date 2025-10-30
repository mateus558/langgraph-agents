import asyncio
from langgraph_sdk import get_client

async def main():
    my_token = "123" # In practice, you would generate a signed token with your auth provider
    client = get_client(
        url="http://localhost:2024",
        api_key=my_token,
    )
    threads = await client.threads.search()
    return threads

if __name__ == "__main__":
    threads = asyncio.run(main())
    print(threads)