import asyncio
from langgraph_sdk import get_client

async def main():
    my_token = "123" # In practice, you would generate a signed token with your auth provider
    client = get_client(
        url="http://192.168.30.100:8123",
        api_key=my_token,
    )
    threads = await client.assistants.search()
    return threads

if __name__ == "__main__":
    threads = asyncio.run(main())
    print(threads)