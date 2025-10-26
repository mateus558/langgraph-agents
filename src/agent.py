from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent

from dotenv import load_dotenv
load_dotenv()  

model = init_chat_model(
        model="openai:gpt-5-nano",
        temperature=0,
        max_retries=3,
        streaming=True,
    )

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You're a helpful assistant."
)



def main():
    print("Hello from agents!")


if __name__ == "__main__":
    main()
