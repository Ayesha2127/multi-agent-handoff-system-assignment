from agents import Runner, Agent,OpenAIChatCompletionsModel,AsyncOpenAI,RunConfig
import os
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key = gemini_api_key,
    base_url= "https://generativelanguage.googleapis.com/v1beta/openai/"
)


model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

config = RunConfig(
    model= model,
    model_provider=external_client,
    tracing_disabled=True,    
)

agent = Agent(
    name= "Frontend Expert",
    instructions="You are a frontend expert."
)

runner = Runner.run_sync(
    agent,
    input="Hello,How are you?",
    run_config=config,
)

print(runner.final_output)
   