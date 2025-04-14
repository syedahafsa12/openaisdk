
import chainlit as cl
import os
from dotenv import load_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini Client Setup
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Summarizer Agent
summarizer_agent = Agent(
    name="MeetingSummarizer",
    instructions="""
You're a professional meeting assistant.
Read unstructured meeting notes and return:
1. A bullet-point summary.
2. Action items with responsible people and deadlines (if mentioned).
"""
)

# General Agent for Casual or Small Talk
general_agent = Agent(
    name="FriendlyResponder",
    instructions="""
You're a helpful and friendly chatbot.
Reply to general greetings or small talk like 'hi', 'gn', 'how are you?', etc., in a casual and friendly tone.
"""
)

# Utility function: Check if it's a real meeting note
def is_meeting_note(text: str) -> bool:
    return len(text.strip().split()) > 10 or "\n" in text

@cl.on_message
async def handle_message(message: cl.Message):
    user_input = message.content.strip()

    if not user_input:
        await cl.Message(content="❗Please type something.").send()
        return

    try:
        if is_meeting_note(user_input):
            response = Runner.run_sync(
                summarizer_agent,
                f"Summarize the following meeting notes:\n{user_input}",
                run_config=config
            )
            await cl.Message(
                content=f"✅ **Summary:**\n\n{response.final_output.strip()}"
            ).send()
        else:
            response = Runner.run_sync(
                general_agent,
                user_input,
                run_config=config
            )
            await cl.Message(
                content=response.final_output.strip()
            ).send()

    except Exception as e:
        await cl.Message(
            content=f"⚠️ Error: {str(e)}"
        ).send()
