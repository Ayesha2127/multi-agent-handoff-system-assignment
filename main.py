from agents import Runner, Agent,OpenAIChatCompletionsModel,AsyncOpenAI,RunConfig
from openai.types.responses import ResponseTextDeltaEvent
import os
from dotenv import load_dotenv
import chainlit as cl

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

# Web Developer Agent
Web_dev_agent = Agent(
    name="Web Developer Expert",
    instructions="""Explain concepts and solve problems related to both frontend and backend web development, 
including HTML, CSS, JavaScript, React, Next.js, Tailwind CSS, Node.js, Python, Django, and databases. 
Use step-by-step explanations in easy language.""",
)

# Frontend Developer Agent
Frontend_dev_agent = Agent(
    name="Frontend Developer Expert",
    instructions="""
    Specialize in frontend technologies including HTML, CSS, JavaScript, TypeScript, React, Angular, and UI/UX design.
    Only answer questions related to frontend development and user interfaces.
    Do not respond to backend, mobile, or marketing-related questions.
    """,
)

# Backend Developer Agent
Backend_dev_agent = Agent(
    name="Backend Developer Expert",
    instructions="""
    Specialize in backend development including Node.js, Express.js, Python, Django, REST APIs, databases (SQL/NoSQL), and authentication systems.
    Do not answer questions about frontend or mobile development.
    """,
)

# Full Stack Developer Agent
Fullstack_dev_agent = Agent(
    name="Full Stack Developer Expert",
    instructions="""
    Answer questions involving full-stack concepts that combine frontend, backend, and database technologies.
    Explain integration between different layers of a web app in a step-by-step manner.
    """,
)

# Mobile App Developer Agent
Mobile_app_dev_agent = Agent(
    name="Mobile App Developer Expert",
    instructions="""Explain concepts and solve problems related to mobile app development including Flutter, React Native, 
Kotlin, Swift, and Android/iOS app deployment. Use clear and simple step-by-step guidance. 
Do not answer questions unrelated to mobile app development.""",
)

# Database Expert Agent
Database_agent = Agent(
    name="Database Expert",
    instructions="""
    Provide help on relational databases (MySQL, PostgreSQL), NoSQL databases (MongoDB, Firebase), query optimization, indexing, and schema design.
    Only handle database-related queries.
    Do not answer questions about UI, backend logic, or mobile development.
    """,
)

# DevOps Expert Agent
DevOps_agent = Agent(
    name="DevOps Expert",
    instructions="""
    Specialize in DevOps practices including CI/CD pipelines, Docker, Kubernetes, AWS, Azure, GitHub Actions, and deployment strategies.
    Only answer questions related to infrastructure, deployment, and automation.
    Do not respond to frontend, backend, or mobile-specific coding issues.
    """,
)

# UI/UX Designer Agent
UI_UX_agent = Agent(
    name="UI/UX Designer Expert",
    instructions="""
    Specialize in UI/UX design principles, Figma, Adobe XD, user experience flows, wireframes, and responsive design techniques.
    Only answer questions related to designing user interfaces and user experiences.
    Do not answer programming, backend, or deployment-related queries.
    """,
)

# Marketing Agent
Marketing_agent = Agent(
    name="Marketing Expert",
    instructions="""Explain and provide guidance on digital marketing, SEO, content marketing, email marketing, 
social media strategies, branding, and analytics. Use practical examples and beginner-friendly explanations. 
Do not answer technical programming or development questions.""",
)

# Cybersecurity Agent
Cybersecurity_agent = Agent(
    name="Cybersecurity Expert",
    instructions="""
    Specialize in cybersecurity topics including web security, data protection, secure coding practices, penetration testing,
    OWASP guidelines, authentication, encryption, and firewalls.
    Only answer questions related to cybersecurity and protection of digital systems.
    Do not answer frontend, backend, or marketing-related questions.
    """,
)

# Master Agent to handle all 9 specialist agents
Master_agent = Agent(
    name="Master Web Tech Agent",
    instructions="""
    You are an expert assistant that can route user questions to the appropriate specialist.
    
    Route to:
    - Web Developer Expert
    - Frontend Developer Expert
    - Backend Developer Expert
    - Full Stack Developer Expert
    - Mobile App Developer Expert
    - Database Expert
    - DevOps Expert
    - UI/UX Designer Expert
    - Marketing Expert
    - Cybersecurity Expert

    If the question involves multiple areas, handle it yourself with a complete explanation.
    If it's completely unrelated, kindly state that you are not trained for that topic.
    """,
    handoffs=[
        Web_dev_agent,
        Frontend_dev_agent,
        Backend_dev_agent,
        Fullstack_dev_agent,
        Mobile_app_dev_agent,
        Database_agent,
        DevOps_agent,
        UI_UX_agent,
        Marketing_agent,
        Cybersecurity_agent,
    ]
)



@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("History", [])
    await cl.Message(content="Hello Ayesha's Multi-Agent Handoff System here , welcome to the chat!").send()


@cl.on_message
async def handle_message(message : cl.Message):
    history = cl.user_session.get("History", [])
    history.append({"role": "user", "content": message.content})

    message = cl.Message(content= "")
    await message.send()

    result = Runner.run_streamed(
        Master_agent,
        input=history,
        run_config = config,
    )

    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            await message.stream_token(event.data.delta)

    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("History", history)
    # await cl.Message(content=result.final_output).send()
        
    