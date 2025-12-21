import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

load_dotenv()
# Konfiguration
# Lade Umgebungsvariable aus .env für OPENAI_API_KEY
load_dotenv(verbose=True)
# Prüfen ob OPENAI_API_KEY gesetzt ist
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY wurde nicht gefunden in .env Datei")

else:
    print("OPENAI_API_KEY wurde gefunden in .env Datei")

    # Jetzt wird das LLM initialisiert. Wir verwenden gpt-4o für besseres Reasing.
    # Eine niedrige temperature wird verwendet für mehr deterministischen Output.
    try:
        llm = ChatOpenAI(model="gpt-4-turbo")
        print(f"LLM erfolgreich geladen: {llm.model_name}")
    except Exception as e:
        print(f"Fehler beim laden des LLM: {e}")
        llm = None

planner_writer_agent = Agent(
    role='Article Planner and Writer',
    goal='Plan and then write a concise, engaging summary on a specified topic.',
    backstory=(
        'You are an expert technical writer and content strategist.'
        'Your strength lies in creating a clear, actionable plan before writing'
        'ensuring the final summary is both informative and easy to digest'
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

topic = "The importance of Reinforcement Learning in AI"

high_level_task = Task(
    description=(
        f"1. Create a bullet-point plan for a summary on the topic: '{topic}'.\n"
        f"2. Write the summary based on your plan, keeping it around 200 words."
    ),
    expected_output=(
        "A final report containing two distinct sections:\n\n"
        "### Plan\n"
        "- A bulleted list outlining the main points of the summary. \n\n"
        "### Summary\n"
        "- A concise and well-structured summary ot the topic."
    ),
    agent=planner_writer_agent,
)

crew = Crew(
    agents=[planner_writer_agent],
    tasks=[high_level_task],
    process=Process.sequential,
)

print("## Running the planning and writing task ##")
result = crew.kickoff()
print("\n\n---\n## Task Result ##\n---")
print(result)
