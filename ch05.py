import os
import asyncio
import nest_asyncio
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool as langchain_tool
from langchain.agents import create_agent

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
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
        print(f"LLM erfolgreich geladen: {llm.model_name}")
    except Exception as e:
        print(f"Fehler beim laden des LLM: {e}")
        llm = None

# Definiere ein Tool
@langchain_tool
def suche_information(anfrage: str) -> str:
    """
    """
    print(f"\n --- Tool aufgerufen: suche_information mit Anfrage: {anfrage}---")

    simulierte_ergebnisse = {
        "Wetter in London": "Das Wetter in London ist momentan bewölkt mit einer Temperatur von 15° C",
        "Hauptstadt von Frankreich": "Die Hauptstadt von Frankreich ist Paris",
        "Bevölkerung der Erde": "Die geschätzte Bevölkerung der Erde ist etwa 8 Milliarden Menschen",
        "Höchster Berg": "Der höchste Berg ist der Mount Everest",
        "default": f"Simuliertes Suchergebnis für '{anfrage}': Keine Information gefunden",
    }

    ergebnis = simulierte_ergebnisse.get(anfrage, simulierte_ergebnisse["default"])

    print(f"--- TOOL Ergebnis: {ergebnis} ---")
    return ergebnis

if llm:
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", "Du bist ein hilfreicher Assistent"),
        ("human", "{input}"),
    ])

agent = create_agent(model=llm, system_prompt=agent_prompt)

async def run_agent_with_tool(anfrage: str):
    print(f"\n--- Agent mit Anfrage: {anfrage}---")

    try:
        antwort = await agent.ainvoke({"input": anfrage})
        print("\n Finale Agent Anwort ----")
        print(antwort["output"])
    except Exception as e:
        print(f"\nEs trat ein Fehler auf während der Ausführung des Agents: {e}")

async def main():
    tasks = [
        run_agent_with_tool("Was ist die Hauptstadt von Frankreich?"),
    ]
    await asyncio.gather(*tasks)

nest_asyncio.apply()
asyncio.run(main())
