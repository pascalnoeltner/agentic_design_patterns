import os
import asyncio
import nest_asyncio
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

# --- Setup ---
load_dotenv(verbose=True)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY wurde nicht gefunden in .env Datei")
print("OPENAI_API_KEY wurde gefunden in .env Datei")

llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# --- Tool ---
@tool
def suche_information(anfrage: str) -> str:
    """Simulierte Suche nach Informationen zu einer Anfrage."""
    print(f"\n--- Tool aufgerufen: suche_information | Anfrage: {anfrage} ---")

    simulierte_ergebnisse = {
        "Wetter in London": "Das Wetter in London ist momentan bewölkt mit einer Temperatur von 15° C",
        "Hauptstadt von Frankreich": "Die Hauptstadt von Frankreich ist Paris",
        "Bevölkerung der Erde": "Die geschätzte Bevölkerung der Erde ist etwa 8 Milliarden Menschen",
        "Höchster Berg": "Der höchste Berg ist der Mount Everest",
    }

    ergebnis = simulierte_ergebnisse.get(anfrage, f"Simuliertes Suchergebnis für '{anfrage}': Keine Information gefunden")
    print(f"--- TOOL Ergebnis: {ergebnis} ---")
    return ergebnis

# --- Agent (v1) ---
agent = create_agent(
    model=llm,
    tools=[suche_information],
    system_prompt=(
        "Du bist ein hilfreicher Assistent. "
        "Wenn dir Fakten fehlen oder du Informationen nachschlagen sollst, "
        "nutze das Tool 'suche_information'. "
        "Gib am Ende eine klare Antwort."
    ),
)

async def run_agent(anfrage: str):
    print(f"\n--- Agent mit Anfrage: {anfrage} ---")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": anfrage}]}
    )

    # In v1 ist 'result' typischerweise ein State mit messages
    # Wir geben die letzte Message aus:
    last_msg = result["messages"][-1]
    print("\n--- Finale Agent Antwort ---")
    print(last_msg.content)

async def main():
    await asyncio.gather(
        run_agent("Was ist die Hauptstadt von Frankreich?"),
        run_agent("Suche bitte: Hauptstadt von Frankreich"),
        run_agent("Wetter in London"),
    )

nest_asyncio.apply()
asyncio.run(main())
