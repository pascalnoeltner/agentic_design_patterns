import os
import asyncio
#from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableParallel

from dotenv import load_dotenv

load_dotenv()

try:
    llm = ChatOpenAI()
except Exception as e:
    print(f"Fehler beim laden des LLM: {e}")
    llm = None

if llm:
    print(f"Geladen: " + llm.model_name)

zusammenfassung_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Fasse das folgende Thema kurz und bündig zusammen:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

fragen_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Erstelle drei interessante Fragen über das folgende Thema:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

eckpunkte_chain: Runnable = (
    ChatPromptTemplate.from_messages([
        ("system", "Identifiziere 5-10 wichtige Eckpunkte über das folgende Thema, getrennt durch Komma:"),
        ("user", "{topic}")
    ])
    | llm
    | StrOutputParser()
)

### Jetzt bauen wir den Parallelisierungs + Synthese Chain
# 1. Definiere die Blöcke die parallel laufen sollen. Die Ergebnisse werden zusammen
# mit dem ursprünglichen Thema in den nächsten Schritt gespeist.

haupt_chain = RunnableParallel(
    {
        "Zusammenfassung": zusammenfassung_chain,
        "Fragen": fragen_chain,
        "Eckpunkte": eckpunkte_chain,
        "topic": RunnablePassthrough(),
    }
)

# 2. Definiere den endgültigen Synthese Prompt der die parallelen Ergebnisse kombiniert.

synthese_prompt = ChatPromptTemplate.from_messages([
    ("system", """Basierend auf den folgenden Informationen:
    Zusammenfassung: {Zusammenfassung}
    Fragen: {Fragen}
    Eckpunkte: {Eckpunkte}
    Erstelle eine zusammenfassende Antwort."""),
    ("user", "Original Thema: {topic}")
])

# 3. Konstruiere den kompletten Chain by piping the parallel results directly
gesamter_parallel_chain = haupt_chain | synthese_prompt | llm | StrOutputParser()

# Run the chain
async def run_parallel_example(thema: str) -> None:
    """

    :param thema:
    :return:
    """
    if not llm:
        print("Fehler beim laden des LLM")
        return
    print(f"\n--- Running Parallel LangChain Example für Thema: '{thema}' ----")
    try:
        antwort = await gesamter_parallel_chain.ainvoke(thema)
        print("\n ---- Endergebnis ----")
        print(antwort)
    except Exception as e:
        print(f"Es ist ein Fehler aufgetreten: {e}")

if __name__ == "__main__":
    test_thema = "Hochdosis Folat und B12 für Cellular Reprogramming (Stammzellerneuerung)"
    asyncio.run(run_parallel_example(test_thema))
