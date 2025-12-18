from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

from dotenv import load_dotenv
load_dotenv()

try:
    llm = ChatOpenAI()
    print(f"LLM initialisiert: {llm.model_name}")
except Exception as e:
    print(f"Fehler beim initialisieren des LLM: {e}")
    llm = None

# Hier werden Agent Handler definiert: -------------------------------------------------------------------------------#
def beschwerde_handler(request: str) -> str:
    """Simuliere wie der Beschwerde Handler eine Anfrage bearbeitet"""
    print("\n--- ÜBERGABE AN BESCWERDE HANDLER---")
    return f"Beschwerde Handler hat folgende Anfrage bearbeitet: {request}"

def lob_handler(request: str) -> str:
    """Simuliere wie der Lob Handler eine Anfrage bearbeitet"""
    print("\n--- ÜBERGABE AN LOB HANDLER---")
    return f"Lob Handler hat folgende Anfrage bearbeitet: {request}"

def neutral_handler(request: str) -> str:
    """Simuliere wie der Neutral Handler eine Anfrage bearbeitet"""
    print("\n--- ÜBERGABE AN NEUTRAL HANDLER---")
    return f"Neutral Handler hat folgende Anfrage bearbeitet: {request}"
########################################################################################################################
# Dieser Chain entscheidet welcher Handler benutzt wird.

coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analysiere die Eingabe des Benutzers und bestimme welcher spezialisierte Handler benutzt werden soll
                    - Wenn die Eingabe ist überwiegend zufrieden und voll des Lobes 'lob'.
                    - Wenn die Eingabe ist überwiegend unzufrieden und beschwert sich 'beschwerde'.
                    - Wenn die Eingabe keine klar Tendenz von unzufrieden oder zufrieden hat 'neutral_handler'.
                    Gib nur EIN Word aus: 'lob', 'beschwerde', oder 'neutral'."""),
                    ("user", "{request}")
    ])
# Jetzt wird geprüft ob das LLM verfügbar ist, wenn ja wird der coordinator_router_chain erstellt
if llm:
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

# Jetzt wird die Delegationslogik erstellt
branches = {
    "lob": RunnablePassthrough.assign(output=lambda x: lob_handler(x['request']['request'])),
    "beschwerde": RunnablePassthrough.assign(output=lambda x: beschwerde_handler(x['request']['request'])),
    "neutral": RunnablePassthrough.assign(output=lambda x: neutral_handler(x['request']['request'])),
}
# Jetzt wird die RunnableBranch erstellt. Sie nimmt den Output der Router Chain entgegen
delegation_branch = RunnableBranch(
    (lambda x: x['decision'].strip() == 'lob', branches["lob"]),
    (lambda x: x['decision'].strip() == 'beschwerde', branches["beschwerde"]),
    (lambda x: x['decision'].strip() == 'neutral', branches["neutral"]),
    branches["neutral"],
)

coordinator_agent = {
    "decision": coordinator_router_chain,
    "request": RunnablePassthrough()
} | delegation_branch | (lambda x: x['output'])

def main():
    if not llm:
        print("Fehler beim initialisieren des LLM")
        return

    print("--- Teste mit einer schlechten Bewertung ---")
    test_a = "Wirklich der mieseste Urlaub den ich je erlebt habe!"
    ergebnis_a = coordinator_agent.invoke({"request": test_a})
    print(ergebnis_a)

    print("--- Teste mit einer guten Bewertung ---")
    test_b = "Das war der schönste Urlaub unseres Lebens!"
    ergebnis_b = coordinator_agent.invoke({"request": test_b})
    print(ergebnis_b)

    print("--- Teste mit einer neutralen Bewertung ---")
    test_c = "Es war nicht schlecht aber auch nicht wirklich gut."
    ergebnis_c = coordinator_agent.invoke({"request": test_c})
    print(ergebnis_c)

if __name__ == "__main__":
    main()

