import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Konfiguration
# Lade Umgebungsvariable aus .env für OPENAI_API_KEY
load_dotenv()
# Prüfen ob OPENAI_API_KEY gesetzt ist
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY wurde nicht gefunden in .env Datei")
else:
    print("OPENAI_API_KEY wurde gefunden in .env Datei")

# Jetzt wird das LLM initialisiert. Wir verwenden gpt-4o für besseres Reasing.
# Eine niedrige temperature wird verwendet für mehr deterministischen Output.
try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    print(f"LLM erfolgreich geladen: {llm.model_name}")
except Exception as e:
    print(f"Fehler beim laden des LLM: {e}")

def run_reflection_loop():
    """
    Hier demonstrieren wir eine mehrstufige AI Reflektion Schleife
    um progessiv eine Python Funktion zu verbessern.

    """

    aufgabe_prompt = """
    Deine Aufgabe ist es, es Python Funktion namens 'calculate_fibonacci'
    zu erstellen.
    Die Funktion sollte folgendes tun:
    1. Zwei Fibbonacci Zahlen aus der Fibonacci Reihe als Startwert akzeptieren.
    2. Wenn keine angegeben, n1 = 1 und n2 = 1 verwenden.
    3. Die Fibbonaci Reihe berechnen bis zu einem angegebenen Wert.
    4. Behandle Fehler: Erzeuge ein ValueError wenn die angegeben Startwerte keine Fibbonacci Werte sind.
    """
    # Die Reflektions Schleife:
    max_wiederholung = 3
    aktueller_code = ""

    # Wir werden eine Ausgabe Historie erzeugen um den Context in jedem Schritt zu speichern.

    ausgabe_historie = [HumanMessage(content=aufgabe_prompt)]
    for i in range(max_wiederholung):
        print("\n" + "="*25 + f"REFLEKTION SCHLEIFE: WIEDERHOLUNG {i + 1}" + "="*25)
        # --- 1. ERZEUGE / VERFEINERE
        # In der ersten Wiederholung wird das Ergebnis erzeugt, in den weiteren
        # Wiederholungen verfeinert.
        if i == 0:
            print("\n>>> RUNDE 1: ERZEUGE initialen Code...")
            # Die erste Nachricht ist nur der Aufgaben Prompt
            antwort = llm.invoke(ausgabe_historie)
            aktueller_code = antwort.content
        else:
            print("\n>>> RUNDE 1: VERFEINERE code basierend auf vorhergehender Kritik...")
            # Die Nachricht enhält nun die Aufgabe,
            # den letzten Code, und die letzte Kritik
            # Wir weisen das Model an die Kritik anzuwenden
            ausgabe_historie.append(HumanMessage(content="Bitte verfeinere den Code unter zur Hilfe nahme der beigefügten Kritik."))
            antwort = llm.invoke(ausgabe_historie)
            aktueller_code = antwort.content
        print("\n--- Erzeugter Code (v" + str(i + 1) + ") ---\n" + aktueller_code)
        ausgabe_historie.append(antwort) # Füge den generierten Code zur Historie hinzu
        # ---- 2. REFLEKTIERE
        print("\n>>> RUNDE 2: REFLEKTIERE den generierten Code...")
        # Erzeuge einen speziellen Prompt für den Reflektor Agent
        # Dieser frägt das Model als Senior Code reviewer zu handeln
        reflector_prompt = [
            SystemMessage(content="""
                Du bist ein Senior Software Engineer und ein Experte in Python.
                Deine Rolle ist es den Code einer gründlichen Überprüfung zu unterziehen.
                Überprüfe kritisch den gegebenen Python Code zur Original Aufgaben Stellung.
                Schaue nach Fehler, Syntax, fehlender Klammern und Stellen zur Verbesserung.
                Wenn der Code perfekt ist und allen Anforderungen genügt, antworte mit der
                einzigen Phrase 'CODE_IS_PERFECT'. Andernfalls stelle eine ausführliche 
                Liste mit Verbesserungsvorschlägen zur Verfügung.
                """),
                HumanMessage(content=f"Original     Aufgabe:\n{aufgabe_prompt}\n\n Code zur Überprüfung:\n{aktueller_code}")
            ]
        kritik_antwort = llm.invoke(reflector_prompt)
        kritik = kritik_antwort.content
        # 3. STOP KRITERIUM
        if "CODE_IS_PERFECT" in kritik:
            print("\n ----KRITIK ---- \nKeine weitere Kritik gefunden. Code ist den Anforderungen genügend")
            break
        print("\n---- KRITIK: ----\n" + kritik)
        # Füge die Kritik zur Historie für die nächste Verfeinerungsschleife hinzu
        ausgabe_historie.append(HumanMessage(content=f"Kritik des letzten Codes:\n{kritik}"))

    print("\n" + "="*30 + "ENDERGEBNIS" + "="*30)
    print("\nZuletzt verbesserter Code nach dem letzten Reflektions Prozess:\n")
    print(aktueller_code)

if __name__ == "__main__":
    run_reflection_loop()

