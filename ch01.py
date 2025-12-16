import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Aus Sicherheitsgründen werden die Umgebungsvariablen aus einer env file gelesen

from dotenv import load_dotenv
from sqlalchemy.dialects.oracle.dictionary import all_mview_comments

load_dotenv()

# Der OPENAI_API_KEY muss in der .env file gespeichert sein

llm = ChatOpenAI()

# ---- 1. Prompt: Informationen Extrahieren ----
prompt_extract = ChatPromptTemplate.from_template(
    "Extrahiere die technischen Spezifikationen von dem folgenden Text:\n\n{text_input}"
)

# ---- 2. Prompt: Zu JSON konvertieren ----
prompt_transform = ChatPromptTemplate.from_template(
    "Transformiere die folgenden Spezifikationen in ein JSON object mit 'cpu', 'gpu', 'memory', 'storage' als Schlüssel:\n\n{spezifikationen}"
)

# ---- Die Kette mit LCEL bauen ----
# StrOutputParser() konvertiert den LLM Output zu einem simplen String.
extraction_chain = prompt_extract | llm | StrOutputParser()

ganzer_chain = (
    {"spezifikationen": extraction_chain}
    | prompt_transform
    | llm
    | StrOutputParser()
)

eingabe_text = """Das GIGABYTE AERO X16 1VH93DEC64AH ist ein Notebook, das alles kann! Der AMD Ryzen AI 7 HX 350 Prozessor mit überzeugender CPU- und NPU-Leistung unterstützt in Kombination mit 32 GB DDR5-Arbeitsspeicher produktive Workflows und rasantes Gaming. Als Speicher sind zwei jeweils 1 TB große M.2-PCIe-SSDs verbaut. Angetrieben von einer NVIDIA GeForce RTX 5060 Laptop-Grafikeinheit kann man seine Kreativität mit KI-optimierter Leistung und Grafik steigern und detailreiche Darstellungen genießen. Das Display des GIGABYTE AERO X16 verfügt über eine Diagonale von 40,6 cm (16") und arbeitet mit WQXGA Auflösung (2560 x 1600 Pixel) sowie einer Wiederholrate von bis zu 165 Hz. Dabei werden Daten mit Pantone-Validierung und Unterstützung für 100% sRGB dargestellt. Dank GiMATE, dem neuen GIGABYTE KI-Agenten, stehen verschiedene KI-Funktionen zur Verfügung, die auf die individuellen Bedürfnisse zugeschnitten sind – ganz intuitiv. Mit einer Bauhöhe von nur 16,7 Millimetern und einem Gewicht von 1,9 Kilogramm ist das GIGABYTE AERO X16 perfekt zum Spielen, Gestalten oder einfach zur Unterhaltung unterwegs geeignet."""

endergebnis = ganzer_chain.invoke({"text_input": eingabe_text})
print("\n---- Endgültiger JSON Output ----")
print(endergebnis)