from google import genai
from app.config import GOOGLE_API_KEY, GENERATION_MODEL

client = genai.Client(api_key=GOOGLE_API_KEY)


def generate(question: str, context: str) -> str:
    prompt = (
        "Você é um especialista em As Crônicas de Gelo e Fogo. "
        "Responda à pergunta abaixo com base apenas no contexto fornecido.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}\n\n"
        "Responda de forma clara e concisa. "
        "Se o contexto não tiver informação suficiente, diga que não sabe."
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=[prompt],
    )
    return response.text
