from openai import OpenAI
from app.config import GROQ_API_KEY, GROQ_MODEL

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


def generate(question: str, context: str) -> str:
    prompt = (
        "Você é um especialista em As Crônicas de Gelo e Fogo. "
        "Responda à pergunta abaixo com base apenas no contexto fornecido.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}\n\n"
        "Responda de forma clara e concisa. "
        "Se o contexto não tiver informação suficiente, diga que não sabe."
    )

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
