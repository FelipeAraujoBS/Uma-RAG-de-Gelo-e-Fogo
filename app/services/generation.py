from openai import AsyncOpenAI
from app.config import GROQ_API_KEY, GROQ_MODEL

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


async def generate(question: str, context: str) -> str:
    prompt = (
        "Você é um especialista em As Crônicas de Gelo e Fogo. "
        "Responda à pergunta abaixo usando o contexto fornecido como fonte principal, "
        "mas você pode complementar com seu conhecimento próprio se necessário.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}\n\n"
        "Responda de forma clara e concisa. "
        "Se o contexto tiver a informação, cite-o. Se não tiver, use seu conhecimento, "
        "mas avise que a informação não está nos trechos recuperados."
    )

    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
