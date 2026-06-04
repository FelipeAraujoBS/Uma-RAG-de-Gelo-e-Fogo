from openai import AsyncOpenAI
from app.config import GROQ_API_KEY, GROQ_MODEL

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


async def generate(question: str, context: str) -> str:
    prompt = (
        "Você é um especialista em As Crônicas de Gelo e Fogo.\n\n"
        "Use o contexto recuperado como fonte prioritária.\n"
        "Se o contexto não contiver a resposta, use conhecimento próprio.\n\n"
        "Regras de resposta:\n"
        "- Responda primeiro de forma direta.\n"
        "- Seja breve (máximo de 1–2 frases).\n"
        "- NÃO reproduza trechos do contexto automaticamente.\n"
        "- Só mencione ou cite o contexto se a pergunta pedir evidência, explicação ou justificativa.\n"
        "- Se a resposta não estiver no contexto recuperado, diga brevemente que foi respondida com conhecimento geral.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta:\n{question}\n\n"
        "Resposta:"
    )

    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
