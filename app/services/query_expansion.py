from openai import AsyncOpenAI
from app.config import GROQ_API_KEY

_rewrite_client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


async def expand_queries(question: str) -> list[str]:
    resp = await _rewrite_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Pergunta: {question}\n\n"
                    "Gere 3 consultas de busca curtas (3-7 palavras) em portugues "
                    "para encontrar a resposta em livros de As Cronicas de Gelo e Fogo. "
                    "As consultas devem conter palavras que realmente aparecem nos livros. "
                    "Nao repita a pergunta. Retorne APENAS as 3 consultas, uma por linha."
                ),
            }
        ],
        max_tokens=256,
    )
    lines = [
        l.strip().lstrip("*-0123456789. ")
        for l in resp.choices[0].message.content.strip().split("\n")
        if l.strip()
    ]
    queries = [question] + lines[:3]
    return queries
