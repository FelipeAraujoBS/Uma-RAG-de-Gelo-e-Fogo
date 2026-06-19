import time
from openai import AsyncOpenAI
from app.config import GROQ_API_KEY, GROQ_MODEL

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


SYSTEM_PROMPT = """
Você é um especialista em As Crônicas de Gelo e Fogo.

Regras:
- Responda com base no seu conhecimento sobre os livros.
- Se contexto foi fornecido e ele estiver correto, use-o como apoio.
- Se o contexto estiver incompleto ou incorreto, complete ou corrija com seu conhecimento.
- Seja completo: se a resposta mudou ao longo dos livros, mencione a evolução.
- Seja direto, sem enrolação.
- NÃO diga "segundo o contexto" ou "de acordo com os textos".
"""


async def generate(question: str, context: str | None = None) -> str:
    has_context = bool(context and context.strip())
    print(f"[TIMING] generate() iniciado | modelo={GROQ_MODEL} | context_tamanho={len(context or '')}", flush=True)

    user_content = (
        f"Contexto recuperado:\n{context}\n\n"
        if has_context
        else "Nenhum contexto relevante foi recuperado.\n\n"
    )

    user_content += f"Pergunta:\n{question}"

    t0 = time.time()
    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
    )
    print(f"[TIMING] Groq API chamada completa = {time.time() - t0:.2f}s", flush=True)

    import re
    content = response.choices[0].message.content.strip()
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
    content = re.sub(r'</?think>', '', content)
    content = re.sub(r'</?thought>', '', content)
    content = content.strip()
    return content