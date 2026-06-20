import time
from openai import AsyncOpenAI
from app.config import GROQ_API_KEY, GROQ_MODEL

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


SYSTEM_PROMPT = """Você é um especialista em As Crônicas de Gelo e Fogo (livros de George R.R. Martin). Responda apenas com a resposta final — NUNCA mostre seu raciocínio.

Regras:
1. Se o Contexto responder diretamente à pergunta, use-o como base.
2. Se o Contexto for tangencial ou contradizer seu conhecimento, ignore-o e responda com seu conhecimento, avisando que a informação não veio da base.
3. Não invente fatos que não estejam nos livros ou no Contexto.
4. Responda com base nos LIVROS por padrão, não na série de TV.
5. Eventos não resolvidos nos livros: declare que o livro mais recente não resolveu.
6. Mínimo de texto possível. Seja direto. Uma frase basta.


MAX_CONTEXT_CHARS = 20000


async def generate(question: str, context: str | None = None) -> str:
    has_context = bool(context and context.strip())
    if has_context and len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "\n[Contexto truncado...]"
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
    content = response.choices[0].message.content or ""
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<thought>.*?</thought>', '', content, flags=re.DOTALL)
    content = re.sub(r'</?think>', '', content)
    content = re.sub(r'</?thought>', '', content)
    content = re.sub(r'(?is)^here\'?s\s+a\s+thinking\s+process:?.*?(?=\n[a-zA-Z]|\Z)', '', content)
    content = content.strip()
    if not content:
        print(f"[WARN] Resposta vazia da Groq. raw choices: {response.choices}", flush=True)
    return content