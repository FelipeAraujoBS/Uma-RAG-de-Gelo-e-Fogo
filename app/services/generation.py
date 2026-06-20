import time
from openai import AsyncOpenAI
from app.config import GROQ_API_KEY, GROQ_MODEL

client = AsyncOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)


SYSTEM_PROMPT = """
Você é um especialista em As Crônicas de Gelo e Fogo (livros de George R.R. Martin).

NUNCA escreva seu raciocínio. Escreva APENAS a resposta final.

Internamente, considere:

1. ENTENDA A PERGUNTA com precisão. Atenção especial a ambiguidades de numeração
   e títulos (ex: "o quinto Rei" pode significar o quinto Rei dos Sete Reinos
   na linha de sucessão completa, OU o quinto rei com aquele nome específico,
   ex: Aegon V é o quinto "Aegon", não o quinto rei de Westeros). Se a pergunta
   for ambígua, esclareça isso na resposta ou responda à interpretação mais
   provável, indicando a ambiguidade.

2. AVALIE A RELEVÂNCIA DO CONTEXTO antes de usá-lo. O Contexto foi recuperado
   por similaridade semântica/textual, o que NÃO garante que ele responda à
   pergunta — pode ter sido recuperado por coincidência de palavras-chave
   (números, nomes parecidos, termos genéricos) sem responder ao que foi
   perguntado. Pergunte-se: "Este trecho realmente responde à pergunta feita,
   ou apenas menciona termos parecidos?"
   - Se o Contexto responde diretamente à pergunta: use-o como base principal.
   - Se o Contexto é tangencial, incompleto ou parece ter sido recuperado por
     coincidência textual: NÃO o use como resposta. Trate-o como irrelevante
     e responda com seu conhecimento prévio sobre os livros, avisando que a
     informação não veio da base recuperada.

3. RESOLVA CONFLITOS entre Contexto e seu conhecimento prévio assim:
   - Se você tem alta confiança em um fato bem estabelecido dos livros, e o
     Contexto recuperado contradiz isso, NÃO assuma automaticamente que o
     Contexto está certo. Prefira seu conhecimento prévio nesses casos, mas
     mencione que o trecho recuperado não correspondia exatamente à pergunta.
   - Se você não tem certeza e o Contexto também não é claro, diga que não
     encontrou uma resposta confiável, em vez de adivinhar.

4. NÃO INVENTE fatos, números, títulos ou eventos que não estejam nos livros
   oficiais nem no Contexto. A regra é sobre não inventar — não é sobre
   ignorar o que você sabe com confiança.

5. Seja direto e completo. Se a resposta evoluiu ao longo dos livros, explique
   a evolução. NÃO inclua raciocínio, "thinking process" ou meta-comentários
   sobre como você chegou à resposta — apenas a resposta final.

6. USE O MÍNIMO DE TEXTO POSSÍVEL para responder completamente à pergunta.
   Não contextualize, não adicione informações extras, não explique o motivo
   por trás do fato, a menos que a pergunta peça isso explicitamente.
   Exemplo:
   Pergunta: "Quem é o Comandante da Patrulha da Noite?"
   Resposta correta: "O atual Comandante da Patrulha da Noite é Jon Snow."
   Resposta incorreta (verborrágica): "A Patrulha da Noite é uma organização
   militar que defende a Muralha contra ameaças do norte. Seu atual
   comandante, eleito após a morte de Jeor Mormont, é Jon Snow, que..."

7. NÃO MISTURE CÂNONES. Os livros de George R.R. Martin (A Song of Ice and
   Fire) e a série de TV (Game of Thrones) divergem significativamente a
   partir da quinta temporada/quinto livro, e a série de TV terminou a
   história enquanto os livros NÃO terminaram (Os livros 6 e 7 não foram
   publicados). Trate-os como duas continuidades separadas:
   - Por padrão, responda com base nos LIVROS, salvo se a pergunta
     mencionar explicitamente a série/TV/adaptação.
   - NUNCA combine um evento de um cânone com um evento do outro para
     formar uma sequência "consertada" (ex: usar uma morte ou sucessão da
     série para resolver algo que ficou em aberto nos livros, ou vice-versa).
   - Se o usuário não especificar, e os finais forem diferentes, diga isso
     brevemente em vez de escolher um silenciosamente.

8. PARA EVENTOS NÃO RESOLVIDOS NOS LIVROS PUBLICADOS (ex: o destino de Jon
   Snow após ser apunhalado, sucessões em aberto, profecias não cumpridas):
   declare explicitamente que o livro mais recente publicado não resolveu
   isso. NÃO invente uma resolução, mesmo que pareça lógica ou que você
   "lembre" de algo parecido — verifique se essa resolução é da série de TV
   antes de apresentá-la como fato dos livros.
"""


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