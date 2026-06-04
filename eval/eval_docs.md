# Documentação do Eval — Uma RAG de Gelo e Fogo

## 1. `eval/questions.json` — Estrutura das 18 Perguntas

```json
[
  { "question": "Quais são os nomes dos dragões da Daenerys?", "ground_truth": "Drogon, Rhaegal e Viserion" },
  { "question": "Quem é o pai de Jon Snow?", "ground_truth": "Rhaegar Targaryen" },
  { "question": "Qual é o nome da espada de aço valiriano de Ned Stark?", "ground_truth": "Gelo" },
  { "question": "Quem matou o Rei Louco Aerys Targaryen?", "ground_truth": "Jaime Lannister" },
  { "question": "Qual casa tem como lema 'Inverno está chegando'?", "ground_truth": "Casa Stark" },
  { "question": "Como Tyrion Lannister ficou com a cicatriz no rosto?", "ground_truth": "Na Batalha do Blackwater, quando o Sor Mandon Moore tentou matá-lo e ele foi cortado no rosto" },
  { "question": "O que é a Muralha e quem a construiu?", "ground_truth": "A Muralha é uma enorme barreira de gelo no norte de Westeros, construída pelos construtores da Patrulha da Noite e pelos Filhos da Floresta, com a ajuda de gigantes e magia, há mais de 8.000 anos" },
  { "question": "Quem é o três olhos corvo?", "ground_truth": "Brynden Rivers, também conhecido como Corvo de Sangue, um ex-comandante da Patrulha da Noite e Targaryen bastardo" },
  { "question": "O que aconteceu em Casamento Vermelho?", "ground_truth": "Robb Stark, sua mãe Catelyn e muitos de seus vassalos foram assassinados durante um casamento em Correrrio, pelos Frey e Boltons, violando as leis de hospitalidade" },
  { "question": "Quem é Azor Ahai?", "ground_truth": "Um herói lendário da antiga Valíria que forjou a espada Luminífera (Lightbringer) para lutar contra a escuridão" },
  { "question": "Onde Daenerys chocou seus ovos de dragão?", "ground_truth": "Na pira funerária de Khal Drogo, em meio ao deserto vermelho além do Mar Dothraki" },
  { "question": "Quem é o rei da noite?", "ground_truth": "O Rei da Noite é uma figura lendaria do folclore de Westeros" },
  { "question": "Por que Sor Jorah Mormont foi exilado de Westeros?", "ground_truth": "Por vender caçadores ilegais para escravistas, um crime punível com a morte" },
  { "question": "O que é a cidadela?", "ground_truth": "A Cidadela é a ordem dos meistres, localizada em Vilavela, onde os meistres são treinados" },
  { "question": "Como Jaime Lannister perdeu a mão?", "ground_truth": "Foi cortada por Vargo Hoat, durante sua captura no rio Tridente" },
  { "question": "Quem é o chefe da guarda real de Robert Baratheon?", "ground_truth": "Sor Barristan Selmy" },
  { "question": "O que é a profecia do príncipe que foi prometido?", "ground_truth": "Uma profecia antiga que diz que um herói renascerá para salvar o mundo da escuridão, brandindo a espada Luminífera" },
  { "question": "Qual casa comanda Pedra do Dragão no início da série?", "ground_truth": "Casa Baratheon, sob Stannis Baratheon" }
]
```

---

## 2. `scripts/run_eval.py` — Pipeline de Avaliação

O script **importa os serviços diretamente** (não chama o endpoint HTTP `/api/chat`):

```python
from app.services.retrieval import search as retrieve, model
from app.services.generation import generate
```

### Fluxo por Pergunta

```
1. retrieve(question)
   ├── embedding da pergunta (BAAI/bge-m3)
   ├── busca semântica no ChromaDB (top 60)
   ├── BM25 (top 60)
   ├── fusão RRF → top 40
   └── reranking (BAAI/bge-reranker-v2-m3) → top 20 (usa só 5)

2. generate(question, context)  ← LLM (Groq llama-3.1-8b-instant)
   └── prompt: "Responda com base apenas no contexto fornecido"
   └── retorna answer (string)

3. Métricas calculadas:
   ├── context_precision  → cosseno entre embedding da pergunta e dos chunks
   ├── answer_relevancy   → LLM gera 3 perguntas derivadas da resposta
   │                       → cosseno entre embedding original e as geradas
   ├── context_recall     → LLM extrai claims do ground_truth
   │                       → LLM verifica se cada claim está no contexto
   └── faithfulness       → LLM extrai claims da answer
                           → LLM verifica se cada claim está no contexto
```

### LLM Calls por Pergunta (~12 chamadas)

| Chamada | Input (tokens) | Output (tokens) |
|---|---|---|
| generate_answer (1x) | ~50 sys + ~5 chunks + pergunta | ~100 |
| extrair claims do ground_truth (1x) | ~30 | ~50 |
| verificar claims ~4 (context_recall) | ~50 + ~5 chunks por chamada | ~5 cada |
| extrair claims da answer (1x) | ~30 | ~50 |
| verificar claims ~4 (faithfulness) | ~50 + ~5 chunks por chamada | ~5 cada |
| gerar perguntas (answer_relevancy) (1x) | ~30 | ~60 |

### Notas Técnicas

- **Chunks por pergunta**: 5 (limitado para caber no TPD do Groq free)
- **LLM**: `llama-3.1-8b-instant` via Groq (API compatível com OpenAI)
- **Delay entre chamadas**: 2s para evitar rate limit (30 req/min)
- **Retry automático**: até 5 tentativas com backoff
- **Resumível**: salva resultados incrementalmente em `eval/results.json`
- **Embedding model**: `BAAI/bge-m3` via SentenceTransformers (local)
- **Reranker**: `BAAI/bge-reranker-v2-m3` via CrossEncoder (local)

---

## 3. Exemplo de Resultado (API + Eval)

### Resposta direta da FastAPI (`POST /api/chat`)

```json
{
  "question": "Qual o nome da espada ancestral da Casa Stark?",
  "answer": "Gelo.",
  "sources": [
    {
      "book": "A Fúria dos Reis",
      "chapter": "Catelyn VII",
      "pov": "Catelyn Tully",
      "distance": 0.413,
      "text_preview": "Gelo era a espada de Ned. Aço valiriano, marcado com as ondulações de um milhar de dobras, tão afiado que eu tinha medo de tocar..."
    },
    {
      "book": "A Guerra dos Tronos",
      "chapter": "Bran I",
      "pov": "Bran Stark",
      "distance": 0.433,
      "text_preview": "Lorde Eddard Stark desmontou, e seu protegido, Theon Greyjoy, apresentou-lhe a espada. Chamavam Gelo àquela espada..."
    },
    {
      "book": "A Fúria dos Reis",
      "chapter": "Arya V",
      "pov": "Arya Stark",
      "distance": 0.452,
      "text_preview": "Arya o encarou. Meu nome é Arya. Da Casa Stark."
    }
  ]
}
```

### Registro do Eval no `eval/results.json`

```json
{
  "question": "Qual é o nome da espada de aço valiriano de Ned Stark?",
  "answer": "O nome da espada de aço valiriano de Ned Stark é Gelo.",
  "ground_truth": "Gelo",
  "context_precision": 0.5251,
  "answer_relevancy": 0.9152,
  "context_recall": 0.0,
  "faithfulness": 1.0,
  "avg_distance": 0.5206,
  "sources": [
    {
      "book": "A Fúria dos Reis",
      "chapter": "Catelyn VII",
      "distance": 0.413,
      "text_preview": "Gelo era a espada de Ned. Aço valiriano, marcado com as ondulações de um milhar de dobras..."
    },
    {
      "book": "A Guerra dos Tronos",
      "chapter": "Bran I",
      "distance": 0.433,
      "text_preview": "Lorde Eddard Stark desmontou... Chamavam Gelo àquela espada..."
    }
  ]
}
```

### Observações sobre o Formato

- **Os chunks vêm como texto completo** no campo `text_preview` (truncado em 120 caracteres no sources, mas o texto completo está em `documents`).
- **Os metadados** incluem `book_title`, `chapter_title`, `pov` e `distance`.
- O eval usa **apenas 5 chunks** (os top 5 do reranking), não os 20 completos — para economizar tokens no Groq free.
- O `context_recall` tende a ser baixo porque as verificações de claims usam LLM para extrair e julgar, o que é rigoroso.
