---
title: Uma RAG de Gelo e Fogo
emoji: 🐺
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# Uma RAG de Gelo e Fogo

Microsserviço de RAG (Retrieval-Augmented Generation) para responder perguntas sobre As Crônicas de Gelo e Fogo.

## Variáveis de ambiente

| Variável | Descrição |
|---|---|
| `GROQ_API_KEY` | Chave da API Groq (obrigatória) |
| `DB_PATH` | Caminho para o SQLite com os parágrafos (fallback FTS5) |
| `CHROMA_PATH` | Caminho para o ChromaDB persistido |
