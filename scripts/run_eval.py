import sys
import os
import json
import re

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "questions.json")

# ── services ──────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from app.config import CHROMA_PATH, COLLECTION_NAME, EMBEDDING_MODEL, GOOGLE_API_KEY, GENERATION_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)
chroma = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma.get_collection(COLLECTION_NAME)
llm = genai.Client(api_key=GOOGLE_API_KEY)


# ── retrieval ─────────────────────────────────────────────
def retrieve(question: str, n_results: int = 5) -> dict:
    embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[embedding], n_results=n_results)
    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


def generate_answer(question: str, contexts: list[str]) -> str:
    context = "\n\n".join(contexts)
    prompt = (
        "Voce e um especialista em As Cronicas de Gelo e Fogo. "
        "Responda a pergunta abaixo com base apenas no contexto fornecido.\n\n"
        f"Contexto:\n{context}\n\n"
        f"Pergunta: {question}\n\n"
        "Responda de forma clara e concisa. "
        "Se o contexto nao tiver informacao suficiente, diga que nao sabe."
    )
    response = llm.models.generate_content(model=GENERATION_MODEL, contents=[prompt])
    return response.text


def llm_call(prompt: str) -> str:
    response = llm.models.generate_content(model=GENERATION_MODEL, contents=[prompt])
    return response.text.strip()


# ── metric: context_precision ─────────────────────────────
def context_precision(question: str, contexts: list[str]) -> float:
    q_emb = model.encode(question)
    c_embs = model.encode(contexts)
    sims = []
    for c_emb in c_embs:
        dot = sum(a * b for a, b in zip(q_emb, c_emb))
        q_norm = sum(a * a for a in q_emb) ** 0.5
        c_norm = sum(b * b for b in c_emb) ** 0.5
        sims.append(dot / (q_norm * c_norm) if q_norm * c_norm > 0 else 0.0)
    return sum(sims) / len(sims) if sims else 0.0


# ── metric: context_recall (LLM-as-judge) ─────────────────
def context_recall(question: str, contexts: list[str], ground_truth: str) -> float:
    context_text = "\n\n".join(contexts)

    extract_prompt = (
        "Extraia as afirmacoes ou fatos principais da resposta abaixo. "
        "Retorne cada afirmacao em uma linha separada, sem numeracao.\n\n"
        f"Resposta: {ground_truth}"
    )
    claims_raw = llm_call(extract_prompt)
    claims = [c.strip() for c in claims_raw.split("\n") if c.strip()]

    if not claims:
        return 0.0

    supported = 0
    for claim in claims:
        verify_prompt = (
            "Voce e um juiz de avaliacao RAG. "
            "Determine se a afirmacao abaixo pode ser diretamente suportada "
            "pelo contexto fornecido. Responda apenas SIM ou NAO.\n\n"
            f"Afirmacao: {claim}\n\n"
            f"Contexto:\n{context_text}"
        )
        verdict = llm_call(verify_prompt).upper()
        if "SIM" in verdict:
            supported += 1

    return supported / len(claims) if claims else 0.0


# ── metric: faithfulness (LLM-as-judge) ────────────────────
def faithfulness(answer: str, contexts: list[str]) -> float:
    context_text = "\n\n".join(contexts)

    extract_prompt = (
        "Extraia as afirmacoes ou fatos principais da resposta abaixo. "
        "Retorne cada afirmacao em uma linha separada, sem numeracao.\n\n"
        f"Resposta: {answer}"
    )
    claims_raw = llm_call(extract_prompt)
    claims = [c.strip() for c in claims_raw.split("\n") if c.strip()]

    if not claims:
        return 1.0

    supported = 0
    for claim in claims:
        verify_prompt = (
            "Voce e um juiz de avaliacao RAG. "
            "Determine se a afirmacao abaixo pode ser diretamente suportada "
            "pelo contexto fornecido. Responda apenas SIM ou NAO.\n\n"
            f"Afirmacao: {claim}\n\n"
            f"Contexto:\n{context_text}"
        )
        verdict = llm_call(verify_prompt).upper()
        if "SIM" in verdict:
            supported += 1

    return supported / len(claims) if claims else 1.0


# ── metric: answer_relevancy (RAGAS-style) ─────────────────
def answer_relevancy(question: str, answer: str) -> float:
    gen_prompt = (
        "Gere 3 perguntas diferentes que poderiam ter gerado esta resposta. "
        "Retorne uma pergunta por linha, sem numeracao.\n\n"
        f"Resposta: {answer}"
    )
    gen_raw = llm_call(gen_prompt)
    gen_questions = [q.strip() for q in gen_raw.split("\n") if q.strip()]

    if not gen_questions:
        return 0.0

    q_emb = model.encode(question)
    gen_embs = model.encode(gen_questions)

    scores = []
    for g_emb in gen_embs:
        dot = sum(a * b for a, b in zip(q_emb, g_emb))
        q_norm = sum(a * a for a in q_emb) ** 0.5
        g_norm = sum(b * b for b in g_emb) ** 0.5
        scores.append(dot / (q_norm * g_norm) if q_norm * g_norm > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


# ── main ──────────────────────────────────────────────────
def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Carregadas {len(questions)} perguntas de eval.\n")

    all_metrics = []

    for i, q in enumerate(questions):
        question = q["question"]
        ground_truth = q["ground_truth"]

        print(f"[{i+1}/{len(questions)}] {question[:70]}")

        # retrieval
        results = retrieve(question)
        contexts = results["documents"]
        distances = results["distances"]
        metadatas = results["metadatas"]

        # generation
        answer = generate_answer(question, contexts)

        # metrics
        cp = context_precision(question, contexts)
        ar = answer_relevancy(question, answer)
        cr = context_recall(question, contexts, ground_truth)
        fh = faithfulness(answer, contexts)
        avg_dist = sum(distances) / len(distances)

        all_metrics.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "context_precision": round(cp, 4),
            "answer_relevancy": round(ar, 4),
            "context_recall": round(cr, 4),
            "faithfulness": round(fh, 4),
            "avg_distance": round(avg_dist, 4),
            "sources": [
                {
                    "book": m["book_title"],
                    "chapter": m["chapter_title"],
                    "distance": round(d, 4),
                    "text_preview": c[:120],
                }
                for m, d, c in zip(metadatas, distances, contexts)
            ],
        })

        print(f"  CP (precision):  {cp:.3f}")
        print(f"  AR (relevancy):  {ar:.3f}")
        print(f"  CR (recall):     {cr:.3f}")
        print(f"  FH (faithful):   {fh:.3f}")
        print(f"  Dist:            {avg_dist:.3f}")
        print(f"  Resposta:        {answer[:100]}...")
        print()

    # ── summary ──
    avg_cp = sum(m["context_precision"] for m in all_metrics) / len(all_metrics)
    avg_ar = sum(m["answer_relevancy"] for m in all_metrics) / len(all_metrics)
    avg_cr = sum(m["context_recall"] for m in all_metrics) / len(all_metrics)
    avg_fh = sum(m["faithfulness"] for m in all_metrics) / len(all_metrics)
    avg_dist = sum(m["avg_distance"] for m in all_metrics) / len(all_metrics)

    print("=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)
    print(f"  Perguntas:           {len(all_metrics)}")
    print(f"  Context Precision:   {avg_cp:.3f}  (quanto mais alto, melhor)")
    print(f"  Answer Relevancy:    {avg_ar:.3f}  (quanto mais alto, melhor)")
    print(f"  Context Recall:      {avg_cr:.3f}  (quanto mais alto, melhor)")
    print(f"  Faithfulness:        {avg_fh:.3f}  (quanto mais alto, melhor)")
    print(f"  Cosine Distance:     {avg_dist:.3f}  (quanto mais baixo, melhor)")
    print()

    if avg_cr < 0.5:
        print("  ! Context Recall baixo — os chunks recuperados nao contem")
        print("    a informacao necessaria para responder.")
    if avg_cp < 0.6:
        print("  ! Context Precision baixo — chunks recuperados tem baixa")
        print("    relevancia para a pergunta.")
    if avg_fh < 0.7:
        print("  ! Faithfulness baixo — a resposta do Gemini contem informacoes")
        print("    que nao estao nos chunks recuperados.")

    out_path = os.path.join(os.path.dirname(__file__), "..", "eval", "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {out_path}")


if __name__ == "__main__":
    main()
