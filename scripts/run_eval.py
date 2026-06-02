import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "questions.json")

from openai import OpenAI
from app.services.retrieval import search as retrieve, model
from app.config import GROQ_API_KEY, GROQ_MODEL

llm = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")


def _encode_query(text):
    if isinstance(text, str):
        return model.encode(f"Represent this sentence for searching relevant passages: {text}")
    return model.encode([f"Represent this sentence for searching relevant passages: {t}" for t in text])


def _encode_doc(text):
    if isinstance(text, str):
        return model.encode(text)
    return model.encode(text)


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
    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def llm_call(prompt: str) -> str:
    response = llm.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def context_precision(question: str, contexts: list[str]) -> float:
    q_emb = _encode_query(question)
    c_embs = _encode_doc(contexts)
    sims = []
    for c_emb in c_embs:
        dot = sum(a * b for a, b in zip(q_emb, c_emb))
        q_norm = sum(a * a for a in q_emb) ** 0.5
        c_norm = sum(b * b for b in c_emb) ** 0.5
        sims.append(dot / (q_norm * c_norm) if q_norm * c_norm > 0 else 0.0)
    return sum(sims) / len(sims) if sims else 0.0


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

    q_emb = _encode_query(question)
    gen_embs = _encode_query(gen_questions)

    scores = []
    for g_emb in gen_embs:
        dot = sum(a * b for a, b in zip(q_emb, g_emb))
        q_norm = sum(a * a for a in q_emb) ** 0.5
        g_norm = sum(b * b for b in g_emb) ** 0.5
        scores.append(dot / (q_norm * g_norm) if q_norm * g_norm > 0 else 0.0)

    return sum(scores) / len(scores) if scores else 0.0


def main():
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"Carregadas {len(questions)} perguntas de eval.\n")

    all_metrics = []

    for i, q in enumerate(questions[:5]):
        question = q["question"]
        ground_truth = q["ground_truth"]

        print(f"[{i+1}/{len(questions)}] {question[:70]}")

        results = retrieve(question)
        contexts = results["documents"]
        distances = results["distances"]
        metadatas = results["metadatas"]

        answer = generate_answer(question, contexts)

        cp = context_precision(question, contexts)
        ar = answer_relevancy(question, answer)
        cr = context_recall(question, contexts, ground_truth)
        fh = faithfulness(answer, contexts)
        avg_dist = sum(distances) / len(distances)

        all_metrics.append({
            "question": question,
            "answer": answer,
            "ground_truth": ground_truth,
            "context_precision": float(round(cp, 4)),
            "answer_relevancy": float(round(ar, 4)),
            "context_recall": float(round(cr, 4)),
            "faithfulness": float(round(fh, 4)),
            "avg_distance": float(round(avg_dist, 4)),
            "sources": [
                {
                    "book": m["book_title"],
                    "chapter": m["chapter_title"],
                    "distance": float(round(d, 4)),
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
        print("  ! Faithfulness baixo — a resposta contem informacoes")
        print("    que nao estao nos chunks recuperados.")

    out_path = os.path.join(os.path.dirname(__file__), "..", "eval", "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {out_path}")


if __name__ == "__main__":
    main()
