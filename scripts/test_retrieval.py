import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.services.retrieval import search


def main():
    question = input("Pergunta: ")
    results = search(question, n_results=5)

    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"], results["metadatas"], results["distances"])
    ):
        print(f"\n--- Resultado {i+1} ---")
        print(f"Livro:     {meta['book_title']}")
        print(f"Capitulo:  {meta['chapter_title']}")
        print(f"POV:       {meta['pov']}")
        print(f"Distancia: {dist:.4f}")
        print(f"Texto:     {doc[:300]}...")


if __name__ == "__main__":
    main()
