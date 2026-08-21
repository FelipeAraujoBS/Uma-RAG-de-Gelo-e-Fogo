import sqlite3

con = sqlite3.connect(r'.\Uma-RAG=de-Gelo-e-Fogo\chroma_store\chroma.sqlite3')
cur = con.cursor()

print("--- collections ---")
print(cur.execute("SELECT id, name FROM collections;").fetchall())

print("--- segments schema ---")
print(cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='segments';").fetchall())

print("--- segments data ---")
print(cur.execute("SELECT * FROM segments;").fetchall())