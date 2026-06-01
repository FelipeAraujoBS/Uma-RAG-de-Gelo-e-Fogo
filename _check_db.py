import sqlite3
conn = sqlite3.connect('chroma_store/chroma.sqlite3')
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cur.fetchall()
print('Tables:', tables)
for t in tables:
    name = t[0]
    cur.execute(f'PRAGMA table_info("{name}")')
    cols = cur.fetchall()
    print(f'\n--- {name} ---')
    for c in cols:
        print(f'  {c[1]} ({c[2]})')
    cur.execute(f'SELECT COUNT(*) FROM "{name}"')
    cnt = cur.fetchone()[0]
    print(f'  Rows: {cnt}')
    if cnt > 0 and cnt <= 10:
        cur.execute(f'SELECT * FROM "{name}" LIMIT 5')
        rows = cur.fetchall()
        for r in rows:
            print(f'  {r}')
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
for t in cur.fetchall():
    name = t[0]
    cur.execute(f'SELECT * FROM "{name}" LIMIT 3')
    rows = cur.fetchall()
    if rows:
        print(f'\n=== Sample from {name} ===')
        for r in rows:
            print(r)
conn.close()
