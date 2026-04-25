import sqlite3

conn = sqlite3.connect(".data/finstock.db")

cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cur.fetchall()]
print("Tables:", tables)

for t in tables:
    cur2 = conn.execute(f"PRAGMA table_info({t})")
    cols = [row[1] for row in cur2.fetchall()]
    print(f"  {t}: {cols}")

conn.close()
