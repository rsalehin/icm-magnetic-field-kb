# scripts/corpus_overview.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.storage.db import get_connection

conn = get_connection()

print("=== Papers by decade ===")
conn.execute("""
    SELECT
        CASE
            WHEN year < 2000 THEN '1995-1999'
            WHEN year < 2005 THEN '2000-2004'
            WHEN year < 2010 THEN '2005-2009'
            WHEN year < 2015 THEN '2010-2014'
            WHEN year < 2020 THEN '2015-2019'
            ELSE '2020-2026'
        END as era,
        COUNT(*) as n
    FROM papers
    GROUP BY era
    ORDER BY era
""").fetchall()
for row in conn.execute("""
    SELECT
        CASE
            WHEN year < 2000 THEN '1995-1999'
            WHEN year < 2005 THEN '2000-2004'
            WHEN year < 2010 THEN '2005-2009'
            WHEN year < 2015 THEN '2010-2014'
            WHEN year < 2020 THEN '2015-2019'
            ELSE '2020-2026'
        END as era,
        COUNT(*) as n
    FROM papers GROUP BY era ORDER BY era
""").fetchall():
    print(f"  {row[0]}: {row[1]} papers")

print("\n=== Top 20 most cited papers in corpus ===")
for row in conn.execute("""
    SELECT arxiv_id, year, citation_count,
           SUBSTR(title, 1, 55) as title
    FROM papers
    ORDER BY citation_count DESC NULLS LAST
    LIMIT 20
""").fetchall():
    print(f"  [{row[1]}] {row[2]:>4} cites — {row[3]}")

print("\n=== Authors with most papers ===")
for row in conn.execute("""
    SELECT author_name, COUNT(*) as n
    FROM authors WHERE position = 0
    GROUP BY author_name
    ORDER BY n DESC LIMIT 15
""").fetchall():
    print(f"  {row[0]:<25} {row[1]} papers")

print("\n=== Journals represented ===")
for row in conn.execute("""
    SELECT journal, COUNT(*) as n
    FROM papers
    GROUP BY journal ORDER BY n DESC LIMIT 10
""").fetchall():
    print(f"  {row[0]:<45} {row[1]}")

conn.close()