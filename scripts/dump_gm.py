"""Read-only GM snapshot dumper. Run anytime, even while bot is running."""
import io
import sqlite3
import sys
from pathlib import Path

# Force UTF-8 stdout so Chinese node names + arrows render on Windows GBK consoles.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

DB = Path("workspace/data/graph_memory.sqlite")


def main():
    if not DB.exists():
        print(f"no GM file at {DB}", file=sys.stderr)
        sys.exit(1)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row

    nodes = db.execute(
        "SELECT id, category, source_type, name, description, importance, "
        "json_extract(metadata,'$.classified') AS classified, created_at "
        "FROM gm_nodes ORDER BY id ASC"
    ).fetchall()
    edges = db.execute(
        "SELECT na.name AS src, e.predicate AS pred, nb.name AS tgt "
        "FROM gm_edges e "
        "JOIN gm_nodes na ON na.id=e.node_a "
        "JOIN gm_nodes nb ON nb.id=e.node_b "
        "ORDER BY e.id ASC"
    ).fetchall()
    cat_counts = db.execute(
        "SELECT category, COUNT(*) AS n FROM gm_nodes GROUP BY category"
    ).fetchall()
    src_counts = db.execute(
        "SELECT source_type, COUNT(*) AS n, "
        "SUM(CASE WHEN json_extract(metadata,'$.classified')=1 THEN 1 ELSE 0 END) AS classified "
        "FROM gm_nodes GROUP BY source_type"
    ).fetchall()

    print(f"=== Graph Memory: {len(nodes)} nodes, {len(edges)} edges ===\n")

    print("By category:")
    for r in cat_counts:
        print(f"  {r['category']:<10} {r['n']}")
    print()

    print("By source_type:")
    for r in src_counts:
        cls = r["classified"] or 0
        print(f"  {r['source_type']:<10} {r['n']}  (classified: {cls})")
    print()

    print("Nodes:")
    for n in nodes:
        cls = "*" if n["classified"] == 1 else " "
        desc = (n["description"] or "").replace("\n", " ")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        print(f"  [{n['id']:>3}] {cls} {n['category']:<9} {n['source_type']:<8} "
              f"imp={n['importance']:.1f}  {n['name']}")
        if desc and desc != n["name"]:
            print(f"            -- {desc}")
    print()

    if edges:
        print("Edges:")
        for e in edges:
            print(f"  {e['src']}  --{e['pred']}-->  {e['tgt']}")


if __name__ == "__main__":
    main()
