"""Smoke tests for schemas.sql — ensures all tables + triggers load."""
import sqlite3
from pathlib import Path

import pytest


SCHEMA_PATH = Path(__file__).parent.parent / "src" / "memory" / "schemas.sql"


def _open(tmp_path):
    db = sqlite3.connect(tmp_path / "t.sqlite")
    db.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return db


def test_schema_file_exists():
    assert SCHEMA_PATH.exists()


def test_creates_all_graph_memory_tables(tmp_path):
    db = _open(tmp_path)
    names = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("gm_nodes", "gm_edges", "gm_communities",
              "gm_node_communities", "kb_registry",
              "kb_meta", "kb_entries", "kb_edges"):
        assert t in names, f"missing table: {t}"


def test_fts_virtual_tables_present(tmp_path):
    db = _open(tmp_path)
    names = {r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "gm_nodes_fts" in names
    assert "kb_entries_fts" in names


def test_node_category_check_constraint(tmp_path):
    db = _open(tmp_path)
    # Valid insert
    db.execute("INSERT INTO gm_nodes(name, category, description) "
               "VALUES('a','FACT','desc')")
    # Invalid category rejected
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO gm_nodes(name, category, description) "
                   "VALUES('b','BOGUS','desc')")


def test_edge_node_order_check(tmp_path):
    db = _open(tmp_path)
    db.execute("INSERT INTO gm_nodes(name,category,description) VALUES('x','FACT','')")
    db.execute("INSERT INTO gm_nodes(name,category,description) VALUES('y','FACT','')")
    # node_a < node_b OK
    db.execute("INSERT INTO gm_edges(node_a,node_b,predicate) VALUES(1,2,'RELATED_TO')")
    # reversed rejected
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO gm_edges(node_a,node_b,predicate) VALUES(2,1,'RELATED_TO')")


def test_fts_trigger_syncs_on_insert(tmp_path):
    db = _open(tmp_path)
    db.execute("INSERT INTO gm_nodes(name,category,description) "
               "VALUES('apple','FACT','red fruit')")
    hits = db.execute(
        "SELECT name FROM gm_nodes_fts WHERE gm_nodes_fts MATCH 'apple'"
    ).fetchall()
    assert hits and hits[0][0] == "apple"


def test_fts_trigger_cleans_on_delete(tmp_path):
    db = _open(tmp_path)
    db.execute("INSERT INTO gm_nodes(name,category,description) "
               "VALUES('banana','FACT','yellow fruit')")
    db.execute("DELETE FROM gm_nodes WHERE name='banana'")
    hits = db.execute(
        "SELECT name FROM gm_nodes_fts WHERE gm_nodes_fts MATCH 'banana'"
    ).fetchall()
    assert hits == []


def test_gm_edge_unique_constraint(tmp_path):
    db = _open(tmp_path)
    db.execute("INSERT INTO gm_nodes(name,category,description) VALUES('x','FACT','')")
    db.execute("INSERT INTO gm_nodes(name,category,description) VALUES('y','FACT','')")
    db.execute("INSERT INTO gm_edges(node_a,node_b,predicate) VALUES(1,2,'RELATED_TO')")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO gm_edges(node_a,node_b,predicate) VALUES(1,2,'RELATED_TO')")


def test_kb_entry_and_edge_insert(tmp_path):
    db = _open(tmp_path)
    db.execute("INSERT INTO kb_entries(content) VALUES('a')")
    db.execute("INSERT INTO kb_entries(content) VALUES('b')")
    db.execute("INSERT INTO kb_edges(entry_a,entry_b,predicate) VALUES(1,2,'RELATED_TO')")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("INSERT INTO kb_edges(entry_a,entry_b,predicate) VALUES(2,1,'RELATED_TO')")
