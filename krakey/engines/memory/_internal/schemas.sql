-- Graph Memory + Knowledge Base schema (DevSpec §7.4 + §8.2).
-- Single file: apply relevant sections to the target database.

-- =============================================================
-- Graph Memory (workspace/data/graph_memory.sqlite)
-- =============================================================

CREATE TABLE IF NOT EXISTS gm_nodes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    category      TEXT    NOT NULL CHECK(category IN
                    ('FACT','RELATION','KNOWLEDGE','TARGET','FOCUS')),
    description   TEXT,
    importance    REAL    DEFAULT 1.0,
    metadata      TEXT,                      -- JSON, e.g. {"classified": true}
    embedding     BLOB,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_count  INTEGER DEFAULT 0,
    source_heartbeat INTEGER,
    source_type   TEXT    DEFAULT 'auto'     -- auto|explicit|compact|sleep
);

CREATE INDEX IF NOT EXISTS idx_gm_nodes_category    ON gm_nodes(category);
CREATE INDEX IF NOT EXISTS idx_gm_nodes_source_type ON gm_nodes(source_type);

CREATE VIRTUAL TABLE IF NOT EXISTS gm_nodes_fts USING fts5(
    name, description,
    content='gm_nodes', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS gm_fts_ai
AFTER INSERT ON gm_nodes BEGIN
    INSERT INTO gm_nodes_fts(rowid, name, description)
    VALUES (new.id, new.name, new.description);
END;

CREATE TRIGGER IF NOT EXISTS gm_fts_ad
AFTER DELETE ON gm_nodes BEGIN
    INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description)
    VALUES ('delete', old.id, old.name, old.description);
END;

CREATE TRIGGER IF NOT EXISTS gm_fts_au
AFTER UPDATE ON gm_nodes BEGIN
    INSERT INTO gm_nodes_fts(gm_nodes_fts, rowid, name, description)
    VALUES ('delete', old.id, old.name, old.description);
    INSERT INTO gm_nodes_fts(rowid, name, description)
    VALUES (new.id, new.name, new.description);
END;

CREATE TABLE IF NOT EXISTS gm_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_a      INTEGER NOT NULL REFERENCES gm_nodes(id) ON DELETE CASCADE,
    node_b      INTEGER NOT NULL REFERENCES gm_nodes(id) ON DELETE CASCADE,
    predicate   TEXT    NOT NULL,
    description TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK(node_a < node_b),
    UNIQUE(node_a, node_b, predicate)
);

CREATE INDEX IF NOT EXISTS idx_gm_edges_a ON gm_edges(node_a);
CREATE INDEX IF NOT EXISTS idx_gm_edges_b ON gm_edges(node_b);

CREATE TABLE IF NOT EXISTS gm_communities (
    community_id      INTEGER PRIMARY KEY,
    name              TEXT,
    summary           TEXT,
    summary_embedding BLOB,
    member_count      INTEGER  DEFAULT 0,
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gm_node_communities (
    node_id      INTEGER NOT NULL REFERENCES gm_nodes(id) ON DELETE CASCADE,
    community_id INTEGER NOT NULL REFERENCES gm_communities(community_id) ON DELETE CASCADE,
    PRIMARY KEY (node_id, community_id)
);

CREATE TABLE IF NOT EXISTS kb_registry (
    kb_id           TEXT    PRIMARY KEY,
    name            TEXT    NOT NULL,
    path            TEXT    NOT NULL,
    description     TEXT,
    topics          TEXT,                        -- JSON array
    entry_count     INTEGER  DEFAULT 0,
    is_archived     INTEGER  DEFAULT 0,          -- BOOLEAN: archived KBs lose GM index node
    index_embedding BLOB,                        -- mean of member entry embeddings; used for revive cosine match
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================
-- Knowledge Base (per-topic SQLite file: workspace/data/knowledge_bases/*.sqlite)
-- Included in the same schema file so a KB bootstrap can reuse it.
-- =============================================================

CREATE TABLE IF NOT EXISTS kb_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS kb_entries (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    content        TEXT    NOT NULL,
    source         TEXT,
    tags           TEXT,                      -- JSON array
    embedding      BLOB,
    importance     REAL    DEFAULT 1.0,
    created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed  DATETIME DEFAULT CURRENT_TIMESTAMP,
    access_count   INTEGER DEFAULT 0,
    superseded_by  INTEGER REFERENCES kb_entries(id),
    is_active      BOOLEAN DEFAULT 1
);

CREATE TABLE IF NOT EXISTS kb_edges (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_a     INTEGER NOT NULL REFERENCES kb_entries(id) ON DELETE CASCADE,
    entry_b     INTEGER NOT NULL REFERENCES kb_entries(id) ON DELETE CASCADE,
    predicate   TEXT    NOT NULL,
    description TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    CHECK(entry_a < entry_b),
    UNIQUE(entry_a, entry_b, predicate)
);

CREATE VIRTUAL TABLE IF NOT EXISTS kb_entries_fts USING fts5(
    content, source, tags,
    content='kb_entries', content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS kb_fts_ai
AFTER INSERT ON kb_entries BEGIN
    INSERT INTO kb_entries_fts(rowid, content, source, tags)
    VALUES (new.id, new.content, new.source, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS kb_fts_ad
AFTER DELETE ON kb_entries BEGIN
    INSERT INTO kb_entries_fts(kb_entries_fts, rowid, content, source, tags)
    VALUES ('delete', old.id, old.content, old.source, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS kb_fts_au
AFTER UPDATE ON kb_entries BEGIN
    INSERT INTO kb_entries_fts(kb_entries_fts, rowid, content, source, tags)
    VALUES ('delete', old.id, old.content, old.source, old.tags);
    INSERT INTO kb_entries_fts(rowid, content, source, tags)
    VALUES (new.id, new.content, new.source, new.tags);
END;
