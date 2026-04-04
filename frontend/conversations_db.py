# frontend/conversations_db.py
"""
Conversation history storage in a separate DuckDB.
Stores conversations and messages with full pipeline traces.
"""

import duckdb
import json
import uuid
from datetime import datetime
from pathlib import Path

CONV_DB_PATH = Path("data/conversations.duckdb")


def get_conv_connection() -> duckdb.DuckDBPyConnection:
    CONV_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(CONV_DB_PATH))


def init_conv_schema(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id            VARCHAR PRIMARY KEY,
            title         VARCHAR,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              VARCHAR PRIMARY KEY,
            conversation_id VARCHAR,
            role            VARCHAR,
            content         VARCHAR,
            intent          VARCHAR,
            abstained       BOOLEAN DEFAULT FALSE,
            pipeline_trace  VARCHAR,
            citations       VARCHAR,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """)
    conn.commit()


def create_conversation(conn, title: str) -> str:
    conv_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO conversations (id, title)
        VALUES (?, ?)
    """, [conv_id, title[:80]])
    conn.commit()
    return conv_id


def add_message(
    conn,
    conversation_id: str,
    role:            str,
    content:         str,
    intent:          str       = None,
    abstained:       bool      = False,
    pipeline_trace:  dict      = None,
    citations:       list      = None,
) -> str:
    msg_id = str(uuid.uuid4())
    conn.execute("""
        INSERT INTO messages
            (id, conversation_id, role, content, intent,
             abstained, pipeline_trace, citations)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        msg_id,
        conversation_id,
        role,
        content,
        intent,
        abstained,
        json.dumps(pipeline_trace or {}),
        json.dumps(citations or []),
    ])
    conn.execute("""
        UPDATE conversations
        SET message_count = message_count + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, [conversation_id])
    conn.commit()
    return msg_id


def list_conversations(conn) -> list[dict]:
    rows = conn.execute("""
        SELECT id, title, created_at, updated_at, message_count
        FROM conversations
        ORDER BY updated_at DESC
        LIMIT 50
    """).fetchall()
    return [
        {
            "id":            r[0],
            "title":         r[1],
            "created_at":    str(r[2]),
            "updated_at":    str(r[3]),
            "message_count": r[4],
        }
        for r in rows
    ]


def get_messages(conn, conversation_id: str) -> list[dict]:
    rows = conn.execute("""
        SELECT id, role, content, intent, abstained,
               pipeline_trace, citations, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY created_at ASC
    """, [conversation_id]).fetchall()
    return [
        {
            "id":             r[0],
            "role":           r[1],
            "content":        r[2],
            "intent":         r[3],
            "abstained":      r[4],
            "pipeline_trace": json.loads(r[5] or "{}"),
            "citations":      json.loads(r[6] or "[]"),
            "created_at":     str(r[7]),
        }
        for r in rows
    ]


def delete_conversation(conn, conversation_id: str) -> None:
    conn.execute(
        "DELETE FROM messages WHERE conversation_id = ?",
        [conversation_id]
    )
    conn.execute(
        "DELETE FROM conversations WHERE id = ?",
        [conversation_id]
    )
    conn.commit()