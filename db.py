from __future__ import annotations

import sqlite3
from typing import Iterable, Optional

DB_NAME = "tube_trends.db"


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    existing = {
        row["name"]
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_videos (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            channel TEXT,
            publishedAt TEXT,
            duration_sec INTEGER,
            views INTEGER,
            likes INTEGER,
            comments INTEGER,
            tags TEXT,
            thumbnail_url TEXT,
            watch_url TEXT,
            region TEXT,
            keyword TEXT,
            category TEXT,
            saved_ts TEXT,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tracked_videos (
            video_id TEXT PRIMARY KEY,
            title TEXT,
            channel TEXT,
            added_ts TEXT,
            category TEXT,
            thumbnail_url TEXT,
            watch_url TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            video_id TEXT,
            ts TEXT,
            view_count INTEGER,
            like_count INTEGER,
            comment_count INTEGER,
            PRIMARY KEY (video_id, ts)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    _ensure_column(conn, "saved_videos", "thumbnail_url", "TEXT")
    _ensure_column(conn, "saved_videos", "watch_url", "TEXT")
    _ensure_column(conn, "tracked_videos", "thumbnail_url", "TEXT")
    _ensure_column(conn, "tracked_videos", "watch_url", "TEXT")
    conn.commit()


def get_connection(db_path: str = DB_NAME) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def upsert_saved_video(
    conn: sqlite3.Connection,
    video_id: str,
    title: str,
    channel: str,
    published_at: str,
    duration_sec: int,
    views: int,
    likes: int,
    comments: int,
    tags: str,
    thumbnail_url: str,
    watch_url: str,
    region: str,
    keyword: str,
    category: str,
    saved_ts: str,
    notes: str = "",
) -> None:
    conn.execute(
        """
        INSERT INTO saved_videos (
            video_id, title, channel, publishedAt, duration_sec, views, likes,
            comments, tags, thumbnail_url, watch_url, region, keyword, category, saved_ts, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            title=excluded.title,
            channel=excluded.channel,
            publishedAt=excluded.publishedAt,
            duration_sec=excluded.duration_sec,
            views=excluded.views,
            likes=excluded.likes,
            comments=excluded.comments,
            tags=excluded.tags,
            thumbnail_url=excluded.thumbnail_url,
            watch_url=excluded.watch_url,
            region=excluded.region,
            keyword=excluded.keyword,
            category=excluded.category,
            saved_ts=excluded.saved_ts,
            notes=excluded.notes
        """,
        (
            video_id,
            title,
            channel,
            published_at,
            duration_sec,
            views,
            likes,
            comments,
            tags,
            thumbnail_url,
            watch_url,
            region,
            keyword,
            category,
            saved_ts,
            notes,
        ),
    )
    conn.commit()


def add_tracked_video(
    conn: sqlite3.Connection,
    video_id: str,
    title: str,
    channel: str,
    added_ts: str,
    category: str,
    thumbnail_url: str,
    watch_url: str,
) -> None:
    conn.execute(
        """
        INSERT INTO tracked_videos (video_id, title, channel, added_ts, category, thumbnail_url, watch_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(video_id) DO UPDATE SET
            title=excluded.title,
            channel=excluded.channel,
            category=excluded.category,
            thumbnail_url=excluded.thumbnail_url,
            watch_url=excluded.watch_url
        """,
        (video_id, title, channel, added_ts, category, thumbnail_url, watch_url),
    )
    conn.commit()


def remove_tracked_video(conn: sqlite3.Connection, video_id: str) -> None:
    conn.execute("DELETE FROM tracked_videos WHERE video_id = ?", (video_id,))
    conn.commit()


def insert_snapshot(
    conn: sqlite3.Connection,
    video_id: str,
    ts: str,
    view_count: int,
    like_count: int,
    comment_count: int,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO snapshots
        (video_id, ts, view_count, like_count, comment_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        (video_id, ts, view_count, like_count, comment_count),
    )
    conn.commit()


def fetch_saved_videos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM saved_videos ORDER BY saved_ts DESC").fetchall()


def fetch_saved_videos_filtered(
    conn: sqlite3.Connection,
    category: Optional[str],
    keyword: Optional[str],
    region: Optional[str],
    min_views: Optional[int],
    shorts_only: bool,
) -> list[sqlite3.Row]:
    clauses = []
    params: list[object] = []
    if category and category != "All":
        clauses.append("category = ?")
        params.append(category)
    if keyword:
        clauses.append("keyword LIKE ?")
        params.append(f"%{keyword}%")
    if region and region != "All":
        clauses.append("region = ?")
        params.append(region)
    if min_views is not None:
        clauses.append("views >= ?")
        params.append(min_views)
    if shorts_only:
        clauses.append("duration_sec <= 60")
    where = " AND ".join(clauses)
    query = "SELECT * FROM saved_videos"
    if where:
        query += f" WHERE {where}"
    query += " ORDER BY saved_ts DESC"
    return conn.execute(query, params).fetchall()


def update_saved_notes(conn: sqlite3.Connection, video_id: str, notes: str) -> None:
    conn.execute(
        "UPDATE saved_videos SET notes = ? WHERE video_id = ?", (notes, video_id)
    )
    conn.commit()


def fetch_tracked_videos(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM tracked_videos ORDER BY added_ts DESC").fetchall()


def fetch_snapshots(conn: sqlite3.Connection, video_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM snapshots WHERE video_id = ? ORDER BY ts ASC", (video_id,)
    ).fetchall()


def fetch_latest_snapshot(conn: sqlite3.Connection, video_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM snapshots WHERE video_id = ? ORDER BY ts DESC LIMIT 1",
        (video_id,),
    ).fetchone()


def get_setting(conn: sqlite3.Connection, key: str, default: str) -> str:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if not row:
        set_setting(conn, key, default)
        return default
    return str(row["value"])


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def iter_rows(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [dict(row) for row in rows]
