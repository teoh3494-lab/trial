from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

import db
import scoring
from gemini_ai import generate_ideas
from yt_api import HttpError, VideoDetail, build_youtube_client, fetch_video_details, search_videos

load_dotenv()

st.set_page_config(page_title="Tube Trends Buddy", layout="wide")


def parse_api_keys(raw: str) -> list[str]:
    return [key.strip() for key in raw.split(",") if key.strip()]


@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    return db.get_connection()


@st.cache_resource
def get_youtube_client(api_key: str):
    return build_youtube_client(api_key)


@st.cache_data(ttl=300)
def cached_search(
    api_key: str,
    query: str,
    region_code: str,
    relevance_language: str,
    published_after: str | None,
    order: str,
    video_duration: str,
    max_results: int,
) -> list[str]:
    yt = get_youtube_client(api_key)
    return search_videos(
        yt,
        query=query,
        region_code=region_code,
        relevance_language=relevance_language,
        published_after=published_after,
        order=order,
        video_duration=video_duration,
        max_results=max_results,
    )


@st.cache_data(ttl=300)
def cached_video_details(api_key: str, video_ids: Iterable[str]) -> list[VideoDetail]:
    yt = get_youtube_client(api_key)
    return fetch_video_details(yt, list(video_ids))


def safe_api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except HttpError as exc:
        st.error(
            "YouTube API error. Check quota/key/params. Details: "
            f"{exc}"
        )
        return None


def format_video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def compute_vph(snapshots: pd.DataFrame, hours: int) -> float:
    if len(snapshots) < 2:
        return 0.0
    cutoff = snapshots["ts"].max() - pd.Timedelta(hours=hours)
    window = snapshots[snapshots["ts"] >= cutoff]
    if len(window) < 2:
        window = snapshots
    first = window.iloc[0]
    last = window.iloc[-1]
    delta_views = last["view_count"] - first["view_count"]
    delta_hours = (last["ts"] - first["ts"]).total_seconds() / 3600
    if delta_hours <= 0:
        return 0.0
    return max(0.0, delta_views / delta_hours)


def get_vph_for_video(video_id: str, hours: int = 1) -> float:
    snapshots = pd.DataFrame(db.iter_rows(db.fetch_snapshots(conn, video_id)))
    if snapshots.empty:
        return 0.0
    snapshots["ts"] = pd.to_datetime(snapshots["ts"], utc=True)
    return compute_vph(snapshots, hours)


def _format_srt_time(seconds: float) -> str:
    millis = int((seconds - int(seconds)) * 1000)
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_count(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def format_age(published_at: str) -> str:
    if not published_at:
        return "-"
    try:
        published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return published_at
    delta = datetime.now(timezone.utc) - published_dt
    if delta.days >= 365:
        years = delta.days // 365
        return f"{years}y ago"
    if delta.days >= 30:
        months = delta.days // 30
        return f"{months}mo ago"
    if delta.days >= 1:
        return f"{delta.days}d ago"
    hours = int(delta.total_seconds() // 3600)
    if hours >= 1:
        return f"{hours}h ago"
    minutes = int(delta.total_seconds() // 60)
    return f"{max(1, minutes)}m ago"


def extract_keywords(*fields: str, limit: int = 12) -> list[str]:
    text = " ".join(fields)
    cleaned = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
    words = [word.lower() for word in cleaned.split() if len(word) >= 4]
    stopwords = {
        "yang",
        "dari",
        "with",
        "this",
        "that",
        "untuk",
        "dalam",
        "akan",
        "video",
        "channel",
        "anda",
        "kamu",
        "dengan",
        "yang",
        "pada",
        "dari",
        "the",
        "and",
        "for",
        "atau",
        "how",
        "what",
        "your",
    }
    keywords = [word for word in words if word not in stopwords]
    counts: dict[str, int] = {}
    for word in keywords:
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [word for word, _ in ranked[:limit]]


def build_cache_key(name: str, size: int, mtime: float) -> str:
    raw = f"{name}:{size}:{mtime}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


OAUTH_SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]


def get_oauth_credentials(oauth_dir: Path) -> Credentials | None:
    client_secret_path = oauth_dir / "client_secret.json"
    token_path = oauth_dir / "token.json"
    if not client_secret_path.exists():
        return None
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(
            str(token_path),
            scopes=OAUTH_SCOPES,
        )
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secret_path),
            scopes=OAUTH_SCOPES,
        )
        creds = flow.run_local_server(port=0)
        oauth_dir.mkdir(exist_ok=True)
        token_path.write_text(creds.to_json())
    return creds


def fetch_latest_channel_uploads(yt, channel_id: str, max_results: int = 5) -> list[dict]:
    request = (
        yt.search()
        .list(
            part="snippet",
            type="video",
            channelId=channel_id,
            order="date",
            maxResults=max_results,
        )
    )
    response = request.execute()
    items = response.get("items", [])
    uploads = []
    for item in items:
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
        uploads.append(
            {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "published_at": snippet.get("publishedAt", ""),
            }
        )
    return uploads


@st.dialog("Analitik Video")
def show_video_dialog(
    video: dict,
    config: dict,
    conn: sqlite3.Connection,
    default_tab: str | None = None,
) -> None:
    st.markdown(
        f"**{video['title']}**  \n{video['channel']}"
    )
    thumb_col, info_col = st.columns([1, 2])
    with thumb_col:
        st.image(video["thumbnail"], use_container_width=True)
    with info_col:
        vph_1h = get_vph_for_video(video["video_id"], 1)
        views = int(video["views"])
        likes = int(video["likes"])
        hours_since_publish = 0.0
        if video.get("publishedAt"):
            published_dt = datetime.fromisoformat(video["publishedAt"].replace("Z", "+00:00"))
            hours_since_publish = max(
                0.0, (datetime.now(timezone.utc) - published_dt).total_seconds() / 3600
            )
        likes_per_view = likes / views if views else 0.0
        breakdown = scoring.compute_viral_score(
            vph_1h=vph_1h,
            views=views,
            hours_since_publish=hours_since_publish,
            likes_per_view=likes_per_view,
        )
        metric_cols = st.columns(6)
        metric_cols[0].metric("Views", format_count(views))
        metric_cols[1].metric("VPH/hr", f"{vph_1h:.2f}")
        metric_cols[2].metric("Duration", format_duration(int(video["duration_sec"])))
        metric_cols[3].metric("Published", format_age(video["publishedAt"]))
        metric_cols[4].metric("Likes", format_count(likes))
        metric_cols[5].metric("Viral Score", f"{breakdown.total_score:.1f}")

    tabs = st.tabs(["Performance", "Kata Kunci", "Video Channel", "Transkrip"])
    if default_tab:
        st.caption(f"Tab default: {default_tab}.")

    with tabs[0]:
        snapshots = pd.DataFrame(db.iter_rows(db.fetch_snapshots(conn, video["video_id"])))
        if not snapshots.empty:
            snapshots["ts"] = pd.to_datetime(snapshots["ts"], utc=True)
            snapshots = snapshots.sort_values("ts")
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=snapshots["ts"],
                    y=snapshots["view_count"],
                    mode="lines+markers",
                    name="Views",
                )
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No snapshots yet for this video.")

    with tabs[1]:
        keywords = extract_keywords(
            video.get("title", ""),
            video.get("description", ""),
            video.get("tags", ""),
        )
        if keywords:
            st.write("Saran kata kunci:")
            st.write(", ".join(keywords))
        else:
            st.info("No keyword suggestions available.")
        if config.get("gemini_key"):
            st.caption("Gemini tersedia untuk ide tambahan jika diperlukan.")

    with tabs[2]:
        channel_link = (
            f"https://www.youtube.com/channel/{video['channel_id']}"
            if video.get("channel_id")
            else f"https://www.youtube.com/results?search_query={video['channel']}"
        )
        st.markdown(f"**Channel:** {video['channel']}")
        st.markdown(f"[Buka Channel]({channel_link})")
        if config.get("yt_key") and video.get("channel_id"):
            with st.spinner("Memuat upload terbaru channel..."):
                yt = get_youtube_client(config["yt_key"])
                try:
                    uploads = fetch_latest_channel_uploads(yt, video["channel_id"], max_results=5)
                except HttpError as exc:
                    st.error(f"YouTube API error: {exc}")
                    uploads = []
            if uploads:
                st.markdown("**Upload terbaru:**")
                for item in uploads:
                    link = format_video_url(item["video_id"])
                    st.markdown(f"- [{item['title']}]({link})")
            else:
                st.info("Tidak ada data upload terbaru.")
        else:
            st.info("Tambahkan API key untuk melihat upload terbaru channel.")

    with tabs[3]:
        st.markdown(
            "Mode A (OAuth): `captions.list` hanya daftar track, teks asli harus melalui "
            "`captions.download`."
        )
        oauth_dir = Path("oauth")
        cache_dir = Path("cache_srt")
        cache_dir.mkdir(exist_ok=True)
        oauth_ready = (oauth_dir / "client_secret.json").exists()
        if not oauth_ready:
            st.warning(
                "OAuth belum dikonfigurasi. Tambahkan `oauth/client_secret.json` untuk mencoba "
                "download caption dari YouTube."
            )

        track_key = f"dialog_tracks_{video['video_id']}"
        if st.button("Cek Caption Tracks (OAuth)", disabled=not oauth_ready):
            with st.spinner("Mengambil daftar caption..."):
                try:
                    creds = get_oauth_credentials(oauth_dir)
                    if not creds:
                        st.error("OAuth credentials tidak tersedia.")
                    else:
                        yt_oauth = build(
                            "youtube", "v3", credentials=creds, cache_discovery=False
                        )
                        response = (
                            yt_oauth.captions()
                            .list(part="snippet", videoId=video["video_id"])
                            .execute()
                        )
                        items = response.get("items", [])
                        tracks = [
                            {
                                "id": item["id"],
                                "language": item["snippet"].get("language"),
                                "name": item["snippet"].get("name", ""),
                                "trackKind": item["snippet"].get("trackKind", ""),
                            }
                            for item in items
                        ]
                        st.session_state[track_key] = tracks
                except HttpError as exc:
                    if exc.resp.status == 403:
                        st.error(
                            "Izin tidak cukup untuk captions API. Gunakan mode transkrip lokal."
                        )
                    else:
                        st.error(f"YouTube API error: {exc}")

        tracks = st.session_state.get(track_key, [])
        srt_text_key = f"srt_text_{video['video_id']}"
        if tracks:
            st.dataframe(pd.DataFrame(tracks), use_container_width=True)
            track_id = st.selectbox(
                "Pilih track untuk download",
                [track["id"] for track in tracks],
                key=f"track_select_{video['video_id']}",
            )
            if st.button("Download Caption (SRT)", key=f"download_{video['video_id']}"):
                with st.spinner("Downloading captions..."):
                    try:
                        creds = get_oauth_credentials(oauth_dir)
                        if not creds:
                            st.error("OAuth credentials tidak tersedia.")
                        else:
                            yt_oauth = build(
                                "youtube", "v3", credentials=creds, cache_discovery=False
                            )
                            request = yt_oauth.captions().download(
                                id=track_id, tfmt="srt"
                            )
                            buffer = io.BytesIO()
                            downloader = MediaIoBaseDownload(buffer, request)
                            done = False
                            while not done:
                                _, done = downloader.next_chunk()
                            buffer.seek(0)
                            srt_text = buffer.read().decode("utf-8", errors="ignore")
                            st.session_state[srt_text_key] = srt_text
                    except HttpError as exc:
                        if exc.resp.status == 403:
                            st.error(
                                "Izin tidak cukup untuk captions API. Gunakan mode transkrip lokal."
                            )
                        else:
                            st.error(f"YouTube API error: {exc}")

        st.markdown("---")
        st.markdown("Mode B: Local Transcribe (Upload File)")
        uploaded = st.file_uploader(
            "Upload audio/video (mp3/wav/mp4)",
            type=["mp3", "wav", "mp4"],
            key=f"upload_{video['video_id']}",
        )
        whisper_spec = importlib.util.find_spec("faster_whisper")
        if whisper_spec is None:
            st.info(
                "Local transcription membutuhkan dependency tambahan. Install dengan:\n"
                "`pip install -r requirements-extra.txt`"
            )
        if uploaded and whisper_spec is not None:
            cache_uploads = Path("cache_uploads")
            cache_uploads.mkdir(exist_ok=True)
            upload_path = cache_uploads / uploaded.name
            upload_path.write_bytes(uploaded.getbuffer())
            file_stat = upload_path.stat()
            cache_key = build_cache_key(uploaded.name, file_stat.st_size, file_stat.st_mtime)
            srt_path = cache_dir / f"{cache_key}.srt"

            if st.button("Transcribe to SRT", key=f"transcribe_{video['video_id']}"):
                if srt_path.exists():
                    st.success("Menggunakan cache transkrip.")
                else:
                    with st.spinner("Transcribing..."):
                        from faster_whisper import WhisperModel  # type: ignore

                        model = WhisperModel("base", device="cpu", compute_type="int8")
                        segments, _ = model.transcribe(str(upload_path))
                        srt_lines = []
                        for idx, segment in enumerate(segments, start=1):
                            start = _format_srt_time(segment.start)
                            end = _format_srt_time(segment.end)
                            srt_lines.append(
                                f"{idx}\n{start} --> {end}\n{segment.text.strip()}\n"
                            )
                        srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
                        st.success("Transkrip selesai dibuat.")
                st.session_state[srt_text_key] = srt_path.read_text(encoding="utf-8")

        srt_text = st.session_state.get(srt_text_key, "")
        if srt_text:
            st.text_area(
                "Transkrip (SRT)",
                value=srt_text,
                height=200,
                key=f"srt_area_{video['video_id']}",
            )
            if st.button("Copy Transkrip", key=f"copy_{video['video_id']}"):
                st.toast("Transkrip siap dicopy.")
            st.download_button(
                "Download (.srt)",
                data=srt_text.encode("utf-8"),
                file_name=f"{video['video_id']}.srt",
                mime="text/plain",
            )
        else:
            st.info("Belum ada transkrip. Coba mode OAuth atau upload file lokal.")


def sidebar_config() -> dict:
    st.sidebar.title("Configuration")
    api_key_env = os.getenv("YT_API_KEY", "")
    api_keys_env = os.getenv("YT_API_KEYS", "")
    default_keys = api_keys_env or api_key_env
    api_keys_input = st.sidebar.text_input(
        "YouTube API Keys (comma-separated)", value=default_keys, type="password"
    )
    api_keys = parse_api_keys(api_keys_input)
    if not api_keys:
        st.sidebar.warning("Enter YT API key to enable YouTube features.")
    active_key = st.sidebar.selectbox(
        "Active API Key",
        options=api_keys if api_keys else [""],
        format_func=lambda k: "(none)" if not k else f"Key {api_keys.index(k) + 1}",
    )
    gemini_key = st.sidebar.text_input(
        "Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Quota tips**")
    st.sidebar.info(
        "search.list is expensive, videos.list is cheaper. Cache results (TTL 300s) "
        "and avoid repeated searches to save quota."
    )
    return {
        "yt_key": active_key,
        "gemini_key": gemini_key,
    }


config = sidebar_config()
conn = get_db_connection()

st.title("Tube Trends Buddy")


search_tab, vph_tab, heatmap_tab, market_tab, idea_tab, bank_tab, tracked_tab, captions_tab = st.tabs(
    [
        "Super Search",
        "VPH Tracking",
        "Best Upload Time",
        "Market Dashboard",
        "Idea Generator",
        "Bank Video",
        "Tracked",
        "Captions",
    ]
)


with search_tab:
    st.subheader("Super Search Engine")
    col1, col2, col3 = st.columns(3)
    with col1:
        query = st.text_input("Keyword (q)")
        region_code = st.text_input("regionCode", value="ID")
        relevance_language = st.text_input("relevanceLanguage", value="id")
    with col2:
        published_days = st.number_input("Published within last X days", min_value=1, max_value=365, value=7)
        order = st.selectbox("Order", ["date", "relevance", "viewCount", "rating"])
        video_duration = st.selectbox("videoDuration", ["any", "short", "medium", "long"])
    with col3:
        max_results = st.slider("Max results", min_value=5, max_value=50, value=20)
        shorts_only = st.checkbox("Shorts asli (<= 60s)")
        category_input = st.text_input("Category / Niche", value="General")

    if "search_running" not in st.session_state:
        st.session_state["search_running"] = False
    search_button = st.button(
        "Search",
        disabled=not (config["yt_key"] and query) or st.session_state["search_running"],
    )

    if search_button:
        st.session_state["search_running"] = True
        published_after = (
            datetime.now(timezone.utc) - timedelta(days=published_days)
        ).isoformat()
        with st.spinner("Searching YouTube..."):
            ids = safe_api_call(
                cached_search,
                config["yt_key"],
                query,
                region_code,
                relevance_language,
                published_after,
                order,
                video_duration,
                max_results,
            )
            if ids is None:
                ids = []
            details = safe_api_call(cached_video_details, config["yt_key"], ids)
            if details is None:
                details = []
        st.session_state["search_running"] = False
        rows = []
        for detail in details:
            if shorts_only and detail.duration_sec > 60:
                continue
            vph_1h = get_vph_for_video(detail.video_id, 1)
            rows.append(
                {
                    "thumbnail": detail.thumbnail_url,
                    "video_id": detail.video_id,
                    "title": detail.title,
                    "channel": detail.channel,
                    "channel_id": detail.channel_id,
                    "publishedAt": detail.published_at,
                    "duration_sec": detail.duration_sec,
                    "views": detail.views,
                    "likes": detail.likes,
                    "comments": detail.comments,
                    "VPH_1h": round(vph_1h, 2),
                    "tags": ", ".join(detail.tags),
                    "description": detail.description,
                    "watch_url": format_video_url(detail.video_id),
                }
            )
        if rows:
            results_df = pd.DataFrame(rows)
            st.session_state["search_results"] = rows
            st.session_state["last_search_results"] = [
                {
                    "title": row["title"],
                    "channel": row["channel"],
                    "views": row["views"],
                    "duration": row["duration_sec"],
                }
                for row in rows
            ]
            if "selected_video_id" not in st.session_state:
                st.session_state["selected_video_id"] = rows[0]["video_id"]
            if st.session_state["selected_video_id"] not in results_df["video_id"].tolist():
                st.session_state["selected_video_id"] = rows[0]["video_id"]

            st.markdown("### Preview Panel")
            preview_id = st.session_state.get("selected_video_id")
            preview_row = next((row for row in rows if row["video_id"] == preview_id), None)
            if preview_row:
                st.video(preview_row["watch_url"])
                metrics_cols = st.columns(6)
                metrics_cols[0].metric("Views", format_count(int(preview_row["views"])))
                metrics_cols[1].metric("VPH/hr", f"{preview_row['VPH_1h']:.2f}")
                metrics_cols[2].metric("Duration", format_duration(int(preview_row["duration_sec"])))
                metrics_cols[3].metric("Published", format_age(preview_row["publishedAt"]))
                metrics_cols[4].metric("Likes", format_count(int(preview_row["likes"])))
                metrics_cols[5].metric("Comments", format_count(int(preview_row["comments"])))
            else:
                st.info("Pick a video card to preview.")

            st.markdown("### Search Results")
            selected_ids: list[str] = []
            cards_per_row = 4
            for start in range(0, len(rows), cards_per_row):
                cols = st.columns(cards_per_row)
                for col, row in zip(cols, rows[start : start + cards_per_row]):
                    with col:
                        with st.container(border=True):
                            st.image(row["thumbnail"], use_container_width=True)
                            if st.button(
                                f"🎬 {row['title']}",
                                key=f"title_{row['video_id']}",
                            ):
                                st.session_state["selected_video_id"] = row["video_id"]
                            st.caption(row["channel"])
                            metric_row_a = st.columns(3)
                            metric_row_a[0].metric("Views", format_count(int(row["views"])))
                            metric_row_a[1].metric("VPH/hr", f"{row['VPH_1h']:.2f}")
                            metric_row_a[2].metric(
                                "Duration", format_duration(int(row["duration_sec"]))
                            )
                            metric_row_b = st.columns(3)
                            metric_row_b[0].metric(
                                "Published", format_age(row["publishedAt"])
                            )
                            metric_row_b[1].metric("Likes", format_count(int(row["likes"])))
                            metric_row_b[2].metric("Comments", format_count(int(row["comments"])))
                            if st.checkbox("Select", key=f"select_{row['video_id']}"):
                                selected_ids.append(row["video_id"])

                            action_cols = st.columns(3)
                            with action_cols[0]:
                                if st.button(
                                    "📊 Analisis",
                                    key=f"analyze_{row['video_id']}",
                                ):
                                    show_video_dialog(row, config, conn)
                            with action_cols[1]:
                                if st.button(
                                    "📝 Transkrip",
                                    key=f"transcript_{row['video_id']}",
                                ):
                                    show_video_dialog(row, config, conn, default_tab="Transkrip")
                            with action_cols[2]:
                                if st.button(
                                    "💾 Simpan",
                                    key=f"save_{row['video_id']}",
                                ):
                                    now_ts = datetime.now(timezone.utc).isoformat()
                                    db.upsert_saved_video(
                                        conn,
                                        video_id=row["video_id"],
                                        title=row["title"],
                                        channel=row["channel"],
                                        published_at=row["publishedAt"],
                                        duration_sec=int(row["duration_sec"]),
                                        views=int(row["views"]),
                                        likes=int(row["likes"]),
                                        comments=int(row["comments"]),
                                        tags=row["tags"],
                                        thumbnail_url=row["thumbnail"],
                                        watch_url=row["watch_url"],
                                        region=region_code,
                                        keyword=query,
                                        category=category_input,
                                        saved_ts=now_ts,
                                        notes="",
                                    )
                                    st.toast("Saved to bank.")

            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Save to Bank (Selected)"):
                    now_ts = datetime.now(timezone.utc).isoformat()
                    for _, row in results_df.iterrows():
                        if row["video_id"] in selected_ids:
                            db.upsert_saved_video(
                                conn,
                                video_id=row["video_id"],
                                title=row["title"],
                                channel=row["channel"],
                                published_at=row["publishedAt"],
                                duration_sec=int(row["duration_sec"]),
                                views=int(row["views"]),
                                likes=int(row["likes"]),
                                comments=int(row["comments"]),
                                tags=row["tags"],
                                thumbnail_url=row["thumbnail"],
                                watch_url=row["watch_url"],
                                region=region_code,
                                keyword=query,
                                category=category_input,
                                saved_ts=now_ts,
                                notes="",
                            )
                    st.success("Saved to bank.")
            with col_b:
                if st.button("Add to Tracking (Selected)"):
                    now_ts = datetime.now(timezone.utc).isoformat()
                    for _, row in results_df.iterrows():
                        if row["video_id"] in selected_ids:
                            db.add_tracked_video(
                                conn,
                                video_id=row["video_id"],
                                title=row["title"],
                                channel=row["channel"],
                                added_ts=now_ts,
                                category=category_input,
                                thumbnail_url=row["thumbnail"],
                                watch_url=row["watch_url"],
                            )
                    st.success("Added to tracking.")
        else:
            st.info("No results to display.")


with vph_tab:
    st.subheader("VPH Detector & Tracking")
    tracked_rows = db.fetch_tracked_videos(conn)
    if not tracked_rows:
        st.info("No tracked videos. Add from Super Search.")
    else:
        tracked_df = pd.DataFrame(db.iter_rows(tracked_rows))
        if st.button("Snapshot Now (All Tracked)", disabled=not config["yt_key"]):
            with st.spinner("Fetching latest stats..."):
                details = safe_api_call(
                    cached_video_details,
                    config["yt_key"],
                    tracked_df["video_id"].tolist(),
                )
            if details:
                now_ts = datetime.now(timezone.utc).isoformat()
                for detail in details:
                    db.insert_snapshot(
                        conn,
                        video_id=detail.video_id,
                        ts=now_ts,
                        view_count=detail.views,
                        like_count=detail.likes,
                        comment_count=detail.comments,
                    )
                st.success("Snapshot saved.")

        ranking_rows = []
        for _, row in tracked_df.iterrows():
            snapshots = pd.DataFrame(db.iter_rows(db.fetch_snapshots(conn, row["video_id"])))
            if snapshots.empty:
                vph_1h = vph_6h = vph_24h = 0.0
                latest_views = 0
            else:
                snapshots["ts"] = pd.to_datetime(snapshots["ts"], utc=True)
                vph_1h = compute_vph(snapshots, 1)
                vph_6h = compute_vph(snapshots, 6)
                vph_24h = compute_vph(snapshots, 24)
                latest_views = int(snapshots.iloc[-1]["view_count"])
            ranking_rows.append(
                {
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "channel": row["channel"],
                    "VPH_1h": round(vph_1h, 2),
                    "VPH_6h": round(vph_6h, 2),
                    "VPH_24h": round(vph_24h, 2),
                    "latest_views": latest_views,
                }
            )
        ranking_df = pd.DataFrame(ranking_rows).sort_values(by="VPH_1h", ascending=False)
        st.dataframe(ranking_df, use_container_width=True)

        selected_video = st.selectbox(
            "Select video for time-series",
            ranking_df["video_id"].tolist(),
        )
        if selected_video:
            snapshots = pd.DataFrame(
                db.iter_rows(db.fetch_snapshots(conn, selected_video))
            )
            if not snapshots.empty:
                snapshots["ts"] = pd.to_datetime(snapshots["ts"], utc=True)
                snapshots = snapshots.sort_values("ts")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=snapshots["ts"],
                        y=snapshots["view_count"],
                        mode="lines+markers",
                        name="Views",
                    )
                )
                vph_series = [
                    compute_vph(snapshots.iloc[: i + 1], 1) for i in range(len(snapshots))
                ]
                fig.add_trace(
                    go.Scatter(
                        x=snapshots["ts"],
                        y=vph_series,
                        mode="lines+markers",
                        name="VPH_1h",
                        yaxis="y2",
                    )
                )
                fig.update_layout(
                    yaxis2=dict(overlaying="y", side="right", title="VPH_1h"),
                    title="Views & VPH trend",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No snapshots yet.")


with heatmap_tab:
    st.subheader("Best Upload Time Heatmap")
    col1, col2, col3 = st.columns(3)
    with col1:
        channel_id = st.text_input("Channel ID (UCxxxx)")
    with col2:
        num_videos = st.slider("Number of latest videos", min_value=10, max_value=200, value=50)
    with col3:
        tz_choice = st.selectbox("Timezone", ["UTC", "Local"])

    if st.button("Analyze", disabled=not (config["yt_key"] and channel_id)):
        with st.spinner("Fetching channel uploads..."):
            yt = get_youtube_client(config["yt_key"])
            try:
                collected_ids = []
                next_token = None
                while len(collected_ids) < num_videos:
                    request = (
                        yt.search()
                        .list(
                            part="snippet",
                            type="video",
                            channelId=channel_id,
                            order="date",
                            maxResults=min(50, num_videos - len(collected_ids)),
                            pageToken=next_token,
                        )
                    )
                    response = request.execute()
                    items = response.get("items", [])
                    collected_ids.extend(
                        [item["id"]["videoId"] for item in items if "videoId" in item["id"]]
                    )
                    next_token = response.get("nextPageToken")
                    if not next_token:
                        break
            except HttpError as exc:
                st.error(f"YouTube API error: {exc}")
                collected_ids = []

            details = safe_api_call(cached_video_details, config["yt_key"], collected_ids)
            if details is None:
                details = []

        if details:
            data = []
            for detail in details:
                published_at = datetime.fromisoformat(detail.published_at.replace("Z", "+00:00"))
                if tz_choice == "Local":
                    published_at = published_at.astimezone()
                data.append(
                    {
                        "day": published_at.strftime("%A"),
                        "hour": published_at.hour,
                    }
                )
            df = pd.DataFrame(data)
            heatmap = (
                df.groupby(["day", "hour"]).size().reset_index(name="count")
            )
            days_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            heatmap["day"] = pd.Categorical(
                heatmap["day"], categories=days_order, ordered=True
            )
            pivot = heatmap.pivot(index="day", columns="hour", values="count").fillna(0)
            fig = px.imshow(
                pivot,
                aspect="auto",
                color_continuous_scale="Blues",
                labels=dict(x="Hour", y="Day", color="Uploads"),
            )
            st.plotly_chart(fig, use_container_width=True)

            top_hours = (
                df["hour"].value_counts().sort_values(ascending=False).head(3).index.tolist()
            )
            top_days = (
                df["day"].value_counts().loc[days_order].sort_values(ascending=False).head(3).index.tolist()
            )
            st.write(f"Top 3 hours: {', '.join(str(h) for h in top_hours)}")
            st.write(f"Top 3 days: {', '.join(top_days)}")
            if top_hours:
                st.success(f"Recommended upload hour: {top_hours[0]}:00")
        else:
            st.info("No data available.")


with market_tab:
    st.subheader("Market Dashboard")
    rpm_value = float(db.get_setting(conn, "rpm_usd", "2.0"))
    rpm_input = st.number_input("RPM assumption (USD)", min_value=0.1, value=rpm_value, step=0.1)
    if st.button("Save RPM"):
        db.set_setting(conn, "rpm_usd", str(rpm_input))
        st.success("RPM saved.")

    tracked_rows = db.fetch_tracked_videos(conn)
    if tracked_rows:
        tracked_df = pd.DataFrame(db.iter_rows(tracked_rows))
        details_map: dict[str, VideoDetail] = {}
        if config["yt_key"]:
            details = safe_api_call(
                cached_video_details,
                config["yt_key"],
                tracked_df["video_id"].tolist(),
            )
            if details:
                details_map = {detail.video_id: detail for detail in details}
        latest_stats = []
        for _, row in tracked_df.iterrows():
            latest = db.fetch_latest_snapshot(conn, row["video_id"])
            if latest:
                views = int(latest["view_count"])
            else:
                views = 0
            hours_since_publish = 0.0
            detail = details_map.get(row["video_id"])
            if detail and detail.published_at:
                published = datetime.fromisoformat(detail.published_at.replace("Z", "+00:00"))
                hours_since_publish = max(
                    0.0, (datetime.now(timezone.utc) - published).total_seconds() / 3600
                )
            elif row.get("added_ts"):
                added_ts = datetime.fromisoformat(row["added_ts"].replace("Z", "+00:00"))
                hours_since_publish = max(
                    0.0, (datetime.now(timezone.utc) - added_ts).total_seconds() / 3600
                )
            vph_1h = 0.0
            snapshots_df = pd.DataFrame(
                db.iter_rows(db.fetch_snapshots(conn, row["video_id"]))
            )
            if not snapshots_df.empty:
                snapshots_df["ts"] = pd.to_datetime(snapshots_df["ts"], utc=True)
                vph_1h = compute_vph(snapshots_df, 1)
            likes_per_view = 0.0
            if latest and views > 0:
                likes_per_view = int(latest["like_count"]) / views
            breakdown = scoring.compute_viral_score(
                vph_1h=vph_1h,
                views=views,
                hours_since_publish=hours_since_publish,
                likes_per_view=likes_per_view,
            )
            latest_stats.append(
                {
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "views": views,
                    "est_revenue_usd": round((views / 1000) * rpm_input, 2),
                    "viral_score": breakdown.total_score,
                }
            )
        stats_df = pd.DataFrame(latest_stats).sort_values(
            by="viral_score", ascending=False
        )
        st.dataframe(stats_df, use_container_width=True)
        with st.expander("Viral Rate Score formula"):
            st.markdown(
                """
                **Viral Score (0-100)** combines:
                - VPH_1h (45%)
                - Total views (20%)
                - Recency (20%)
                - Engagement proxy (likes per view, 15%)

                Each component is normalized to 0-100 based on heuristic max values.
                """
            )
    else:
        st.info("No tracked videos yet.")


with idea_tab:
    st.subheader("Gemini Idea Generator")
    context_source = st.selectbox("Context source", ["Search results", "Bank videos"])
    top_n = st.slider("Use top N", min_value=3, max_value=20, value=10)
    style = st.selectbox("Style", ["viral what-if", "edukasi", "sci-fi story"])
    language = st.selectbox("Output language", ["ID", "EN"])
    count = st.slider("Number of ideas", min_value=1, max_value=5, value=3)

    if not config["gemini_key"]:
        st.warning("Gemini API key missing. Add GEMINI_API_KEY to use generator.")

    if st.button("Generate Ideas", disabled=not config["gemini_key"]):
        context_items = []
        if context_source == "Search results":
            cached = st.session_state.get("last_search_results", [])
            context_items = cached[:top_n]
        else:
            bank_rows = db.fetch_saved_videos(conn)
            for row in bank_rows[:top_n]:
                context_items.append(
                    {
                        "title": row["title"],
                        "channel": row["channel"],
                        "views": row["views"],
                        "duration": row["duration_sec"],
                    }
                )

        with st.spinner("Generating ideas..."):
            response = generate_ideas(
                api_key=config["gemini_key"],
                context=context_items,
                style=style,
                language=language,
                count=count,
            )
        try:
            ideas = json.loads(response)
        except json.JSONDecodeError:
            st.error("Failed to parse Gemini response. Raw output:")
            st.code(response)
        else:
            for idx, idea in enumerate(ideas, start=1):
                st.markdown(f"### Idea {idx}")
                st.write(f"Hook: {idea.get('hook')}")
                st.write(f"Premise: {idea.get('premise')}")
                st.write(f"CTA: {idea.get('cta')}")
                st.write("Titles:")
                st.write("\n".join([f"- {t}" for t in idea.get("titles", [])]))
                st.write(f"Thumbnail: {idea.get('thumbnail')}")


with bank_tab:
    st.subheader("Bank Video")
    categories = ["All"] + sorted(
        {row["category"] for row in db.fetch_saved_videos(conn) if row["category"]}
    )
    filter_category = st.selectbox("Category", categories)
    filter_keyword = st.text_input("Keyword contains")
    filter_region = st.text_input("Region (ISO)")
    filter_min_views = st.number_input("Min views", min_value=0, value=0)
    filter_shorts = st.checkbox("Shorts only (<=60s)")

    filtered_rows = db.fetch_saved_videos_filtered(
        conn,
        category=filter_category,
        keyword=filter_keyword,
        region=filter_region if filter_region else None,
        min_views=filter_min_views if filter_min_views > 0 else None,
        shorts_only=filter_shorts,
    )
    if filtered_rows:
        bank_df = pd.DataFrame(db.iter_rows(filtered_rows))
        bank_df["watch_url"] = bank_df["watch_url"].fillna(
            bank_df["video_id"].apply(format_video_url)
        )
        st.dataframe(
            bank_df,
            use_container_width=True,
            column_config={
                "thumbnail_url": st.column_config.ImageColumn("thumbnail"),
                "watch_url": st.column_config.LinkColumn("watch_url"),
            },
        )
        edit_id = st.selectbox("Edit notes for video", bank_df["video_id"].tolist())
        current_notes = (
            bank_df.loc[bank_df["video_id"] == edit_id, "notes"].iloc[0]
            if edit_id in bank_df["video_id"].tolist()
            else ""
        )
        notes_value = st.text_area("Notes", value=current_notes)
        if st.button("Save Notes"):
            db.update_saved_notes(conn, edit_id, notes_value)
            st.success("Notes updated.")
        st.download_button(
            "Export CSV",
            data=bank_df.to_csv(index=False).encode("utf-8"),
            file_name="bank_videos.csv",
            mime="text/csv",
        )
    else:
        st.info("No saved videos.")


with tracked_tab:
    st.subheader("Tracked Videos")
    tracked_rows = db.fetch_tracked_videos(conn)
    if tracked_rows:
        tracked_df = pd.DataFrame(db.iter_rows(tracked_rows))
        st.dataframe(tracked_df, use_container_width=True)
        remove_id = st.selectbox("Remove video", tracked_df["video_id"].tolist())
        if st.button("Remove from Tracking"):
            db.remove_tracked_video(conn, remove_id)
            st.success("Removed.")

        ranking_rows = []
        for _, row in tracked_df.iterrows():
            snapshots = pd.DataFrame(db.iter_rows(db.fetch_snapshots(conn, row["video_id"])))
            if snapshots.empty:
                vph_1h = 0.0
            else:
                snapshots["ts"] = pd.to_datetime(snapshots["ts"], utc=True)
                vph_1h = compute_vph(snapshots, 1)
            ranking_rows.append(
                {
                    "video_id": row["video_id"],
                    "title": row["title"],
                    "VPH_1h": round(vph_1h, 2),
                }
            )
        ranking_df = pd.DataFrame(ranking_rows).sort_values(by="VPH_1h", ascending=False)
        st.download_button(
            "Export VPH Ranking CSV",
            data=ranking_df.to_csv(index=False).encode("utf-8"),
            file_name="vph_ranking.csv",
            mime="text/csv",
        )
    else:
        st.info("No tracked videos.")


with captions_tab:
    st.subheader("Captions (OAuth required)")
    st.markdown(
        "Captions API only lists tracks; to download text you must use `captions.download` with "
        "OAuth and proper permissions."
    )
    cache_dir = Path("cache_srt")
    cache_dir.mkdir(exist_ok=True)
    oauth_dir = Path("oauth")
    client_secret_path = oauth_dir / "client_secret.json"

    st.markdown("### Mode A: YouTube Captions API (OAuth)")
    api_col1, api_col2 = st.columns([2, 1])
    with api_col1:
        api_video_id = st.text_input("Video ID for captions.list")
    with api_col2:
        api_format = st.selectbox("Download format", ["srt", "vtt"])

    oauth_ready = client_secret_path.exists()
    if not oauth_ready:
        st.warning(
            "OAuth not configured. Add ./oauth/client_secret.json to enable captions API mode."
        )

    if st.button(
        "List Caption Tracks",
        disabled=not (oauth_ready and api_video_id),
    ):
        with st.spinner("Listing caption tracks..."):
            try:
                creds = get_oauth_credentials(oauth_dir)
                if not creds:
                    st.error("OAuth credentials not available.")
                else:
                    yt_oauth = build(
                        "youtube", "v3", credentials=creds, cache_discovery=False
                    )
                    response = (
                        yt_oauth.captions()
                        .list(part="snippet", videoId=api_video_id)
                        .execute()
                    )
                    items = response.get("items", [])
                    if not items:
                        st.info("No caption tracks found.")
                    else:
                        tracks = [
                            {
                                "id": item["id"],
                                "language": item["snippet"].get("language"),
                                "name": item["snippet"].get("name", ""),
                                "trackKind": item["snippet"].get("trackKind", ""),
                            }
                            for item in items
                        ]
                        st.session_state["caption_tracks"] = tracks
                        st.dataframe(pd.DataFrame(tracks), use_container_width=True)
            except HttpError as exc:
                if exc.resp.status == 403:
                    st.error(
                        "Not properly authorized / insufficient permissions for captions. "
                        "Use local transcription mode below."
                    )
                else:
                    st.error(f"YouTube API error: {exc}")

    tracks = st.session_state.get("caption_tracks", [])
    if tracks:
        track_id = st.selectbox(
            "Select track to download", [track["id"] for track in tracks]
        )
        if st.button(
            "Download Caption Track",
            disabled=not oauth_ready,
        ):
            with st.spinner("Downloading captions..."):
                try:
                    creds = get_oauth_credentials(oauth_dir)
                    if not creds:
                        st.error("OAuth credentials not available.")
                    else:
                        yt_oauth = build(
                            "youtube", "v3", credentials=creds, cache_discovery=False
                        )
                        request = yt_oauth.captions().download(
                            id=track_id, tfmt=api_format
                        )
                        buffer = io.BytesIO()
                        downloader = MediaIoBaseDownload(buffer, request)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                        buffer.seek(0)
                        filename = cache_dir / f"{track_id}.{api_format}"
                        filename.write_bytes(buffer.read())
                        st.success(f"Saved to {filename}")
                except HttpError as exc:
                    if exc.resp.status == 403:
                        st.error(
                            "Not properly authorized / insufficient permissions for captions. "
                            "Use local transcription mode below."
                        )
                    else:
                        st.error(f"YouTube API error: {exc}")

    st.markdown("---")
    st.markdown("### Mode B: Local Transcribe (Upload File)")
    uploaded = st.file_uploader(
        "Upload audio/video (mp3/wav/mp4)", type=["mp3", "wav", "mp4"]
    )
    whisper_spec = importlib.util.find_spec("faster_whisper")
    if whisper_spec is None:
        st.info(
            "Local transcription requires extra dependencies. Install with:\n"
            "`pip install -r requirements-extra.txt`"
        )
    if uploaded and whisper_spec is not None:
        cache_uploads = Path("cache_uploads")
        cache_uploads.mkdir(exist_ok=True)
        upload_path = cache_uploads / uploaded.name
        upload_path.write_bytes(uploaded.getbuffer())
        file_stat = upload_path.stat()
        cache_key = build_cache_key(uploaded.name, file_stat.st_size, file_stat.st_mtime)
        srt_path = cache_dir / f"{cache_key}.srt"

        if st.button("Transcribe to SRT"):
            if srt_path.exists():
                st.success(f"Using cached transcript: {srt_path}")
            else:
                with st.spinner("Transcribing..."):
                    from faster_whisper import WhisperModel  # type: ignore

                    model = WhisperModel("base", device="cpu", compute_type="int8")
                    segments, _ = model.transcribe(str(upload_path))
                    srt_lines = []
                    for idx, segment in enumerate(segments, start=1):
                        start = _format_srt_time(segment.start)
                        end = _format_srt_time(segment.end)
                        srt_lines.append(f"{idx}\\n{start} --> {end}\\n{segment.text.strip()}\\n")
                    srt_path.write_text("\\n".join(srt_lines), encoding="utf-8")
                    st.success(f"Saved SRT: {srt_path}")
            st.download_button(
                "Download SRT",
                data=srt_path.read_bytes(),
                file_name=srt_path.name,
                mime="text/plain",
            )
