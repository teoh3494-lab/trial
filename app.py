from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

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
            rows.append(
                {
                    "video_id": detail.video_id,
                    "title": detail.title,
                    "channel": detail.channel,
                    "publishedAt": detail.published_at,
                    "duration_sec": detail.duration_sec,
                    "views": detail.views,
                    "likes": detail.likes,
                    "comments": detail.comments,
                    "tags": ", ".join(detail.tags),
                    "video_url": format_video_url(detail.video_id),
                }
            )
        if rows:
            results_df = pd.DataFrame(rows)
            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    "video_url": st.column_config.LinkColumn("video_url"),
                },
            )
            st.session_state["last_search_results"] = [
                {
                    "title": row["title"],
                    "channel": row["channel"],
                    "views": row["views"],
                    "duration": row["duration_sec"],
                }
                for row in rows
            ]

            selected_ids = st.multiselect(
                "Select videos", options=results_df["video_id"].tolist()
            )
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Save to Bank"):
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
                                region=region_code,
                                keyword=query,
                                category=category_input,
                                saved_ts=now_ts,
                                notes="",
                            )
                    st.success("Saved to bank.")
            with col_b:
                if st.button("Add to Tracking"):
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
        bank_df["video_url"] = bank_df["video_id"].apply(format_video_url)
        st.dataframe(
            bank_df,
            use_container_width=True,
            column_config={
                "video_url": st.column_config.LinkColumn("video_url"),
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
    st.info(
        "Captions API requires OAuth and permissions for the target videos. "
        "If you have client_secrets.json and token, place them locally and use your own script."
    )
