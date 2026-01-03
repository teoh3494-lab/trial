from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv

import db
from yt_api import HttpError, build_youtube_client, fetch_video_details


def main() -> None:
    load_dotenv()
    api_keys = os.getenv("YT_API_KEYS") or os.getenv("YT_API_KEY", "")
    api_key_list = [key.strip() for key in api_keys.split(",") if key.strip()]
    if not api_key_list:
        raise SystemExit("No YT_API_KEY or YT_API_KEYS found in environment.")

    yt = build_youtube_client(api_key_list[0])
    conn = db.get_connection()
    tracked = db.fetch_tracked_videos(conn)
    if not tracked:
        print("No tracked videos.")
        return

    video_ids = [row["video_id"] for row in tracked]
    try:
        details = fetch_video_details(yt, video_ids)
    except HttpError as exc:
        print(f"YouTube API error: {exc}")
        return

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
    print(f"Saved snapshots for {len(details)} videos at {now_ts}.")


if __name__ == "__main__":
    main()
