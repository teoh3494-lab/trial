from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


@dataclass
class VideoDetail:
    video_id: str
    title: str
    channel: str
    published_at: str
    duration_sec: int
    views: int
    likes: int
    comments: int
    tags: list[str]
    thumbnail_url: str


def build_youtube_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def parse_iso8601_duration(duration: str) -> int:
    # PT#H#M#S
    hours = 0
    minutes = 0
    seconds = 0
    if not duration.startswith("PT"):
        return 0
    time_str = duration[2:]
    current = ""
    for char in time_str:
        if char.isdigit():
            current += char
            continue
        if char == "H":
            hours = int(current) if current else 0
            current = ""
        elif char == "M":
            minutes = int(current) if current else 0
            current = ""
        elif char == "S":
            seconds = int(current) if current else 0
            current = ""
    return hours * 3600 + minutes * 60 + seconds


def chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def search_videos(
    yt,
    query: str,
    region_code: str,
    relevance_language: str,
    published_after: Optional[str],
    order: str,
    video_duration: str,
    max_results: int,
) -> list[str]:
    request = (
        yt.search()
        .list(
            part="snippet",
            type="video",
            q=query,
            regionCode=region_code,
            relevanceLanguage=relevance_language,
            order=order,
            publishedAfter=published_after,
            videoDuration=video_duration,
            maxResults=max_results,
        )
    )
    response = request.execute()
    items = response.get("items", [])
    return [item["id"]["videoId"] for item in items if "videoId" in item["id"]]


def fetch_video_details(yt, video_ids: Iterable[str]) -> list[VideoDetail]:
    details: list[VideoDetail] = []
    ids = list(video_ids)
    if not ids:
        return details
    for batch in chunk_list(ids, 50):
        request = (
            yt.videos()
            .list(
                part="snippet,contentDetails,statistics",
                id=",".join(batch),
                maxResults=50,
            )
        )
        response = request.execute()
        for item in response.get("items", []):
            stats = item.get("statistics", {})
            snippet = item.get("snippet", {})
            content = item.get("contentDetails", {})
            tags = snippet.get("tags") or []
            thumbnails = snippet.get("thumbnails", {})
            thumbnail_url = ""
            for key in ("high", "medium", "default"):
                if thumbnails.get(key, {}).get("url"):
                    thumbnail_url = thumbnails[key]["url"]
                    break
            details.append(
                VideoDetail(
                    video_id=item.get("id", ""),
                    title=snippet.get("title", ""),
                    channel=snippet.get("channelTitle", ""),
                    published_at=snippet.get("publishedAt", ""),
                    duration_sec=parse_iso8601_duration(content.get("duration", "PT0S")),
                    views=int(stats.get("viewCount", 0)),
                    likes=int(stats.get("likeCount", 0)) if stats.get("likeCount") else 0,
                    comments=int(stats.get("commentCount", 0))
                    if stats.get("commentCount")
                    else 0,
                    tags=tags,
                    thumbnail_url=thumbnail_url,
                )
            )
    return details


__all__ = [
    "HttpError",
    "VideoDetail",
    "build_youtube_client",
    "parse_iso8601_duration",
    "search_videos",
    "fetch_video_details",
]
