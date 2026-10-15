#!/usr/bin/env python3
"""Resumable downloader for the RoboCasa365 Human300 pretraining split."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import time
import uuid


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "third_party/RoboCasa365"
REGISTRY = SOURCE_ROOT / "robocasa/utils/dataset_registry.py"
BOX_LINKS = SOURCE_ROOT / "robocasa/models/assets/box_links/box_links_ds.json"
DEST_ROOT = Path("data/robocasa365")
CONTROL_DIR = DEST_ROOT / "_download"
MANIFEST_PATH = CONTROL_DIR / "human300_manifest.jsonl"
STATUS_PATH = CONTROL_DIR / "human300_status.json"
EVENTS_PATH = CONTROL_DIR / "human300_events.jsonl"
SOURCE_COMMIT = "b4684e6ee37d377cc392e98302a6b916d588b415"
VIEWS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)

lock = threading.Lock()
statuses: dict[str, dict] = {}
download_proxy: str | None = None

PROXY_ENV_VARS = (
    "ALL_PROXY",
    "FTP_PROXY",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "all_proxy",
    "ftp_proxy",
    "https_proxy",
    "http_proxy",
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def append_event(event: str, **fields) -> None:
    record = {"time": now(), "event": event, **fields}
    with lock:
        with EVENTS_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def flush_status() -> None:
    with lock:
        counts = {
            name: 0
            for name in ("pending", "downloading", "extracting", "complete", "failed")
        }
        for item in statuses.values():
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        payload = {
            "updated_at": now(),
            "source_commit": SOURCE_COMMIT,
            "total": len(statuses),
            "counts": counts,
            "items": statuses,
        }
        temp = STATUS_PATH.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp, STATUS_PATH)


def set_status(key: str, status: str, **fields) -> None:
    with lock:
        statuses[key] = {
            **statuses.get(key, {}),
            "status": status,
            "updated_at": now(),
            **fields,
        }
    flush_status()


def get_pretrain300_tasks() -> list[str]:
    tree = ast.parse(REGISTRY.read_text(encoding="utf-8"), filename=str(REGISTRY))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "PRETRAINING_TASKS"
            for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Call):
            break
        for keyword in node.value.keywords:
            if keyword.arg == "pretrain300":
                tasks = ast.literal_eval(keyword.value)
                if len(tasks) != 300 or len(set(tasks)) != 300:
                    raise RuntimeError(
                        f"Expected 300 unique tasks, got {len(tasks)}/{len(set(tasks))}"
                    )
                return tasks
    raise RuntimeError(
        "PRETRAINING_TASKS['pretrain300'] not found by static AST parsing"
    )


def build_manifest() -> list[dict]:
    tasks = set(get_pretrain300_tasks())
    links = json.loads(BOX_LINKS.read_text(encoding="utf-8"))
    items = []
    for key, shared_url in links.items():
        parts = key.split("/")
        if (
            len(parts) == 5
            and parts[0] == "pretrain"
            and parts[1] in {"atomic", "composite"}
            and parts[2] in tasks
            and parts[4] == "lerobot.tar"
        ):
            token = shared_url.rstrip("/").split("/")[-1]
            items.append(
                {
                    "key": key,
                    "category": parts[1],
                    "task": parts[2],
                    "date": parts[3],
                    "shared_url": shared_url,
                    # The tenant host redirects through utexas.app.box.com, which is not
                    # reachable from the managed Huawei Cloud proxy. The canonical Box
                    # host resolves the same public token directly to Box's CDN.
                    "download_url": f"https://www.box.com/public/static/{token}.tar",
                    "destination": str(DEST_ROOT.joinpath(*parts[:4], "lerobot")),
                    "source_commit": SOURCE_COMMIT,
                }
            )
    items.sort(key=lambda x: x["key"])
    selected_tasks = {item["task"] for item in items}
    if len(items) != 300 or selected_tasks != tasks:
        missing = sorted(tasks - selected_tasks)
        raise RuntimeError(
            f"Expected exact Human300 manifest; items={len(items)}, missing={missing}"
        )
    MANIFEST_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
        encoding="utf-8",
    )
    return items


def validate_dataset(base: Path) -> dict:
    required = (
        base / "meta/info.json",
        base / "meta/episodes.jsonl",
        base / "meta/tasks.jsonl",
        base / "meta/modality.json",
        base / "data",
        base / "videos",
    )
    missing = [str(path.relative_to(base)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing required paths: {missing}")
    info = json.loads((base / "meta/info.json").read_text(encoding="utf-8"))
    features = info.get("features", {})
    undeclared = [view for view in VIEWS if view not in features]
    if undeclared:
        raise RuntimeError(f"missing declared camera features: {undeclared}")
    parquet_count = sum(1 for _ in (base / "data").rglob("*.parquet"))
    if parquet_count == 0:
        raise RuntimeError("no parquet data files")
    video_counts = {}
    for view in VIEWS:
        count = sum(1 for _ in (base / "videos").rglob(f"{view}/*.mp4"))
        if count == 0:
            raise RuntimeError(f"no videos for {view}")
        video_counts[view] = count
    episode_lines = sum(
        1
        for line in (base / "meta/episodes.jsonl").open(encoding="utf-8")
        if line.strip()
    )
    task_lines = sum(
        1 for line in (base / "meta/tasks.jsonl").open(encoding="utf-8") if line.strip()
    )
    expected_episodes = info.get("total_episodes")
    if expected_episodes is not None and episode_lines != expected_episodes:
        raise RuntimeError(
            f"episode metadata mismatch: info={expected_episodes}, jsonl={episode_lines}"
        )
    if parquet_count != episode_lines:
        raise RuntimeError(
            f"parquet/episode mismatch: parquet={parquet_count}, episodes={episode_lines}"
        )
    for view, count in video_counts.items():
        if count != episode_lines:
            raise RuntimeError(
                f"video/episode mismatch for {view}: videos={count}, episodes={episode_lines}"
            )
    return {
        "episodes": episode_lines,
        "tasks": task_lines,
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "parquet": parquet_count,
        "videos_per_view": video_counts,
    }


def safe_extract(tar_path: Path, extract_root: Path) -> None:
    with tarfile.open(tar_path, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeError("empty tar archive")
        root = extract_root.resolve()
        for member in members:
            member_path = (extract_root / member.name).resolve()
            if member_path != root and root not in member_path.parents:
                raise RuntimeError(f"unsafe tar member: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(
                    f"links are not accepted in dataset tar: {member.name}"
                )
        archive.extractall(extract_root)


def archive_invalid(path: Path, label: str) -> Path:
    backup = path.with_name(f"{path.name}.{label}.{int(time.time())}")
    os.replace(path, backup)
    return backup


def download_item(item: dict) -> None:
    key = item["key"]
    dest = Path(item["destination"])
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            details = validate_dataset(dest)
            set_status(
                key,
                "complete",
                destination=str(dest),
                skipped_existing=True,
                validation=details,
            )
            return
        except Exception as exc:
            backup = archive_invalid(dest, "invalid")
            append_event(
                "existing_dataset_archived", key=key, error=str(exc), backup=str(backup)
            )

    partial = parent / "lerobot.tar.partial"
    finished_tar = parent / "lerobot.tar"
    set_status(key, "downloading", destination=str(dest), partial=str(partial))
    append_event(
        "download_started",
        key=key,
        partial_bytes=partial.stat().st_size if partial.exists() else 0,
    )
    curl = ["curl"]
    if download_proxy:
        curl.extend(("--proxy", download_proxy))
    else:
        curl.extend(("--noproxy", "*"))
    curl.extend(
        (
            "--location",
            "--fail",
            "--silent",
            "--show-error",
            "--retry",
            "20",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "--speed-time",
            "180",
            "--speed-limit",
            "1024",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            item["download_url"],
        )
    )
    curl_env = os.environ.copy()
    for name in PROXY_ENV_VARS:
        curl_env.pop(name, None)
    subprocess.run(curl, check=True, env=curl_env)
    os.replace(partial, finished_tar)
    set_status(key, "extracting", tar_bytes=finished_tar.stat().st_size)

    extract_root = parent / f".lerobot.extracting.{os.getpid()}.{uuid.uuid4().hex}"
    extract_root.mkdir()
    try:
        safe_extract(finished_tar, extract_root)
        extracted = extract_root / "lerobot"
        details = validate_dataset(extracted)
        os.replace(extracted, dest)
        extracted_readme = extract_root / "README.md"
        if extracted_readme.exists() and not (parent / "README.md").exists():
            os.replace(extracted_readme, parent / "README.md")
        # The archive is deleted only after extraction and full structural validation succeed.
        finished_tar.unlink()
        set_status(
            key,
            "complete",
            destination=str(dest),
            skipped_existing=False,
            validation=details,
        )
        append_event("complete", key=key, destination=str(dest), validation=details)
    except Exception:
        if finished_tar.exists():
            backup = archive_invalid(finished_tar, "corrupt")
            append_event("tar_archived_after_failure", key=key, backup=str(backup))
        raise
    finally:
        if extract_root.exists():
            shutil.rmtree(extract_root)


def main() -> int:
    global SOURCE_ROOT, REGISTRY, BOX_LINKS, DEST_ROOT, CONTROL_DIR
    global MANIFEST_PATH, STATUS_PATH, EVENTS_PATH, download_proxy

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--destination-root", type=Path, default=DEST_ROOT)
    parser.add_argument(
        "--proxy",
        help="Explicit proxy for dataset curl requests; inherited proxy variables are always ignored",
    )
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    SOURCE_ROOT = args.source_root.expanduser().resolve()
    DEST_ROOT = args.destination_root.expanduser().resolve()
    REGISTRY = SOURCE_ROOT / "robocasa/utils/dataset_registry.py"
    BOX_LINKS = SOURCE_ROOT / "robocasa/models/assets/box_links/box_links_ds.json"
    CONTROL_DIR = DEST_ROOT / "_download"
    MANIFEST_PATH = CONTROL_DIR / "human300_manifest.jsonl"
    STATUS_PATH = CONTROL_DIR / "human300_status.json"
    EVENTS_PATH = CONTROL_DIR / "human300_events.jsonl"
    download_proxy = args.proxy
    actual_commit = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != SOURCE_COMMIT:
        raise SystemExit(
            f"RoboCasa source must be pinned at {SOURCE_COMMIT}, got {actual_commit}"
        )
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    items = build_manifest()
    for item in items:
        statuses[item["key"]] = {
            "status": "pending",
            "destination": item["destination"],
            "updated_at": now(),
        }
    flush_status()
    if args.manifest_only:
        print(
            f"validated exact Human300 manifest: {len(items)} items -> {MANIFEST_PATH}"
        )
        return 0
    append_event("run_started", pid=os.getpid(), workers=args.workers, total=len(items))
    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_item, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                future.result()
            except Exception as exc:
                failed += 1
                set_status(item["key"], "failed", error=repr(exc))
                append_event("failed", key=item["key"], error=repr(exc))
    append_event("run_finished", failed=failed, total=len(items))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
