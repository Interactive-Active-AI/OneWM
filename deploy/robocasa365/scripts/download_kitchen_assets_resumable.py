#!/usr/bin/env python3
"""Resumable, validated downloader for RoboCasa kitchen assets."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import threading
import time
import uuid
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO_ROOT / "third_party/RoboCasa365"
DEFAULT_ASSETS_ROOT = SOURCE_ROOT / "robocasa/models/assets"
DEFAULT_CACHE_ROOT = Path("data/robocasa365/kitchen_assets")
BOX_LINKS_RELATIVE = Path("robocasa/models/assets/box_links/box_links_assets.json")
SOURCE_COMMIT = "b4684e6ee37d377cc392e98302a6b916d588b415"

# Sizes were verified with HTTP Range requests against the Box links pinned by
# SOURCE_COMMIT. Exact byte checks prevent an HTML error response from being
# mistaken for a valid archive.
ASSETS = (
    {
        "key": "tex",
        "box_key": "textures",
        "archive_folder": "textures",
        "install_base": ".",
        "expected_bytes": 543_180_421,
    },
    {
        "key": "fixtures_lw",
        "box_key": "fixtures_lightwheel",
        "archive_folder": "fixtures",
        "install_base": ".",
        "expected_bytes": 529_745_465,
    },
    {
        "key": "objs_lw",
        "box_key": "objects_lightwheel",
        "archive_folder": "lightwheel",
        "install_base": "objects",
        "expected_bytes": 791_939_656,
    },
    {
        "key": "tex_generative",
        "box_key": "generative_textures",
        "archive_folder": "generative_textures",
        "install_base": ".",
        "expected_bytes": 1_184_675_536,
    },
    {
        "key": "objs_objaverse",
        "box_key": "objaverse",
        "archive_folder": "objaverse",
        "install_base": "objects",
        "expected_bytes": 2_163_884_721,
    },
    {
        "key": "objs_aigen",
        "box_key": "aigen_objs",
        "archive_folder": "aigen_objs",
        "install_base": "objects",
        "expected_bytes": 5_773_223_039,
    },
)


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


class KitchenAssetDownloader:
    def __init__(self, source_root: Path, assets_root: Path, cache_root: Path):
        self.source_root = source_root
        self.assets_root = assets_root
        self.cache_root = cache_root
        self.control_dir = cache_root / "_download"
        self.archive_dir = cache_root / "archives"
        self.marker_dir = self.control_dir / "completed"
        self.manifest_path = self.control_dir / "kitchen_assets_manifest.jsonl"
        self.status_path = self.control_dir / "kitchen_assets_status.json"
        self.events_path = self.control_dir / "kitchen_assets_events.jsonl"
        self.lock_path = self.control_dir / "kitchen_assets_runner.lock"
        self.items: list[dict] = []
        self.statuses: dict[str, dict] = {}
        self.lock_stream = None
        self.state_lock = threading.RLock()

    def prepare(self) -> None:
        actual_commit = subprocess.check_output(
            ["git", "-C", str(self.source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_commit != SOURCE_COMMIT:
            raise RuntimeError(
                f"RoboCasa source must be pinned at {SOURCE_COMMIT}, got {actual_commit}"
            )

        links_path = self.source_root / BOX_LINKS_RELATIVE
        links = json.loads(links_path.read_text(encoding="utf-8"))
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.marker_dir.mkdir(parents=True, exist_ok=True)
        self.assets_root.mkdir(parents=True, exist_ok=True)

        for spec in ASSETS:
            shared_url = links[spec["box_key"]]
            token = shared_url.rstrip("/").split("/")[-1]
            item = {
                **spec,
                "shared_url": shared_url,
                "download_url": (f"https://utexas.box.com/shared/static/{token}.zip"),
                "destination": str(
                    self.assets_root / spec["install_base"] / spec["archive_folder"]
                ),
                "source_commit": SOURCE_COMMIT,
            }
            self.items.append(item)

        self.manifest_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.items),
            encoding="utf-8",
        )
        self.refresh_statuses()

    def acquire_lock(self) -> None:
        self.lock_stream = self.lock_path.open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"another kitchen asset downloader owns {self.lock_path}"
            ) from exc
        self.lock_stream.seek(0)
        self.lock_stream.truncate()
        self.lock_stream.write(f"pid={os.getpid()}\nstarted_at={now()}\n")
        self.lock_stream.flush()

    def append_event(self, event: str, **fields) -> None:
        record = {"time": now(), "event": event, **fields}
        with self.state_lock:
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def marker_path(self, key: str) -> Path:
        return self.marker_dir / f"{key}.json"

    def installed_from_marker(self, item: dict) -> dict | None:
        marker_path = self.marker_path(item["key"])
        destination = Path(item["destination"])
        if not marker_path.is_file() or not destination.is_dir():
            return None
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("source_commit") != SOURCE_COMMIT
            or marker.get("expected_bytes") != item["expected_bytes"]
            or marker.get("download_url") != item["download_url"]
        ):
            return None
        if not any(path.is_file() for path in destination.rglob("*")):
            return None
        return marker

    def refresh_statuses(self) -> None:
        for item in self.items:
            marker = self.installed_from_marker(item)
            if marker:
                self.statuses[item["key"]] = {
                    "status": "complete",
                    "destination": item["destination"],
                    "updated_at": now(),
                    "validation": marker.get("validation", {}),
                    "skipped_existing": True,
                }
            else:
                previous = self.statuses.get(item["key"], {})
                self.statuses[item["key"]] = {
                    **previous,
                    "status": "pending",
                    "destination": item["destination"],
                    "updated_at": now(),
                }
        self.flush_status()

    def flush_status(self) -> None:
        with self.state_lock:
            counts: dict[str, int] = {}
            for item in self.statuses.values():
                status = item["status"]
                counts[status] = counts.get(status, 0) + 1
            atomic_write_json(
                self.status_path,
                {
                    "updated_at": now(),
                    "source_commit": SOURCE_COMMIT,
                    "assets_root": str(self.assets_root),
                    "cache_root": str(self.cache_root),
                    "total": len(self.statuses),
                    "counts": counts,
                    "items": self.statuses,
                },
            )

    def set_status(self, key: str, status: str, **fields) -> None:
        with self.state_lock:
            self.statuses[key] = {
                **self.statuses.get(key, {}),
                "status": status,
                "updated_at": now(),
                **fields,
            }
            self.flush_status()

    @staticmethod
    def validate_zip_member(info: zipfile.ZipInfo) -> PurePosixPath:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"unsafe ZIP member path: {info.filename}")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise RuntimeError(f"ZIP symlink is not accepted: {info.filename}")
        return path

    def extract_and_install(self, item: dict, archive_path: Path) -> dict:
        staging = self.control_dir / (
            f".{item['key']}.extracting.{os.getpid()}.{uuid.uuid4().hex}"
        )
        staging.mkdir()
        file_infos: list[zipfile.ZipInfo] = []
        try:
            with zipfile.ZipFile(archive_path) as archive:
                if not archive.infolist():
                    raise RuntimeError("empty ZIP archive")
                for info in archive.infolist():
                    relative = self.validate_zip_member(info)
                    target = staging.joinpath(*relative.parts)
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    if target.stat().st_size != info.file_size:
                        raise RuntimeError(
                            f"extracted size mismatch for {info.filename}: "
                            f"expected={info.file_size}, actual={target.stat().st_size}"
                        )
                    file_infos.append(info)

            archive_folder = staging / item["archive_folder"]
            if not archive_folder.is_dir():
                raise RuntimeError(
                    f"ZIP does not contain expected top-level folder "
                    f"{item['archive_folder']!r}"
                )
            if not file_infos:
                raise RuntimeError("ZIP contains no regular files")

            install_base = (self.assets_root / item["install_base"]).resolve()
            install_base.mkdir(parents=True, exist_ok=True)
            for child in staging.iterdir():
                destination = install_base / child.name
                if child.is_dir():
                    shutil.copytree(child, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, destination)

            for info in file_infos:
                relative = self.validate_zip_member(info)
                installed = install_base.joinpath(*relative.parts)
                if (
                    not installed.is_file()
                    or installed.stat().st_size != info.file_size
                ):
                    raise RuntimeError(
                        f"installed asset validation failed: {installed}"
                    )

            return {
                "archive_bytes": archive_path.stat().st_size,
                "files": len(file_infos),
                "uncompressed_bytes": sum(info.file_size for info in file_infos),
            }
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def download_item(self, item: dict) -> None:
        key = item["key"]
        expected_bytes = int(item["expected_bytes"])
        partial = self.archive_dir / f"{key}.zip.partial"
        finished = self.archive_dir / f"{key}.zip"

        marker = self.installed_from_marker(item)
        if marker:
            self.set_status(
                key,
                "complete",
                destination=item["destination"],
                validation=marker.get("validation", {}),
                skipped_existing=True,
            )
            return

        if finished.exists() and finished.stat().st_size != expected_bytes:
            backup = finished.with_name(f"{finished.name}.invalid.{int(time.time())}")
            os.replace(finished, backup)
            self.append_event(
                "invalid_finished_archive_moved",
                key=key,
                backup=str(backup),
                expected_bytes=expected_bytes,
            )
        if partial.exists() and partial.stat().st_size > expected_bytes:
            backup = partial.with_name(f"{partial.name}.invalid.{int(time.time())}")
            os.replace(partial, backup)
            self.append_event("oversized_partial_moved", key=key, backup=str(backup))

        if not finished.exists():
            partial_bytes = partial.stat().st_size if partial.exists() else 0
            self.set_status(
                key,
                "downloading",
                partial=str(partial),
                partial_bytes=partial_bytes,
                expected_bytes=expected_bytes,
            )
            self.append_event(
                "download_started",
                key=key,
                partial_bytes=partial_bytes,
                expected_bytes=expected_bytes,
            )
            if partial_bytes != expected_bytes:
                command = [
                    "curl",
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
                ]
                subprocess.run(command, check=True)
            actual_bytes = partial.stat().st_size
            if actual_bytes != expected_bytes:
                raise RuntimeError(
                    f"downloaded size mismatch for {key}: "
                    f"expected={expected_bytes}, actual={actual_bytes}"
                )
            os.replace(partial, finished)

        self.set_status(
            key,
            "extracting",
            archive=str(finished),
            archive_bytes=finished.stat().st_size,
        )
        validation = self.extract_and_install(item, finished)
        marker_payload = {
            "completed_at": now(),
            "source_commit": SOURCE_COMMIT,
            "download_url": item["download_url"],
            "expected_bytes": expected_bytes,
            "destination": item["destination"],
            "validation": validation,
        }
        atomic_write_json(self.marker_path(key), marker_payload)
        finished.unlink()
        self.set_status(
            key,
            "complete",
            destination=item["destination"],
            validation=validation,
            skipped_existing=False,
        )
        self.append_event(
            "complete", key=key, destination=item["destination"], validation=validation
        )

    def run_pass(self, pass_number: int, workers: int) -> int:
        self.append_event(
            "pass_started", pass_number=pass_number, pid=os.getpid(), workers=workers
        )
        failures = 0
        pending = [
            item for item in self.items if self.installed_from_marker(item) is None
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.download_item, item): item for item in pending}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    failures += 1
                    self.set_status(item["key"], "failed", error=repr(exc))
                    self.append_event("failed", key=item["key"], error=repr(exc))
        complete = sum(
            self.installed_from_marker(item) is not None for item in self.items
        )
        self.append_event(
            "pass_finished",
            pass_number=pass_number,
            complete=complete,
            failed=failures,
            total=len(self.items),
        )
        return complete


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retry-delay-seconds", type=int, default=300)
    parser.add_argument("--single-pass", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.retry_delay_seconds < 1:
        raise SystemExit("--retry-delay-seconds must be positive")
    if not 1 <= args.workers <= len(ASSETS):
        raise SystemExit(f"--workers must be between 1 and {len(ASSETS)}")

    downloader = KitchenAssetDownloader(
        args.source_root.expanduser().resolve(),
        args.assets_root.expanduser().resolve(),
        args.cache_root.expanduser().resolve(),
    )
    downloader.prepare()
    if args.manifest_only:
        total_bytes = sum(int(item["expected_bytes"]) for item in downloader.items)
        print(
            f"validated kitchen asset manifest: {len(downloader.items)} items, "
            f"{total_bytes} bytes -> {downloader.manifest_path}"
        )
        return 0

    downloader.acquire_lock()
    available = shutil.disk_usage(downloader.cache_root).free
    required = sum(int(item["expected_bytes"]) for item in downloader.items)
    if available < required * 2:
        raise SystemExit(
            f"insufficient free space: available={available}, conservative_required={required * 2}"
        )

    downloader.append_event(
        "run_started",
        pid=os.getpid(),
        total=len(downloader.items),
        workers=args.workers,
        retry_until_complete=not args.single_pass,
    )
    pass_number = 0
    while True:
        pass_number += 1
        complete = downloader.run_pass(pass_number, args.workers)
        if complete == len(downloader.items):
            downloader.append_event("run_complete", passes=pass_number)
            print(f"all kitchen assets are complete: {downloader.assets_root}")
            return 0
        if args.single_pass:
            return 1
        downloader.append_event(
            "retry_wait",
            pass_number=pass_number,
            complete=complete,
            retry_delay_seconds=args.retry_delay_seconds,
        )
        time.sleep(args.retry_delay_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
