#!/usr/bin/env python3
"""CLI utility to recursively ingest documents from a local directory into Gemini Bridge RAG."""

import argparse
import sys
from pathlib import Path

import httpx

SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".rst", ".py", ".json", ".yaml", ".yml"}


def ingest_file(
    file_path: Path,
    base_url: str,
    client: httpx.Client,
) -> int:
    """Ingest a single file into the Gemini Bridge vector store."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"  [SKIPPED] Could not read {file_path.name}: {exc}")
        return 0

    if not content.strip():
        print(f"  [SKIPPED] {file_path.name} (empty)")
        return 0

    payload = {
        "content": content,
        "source": str(file_path),
        "metadata": {
            "filename": file_path.name,
            "suffix": file_path.suffix,
        },
    }

    url = f"{base_url.rstrip('/')}/api/v1/rag/ingest"
    response = client.post(url, json=payload, timeout=30.0)

    if response.status_code == 201:
        data = response.json()
        chunks = data.get("chunks_count", 0)
        print(f"  ✓ {file_path.name} -> {chunks} chunk(s)")
        return int(chunks)

    print(f"  ✗ Failed to ingest {file_path.name} ({response.status_code}): {response.text}")
    return 0


def main() -> None:
    """Run directory ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest local files and directories into Gemini Bridge knowledge base.",
    )
    parser.add_argument("path", type=str, help="Path to directory or file to ingest")
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000",
        help="Gemini Bridge base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="",
        help="Comma-separated file extensions to include (e.g. .md,.txt)",
    )

    args = parser.parse_args()
    target_path = Path(args.path)

    if not target_path.exists():
        print(f"Error: Path '{target_path}' does not exist.")
        sys.exit(1)

    allowed_exts = (
        {f".{ext.strip().lstrip('.')}" for ext in args.extensions.split(",") if ext.strip()}
        if args.extensions
        else SUPPORTED_EXTENSIONS
    )

    files_to_process: list[Path] = []
    if target_path.is_file():
        files_to_process.append(target_path)
    else:
        files_to_process.extend(
            p for p in sorted(target_path.rglob("*")) if p.is_file() and p.suffix.lower() in allowed_exts
        )

    if not files_to_process:
        print(f"No matching files found in '{target_path}'.")
        sys.exit(0)

    print(f"Found {len(files_to_process)} file(s) to ingest into {args.endpoint}...")

    total_chunks = 0
    successful_files = 0

    with httpx.Client() as client:
        for file_path in files_to_process:
            chunks = ingest_file(file_path, args.endpoint, client)
            if chunks > 0:
                successful_files += 1
                total_chunks += chunks

    print("\n--- Ingestion Summary ---")
    print(f"Files processed : {successful_files}/{len(files_to_process)}")
    print(f"Chunks indexed  : {total_chunks}")


if __name__ == "__main__":
    main()
