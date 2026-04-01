#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动发现并下载 DeepSeek 当前使用的 sha3_wasm_bg.*.wasm 到本项目目录。
默认输出目录为当前工作目录（建议在项目根目录运行）。

用法：
  python fetch_wasm.py
  python fetch_wasm.py --output-dir .
  python fetch_wasm.py --cleanup-old
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, List


BASE_URL = "https://chat.deepseek.com"
HOME_URL = f"{BASE_URL}/"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

JS_URL_PATTERN = re.compile(
    r"(https://chat\.deepseek\.com/_next/static/chunks/[^\"']+?\.js|/_next/static/chunks/[^\"']+?\.js)"
)

WASM_FULL_PATH_PATTERN = re.compile(
    r"(https://chat\.deepseek\.com/_next/static/wasm/sha3_wasm_bg\.[A-Za-z0-9_-]+\.wasm|/_next/static/wasm/sha3_wasm_bg\.[A-Za-z0-9_-]+\.wasm)"
)

WASM_HASH_PATTERN = re.compile(r"sha3_wasm_bg\.([A-Za-z0-9_-]+)\.wasm")


def http_get_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Referer": HOME_URL,
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return raw.decode("utf-8", errors="ignore")


def http_get_bytes(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Referer": HOME_URL,
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def normalize_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("//"):
        return f"https:{url}"
    return urllib.parse.urljoin(BASE_URL, url)


def dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_js_urls(html_text: str) -> List[str]:
    matches = JS_URL_PATTERN.findall(html_text)
    return dedupe_keep_order(normalize_url(m) for m in matches)


def extract_wasm_urls_from_js(js_text: str) -> List[str]:
    # 兼容转义形式，如 \/ _next \/ static \/ wasm
    normalized = js_text.replace("\\/", "/")

    urls: List[str] = []

    # 方式1：直接提取完整 wasm 路径
    for m in WASM_FULL_PATH_PATTERN.findall(normalized):
        urls.append(normalize_url(m))

    # 方式2：只提取 hash，再拼路径
    for hash_part in WASM_HASH_PATTERN.findall(normalized):
        urls.append(f"{BASE_URL}/_next/static/wasm/sha3_wasm_bg.{hash_part}.wasm")

    return dedupe_keep_order(urls)


def discover_wasm_candidates(timeout: int = 20, max_chunks: int = 200) -> List[str]:
    html_text = http_get_text(HOME_URL, timeout=timeout)
    js_urls = extract_js_urls(html_text)

    if not js_urls:
        raise RuntimeError("未在首页中发现 Next.js chunk 链接，无法继续自动发现 wasm。")

    wasm_candidates: List[str] = []

    for idx, js_url in enumerate(js_urls[:max_chunks], start=1):
        try:
            js_text = http_get_text(js_url, timeout=timeout)
        except Exception:
            continue

        found = extract_wasm_urls_from_js(js_text)
        if found:
            wasm_candidates.extend(found)

    wasm_candidates = dedupe_keep_order(wasm_candidates)
    if not wasm_candidates:
        raise RuntimeError("未在 chunk 脚本中发现 sha3_wasm_bg.*.wasm 引用。")

    return wasm_candidates


def download_first_available(candidates: List[str], output_dir: Path, timeout: int = 30) -> Path:
    errors = []
    for url in candidates:
        try:
            data = http_get_bytes(url, timeout=timeout)
            filename = Path(urllib.parse.urlparse(url).path).name
            if not filename.endswith(".wasm"):
                continue
            output_path = output_dir / filename
            output_path.write_bytes(data)
            print(f"[OK] 下载成功: {url}")
            print(f"[OK] 文件保存: {output_path}")
            return output_path
        except Exception as e:
            errors.append(f"{url} -> {e}")

    joined = "\n".join(errors[:10])
    raise RuntimeError(f"候选 wasm 均下载失败，示例错误：\n{joined}")


def cleanup_old_wasm(output_dir: Path, keep_file: Path) -> None:
    for p in output_dir.glob("sha3_wasm_bg.*.wasm"):
        if p.resolve() != keep_file.resolve():
            try:
                p.unlink()
                print(f"[CLEAN] 删除旧文件: {p}")
            except Exception as e:
                print(f"[WARN] 删除旧文件失败: {p}, 错误: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="自动发现并下载 DeepSeek 当前 wasm 到项目目录。"
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="输出目录（默认当前目录）",
    )
    parser.add_argument(
        "--cleanup-old",
        action="store_true",
        help="下载后清理输出目录中其它 sha3_wasm_bg.*.wasm 旧文件",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=20,
        help="HTTP 超时时间（秒），默认 20",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP] 正在发现 wasm 候选链接...")
    candidates = discover_wasm_candidates(timeout=args.timeout)
    print(f"[INFO] 发现候选数量: {len(candidates)}")
    for i, c in enumerate(candidates[:5], start=1):
        print(f"  {i}. {c}")
    if len(candidates) > 5:
        print("  ...")

    print("[STEP] 正在下载可用 wasm...")
    saved = download_first_available(candidates, output_dir, timeout=max(args.timeout, 30))

    if args.cleanup_old:
        print("[STEP] 正在清理旧 wasm...")
        cleanup_old_wasm(output_dir, saved)

    print("[DONE] 完成")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 用户中断")
        raise SystemExit(130)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
