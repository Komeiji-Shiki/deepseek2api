"""DeepSeek 文件上传 & Vision 图片预处理（纯函数，无 app 依赖）。"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import mimetypes
import uuid
from typing import Any, Callable

logger = logging.getLogger("deepseek2api")

DEEPSEEK_HOST = "chat.deepseek.com"
DEEPSEEK_UPLOAD_FILE_URL = f"https://{DEEPSEEK_HOST}/api/v0/file/upload_file"
DEEPSEEK_UPLOAD_TARGET_PATH = "/api/v0/file/upload_file"
MAX_INLINE_FILES = 50

# ── multipart 构建 ────────────────────────────────────────────


def _generate_boundary() -> str:
    return "----DeepSeekUpload" + uuid.uuid4().hex[:16]


def build_multipart_body(filename: str, content_type: str, data: bytes) -> tuple[bytes, str]:
    """构建 multipart/form-data 请求体，返回 (body, content_type_header)。"""
    boundary = _generate_boundary()
    buf = io.BytesIO()
    buf.write(f"--{boundary}\r\n".encode("utf-8"))
    buf.write(
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        .encode("utf-8")
    )
    buf.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    buf.write(data)
    buf.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return buf.getvalue(), f"multipart/form-data; boundary={boundary}"


# ── 响应解析 ──────────────────────────────────────────────────


def extract_file_id(resp_data: dict[str, Any]) -> str:
    """从 DeepSeek 上传响应中提取 file_id。"""
    search_maps: list[dict[str, Any]] = [resp_data]
    data_section = resp_data.get("data", {})
    if isinstance(data_section, dict):
        search_maps.append(data_section)
        biz_data = data_section.get("biz_data", {})
        if isinstance(biz_data, dict):
            search_maps.append(biz_data)
            for key in ("file", "biz_data", "data"):
                nested = biz_data.get(key)
                if isinstance(nested, dict):
                    search_maps.append(nested)

    for m in search_maps:
        if not isinstance(m, dict):
            continue
        for key in ("id", "file_id"):
            val = m.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


# ── data: URL 解码 ────────────────────────────────────────────


def is_data_url(s: str) -> bool:
    return s.strip().lower().startswith("data:")


def decode_data_url(url: str) -> tuple[bytes, str] | None:
    """解码 data: URL，返回 (data, content_type) 或 None。"""
    url = url.strip()
    if not is_data_url(url):
        return None
    header, payload = url.split(",", 1)
    meta = header[len("data:"):]
    is_base64 = ";base64" in meta.lower()
    # 提取 content_type
    parts = meta.split(";")
    content_type = parts[0] if parts and parts[0] else "application/octet-stream"
    if is_base64:
        try:
            return base64.b64decode(payload), content_type
        except Exception:
            return None
    else:
        from urllib.parse import unquote
        try:
            return unquote(payload).encode("utf-8"), content_type
        except Exception:
            return None


# ── 内容哈希（去重）──────────────────────────────────────────


def hash_content(content_type: str, data: bytes) -> str:
    h = hashlib.sha256()
    h.update(content_type.encode("utf-8"))
    h.update(b"\x00")
    h.update(data)
    return h.hexdigest()


# ── Vision 预处理 ─────────────────────────────────────────────

UploadFunc = Callable[[bytes, str, str], str | None]


def preprocess_inline_files(
    messages: list[dict[str, Any]],
    upload_fn: UploadFunc,
) -> tuple[list[dict[str, Any]], list[str]]:
    """扫描消息中的 inline image/file，调用 upload_fn 上传到 DeepSeek。

    upload_fn(data, filename, content_type) -> file_id | None

    返回 (new_messages, ref_file_ids)。
    """
    ref_file_ids: list[str] = []
    upload_cache: dict[str, str] = {}
    inline_count = 0

    def _walk(obj: Any) -> Any:
        nonlocal inline_count
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        if not isinstance(obj, dict):
            return obj

        # 已有 file_id 则收集
        existing_fid = obj.get("file_id")
        if isinstance(existing_fid, str) and existing_fid.strip():
            fid = existing_fid.strip()
            if fid not in ref_file_ids:
                ref_file_ids.append(fid)
            return obj

        # 处理 image_url (data: URL)
        image_url = obj.get("image_url")
        if image_url is not None:
            url_str = ""
            if isinstance(image_url, str):
                url_str = image_url
            elif isinstance(image_url, dict):
                url_str = str(image_url.get("url", ""))
            if is_data_url(url_str):
                decoded = decode_data_url(url_str)
                if decoded and inline_count < MAX_INLINE_FILES:
                    img_data, img_ct = decoded
                    cache_key = hash_content(img_ct, img_data)
                    if cache_key not in upload_cache:
                        ext = mimetypes.guess_extension(img_ct.split(";")[0]) or ".png"
                        fid = upload_fn(img_data, f"image{ext}", img_ct)
                        if fid:
                            upload_cache[cache_key] = fid
                        else:
                            return obj  # 上传失败，保留原始 block
                    fid = upload_cache.get(cache_key)
                    if fid and fid not in ref_file_ids:
                        ref_file_ids.append(fid)
                        inline_count += 1
                    return {"type": "input_image", "file_id": fid or ""}

        # 递归处理嵌套字段
        for key in ("messages", "input", "attachments", "content", "files", "items",
                     "data", "source", "file", "image_url"):
            if key in obj:
                obj[key] = _walk(obj[key])

        return obj

    new_messages = _walk(messages)
    return new_messages, ref_file_ids
