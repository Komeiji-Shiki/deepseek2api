"""DeepSeek 文件上传 & Vision 图片预处理（纯函数，无 app 依赖）。"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import mimetypes
import re
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
    payload_clean = re.sub(r'\s+', '', payload.strip())
    meta = header[len("data:"):]
    is_base64 = ";base64" in meta.lower()
    # 提取 content_type
    parts = meta.split(";")
    content_type = parts[0] if parts and parts[0] else "application/octet-stream"
    if is_base64:
        try:
            return base64.b64decode(payload_clean, validate=True), content_type
        except Exception:
            # fallback: 弹性 base64 解码
            decoded = decode_base64_flexible(payload_clean)
            if decoded:
                return decoded, content_type
            return None
    else:
        from urllib.parse import unquote
        try:
            return unquote(payload).encode("utf-8"), content_type
        except Exception:
            return None


# ── base64 弹性解码 ──────────────────────────────────────────

_BASE64_VARIANTS: list[tuple[str, Any]] = [
    ("std",          base64.b64decode),
    ("url",           base64.urlsafe_b64decode),
]


def decode_base64_flexible(raw: str) -> bytes | None:
    """尝试多种 base64 编码变体解码，返回 bytes 或 None。"""
    raw = raw.strip()
    if not raw:
        return None
    # 自动补齐 padding
    missing = len(raw) % 4
    if missing:
        raw += "=" * (4 - missing)
    for _name, decoder in _BASE64_VARIANTS:
        try:
            return decoder(raw)
        except Exception:
            continue
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


def _pick_filename(block: dict[str, Any], content_type: str, prefix: str) -> str:
    """从 block 中提取文件名，若无则根据 content_type 生成。"""
    for key in ("filename", "file_name", "name"):
        val = block.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().split("/")[-1].split("\\")[-1]
    ext = ".bin"
    if content_type:
        base_ct = content_type.split(";", 1)[0].strip()
        exts = mimetypes.guess_all_extensions(base_ct)
        if exts:
            ext = exts[0]
    return prefix + ext


def _extract_inline_file_block(block: dict[str, Any]) -> tuple[bytes, str, str, str] | None:
    """尝试从 block 中提取内联文件/图片数据。

    返回 (data, content_type, filename, replacement_type) 或 None。
    replacement_type: "input_image" 或 "input_file"
    """
    if not isinstance(block, dict):
        return None

    # 已有 file_id 则跳过
    existing_fid = block.get("file_id")
    if isinstance(existing_fid, str) and existing_fid.strip():
        return None

    # ── 1. image_url（OpenAI 格式）───────────────────────────
    image_url = block.get("image_url")
    if image_url is not None:
        url_str = ""
        extra_ct = ""
        if isinstance(image_url, str):
            url_str = image_url
        elif isinstance(image_url, dict):
            url_str = str(image_url.get("url", ""))
            for k in ("mime_type", "mimeType", "content_type", "contentType"):
                v = image_url.get(k)
                if isinstance(v, str) and v.strip():
                    extra_ct = v.strip()
                    break
        if is_data_url(url_str):
            decoded = decode_data_url(url_str)
            if decoded:
                img_data, img_ct = decoded
                final_ct = extra_ct or img_ct
                filename = _pick_filename(block, final_ct, "image")
                return img_data, final_ct, filename, "input_image"
        return None

    block_type = str(block.get("type", "")).strip().lower()

    # ── 2. inline data 字段（纯 base64 / data: URL）───────────
    for data_key in ("data", "base64", "file_data"):
        raw = block.get(data_key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        raw = raw.strip()
        # data: URL
        if is_data_url(raw):
            decoded = decode_data_url(raw)
            if decoded:
                img_data, img_ct = decoded
                prefix = "image" if "image" in block_type else "upload"
                filename = _pick_filename(block, img_ct, prefix)
                repl_type = "input_image" if "image" in block_type else "input_file"
                return img_data, img_ct, filename, repl_type
        # 纯 base64
        decoded = decode_base64_flexible(raw)
        if decoded:
            content_type = _guess_content_type_from_block(block, decoded)
            prefix = "image" if "image" in block_type else "upload"
            filename = _pick_filename(block, content_type, prefix)
            repl_type = "input_image" if "image" in block_type else "input_file"
            return decoded, content_type, filename, repl_type

    # ── 3. url 字段（直接 data: URL）──────────────────────────
    url_val = block.get("url")
    if isinstance(url_val, str) and is_data_url(url_val):
        decoded = decode_data_url(url_val)
        if decoded:
            img_data, img_ct = decoded
            filename = _pick_filename(block, img_ct, "upload")
            return img_data, img_ct, filename, "input_image"

    # ── 4. 嵌套 file 字段 ─────────────────────────────────────
    nested_file = block.get("file")
    if isinstance(nested_file, dict):
        return _extract_inline_file_block(nested_file)

    return None


def _guess_content_type_from_block(block: dict[str, Any], data: bytes) -> str:
    """从 block 或 data 中猜测 content_type。"""
    for key in ("mime_type", "mimeType", "content_type", "contentType", "media_type", "mediaType"):
        val = block.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    if data:
        # 通过魔数检测
        if data[:4] == b"\x89PNG":
            return "image/png"
        if data[:2] == b"\xff\xd8":
            return "image/jpeg"
        if data[:4] == b"GIF8":
            return "image/gif"
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
    return "application/octet-stream"


class InlineFileUploadError(RuntimeError):
    """内联文件上传失败异常。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def preprocess_inline_files(
    messages: list[dict[str, Any]],
    upload_fn: UploadFunc,
) -> tuple[list[dict[str, Any]], list[str]]:
    """扫描消息中的 inline image/file，调用 upload_fn 上传到 DeepSeek。

    upload_fn(data, filename, content_type) -> file_id | None

    返回 (new_messages, ref_file_ids)。
    上传失败时抛出 InlineFileUploadError，不再保留原始 base64 数据。
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

        # 尝试提取内联文件/图片
        extracted = _extract_inline_file_block(obj)
        if extracted is not None:
            img_data, img_ct, filename, repl_type = extracted
            if inline_count >= MAX_INLINE_FILES:
                raise InlineFileUploadError(
                    f"exceeded maximum of {MAX_INLINE_FILES} inline files per request"
                )
            cache_key = hash_content(img_ct, img_data)
            if cache_key not in upload_cache:
                fid = upload_fn(img_data, filename, img_ct)
                if fid:
                    upload_cache[cache_key] = fid
                else:
                    raise InlineFileUploadError(
                        f"Failed to upload inline file: {filename}"
                    )
            fid = upload_cache.get(cache_key)
            if fid and fid not in ref_file_ids:
                ref_file_ids.append(fid)
                inline_count += 1
            replacement: dict[str, Any] = {
                "type": repl_type,
                "file_id": fid or "",
            }
            if filename:
                replacement["filename"] = filename
            return replacement

        # 递归处理嵌套字段
        for key in ("messages", "input", "attachments", "content", "files", "items",
                     "data", "source", "file", "image_url"):
            if key in obj:
                obj[key] = _walk(obj[key])

        return obj

    new_messages = _walk(messages)
    return new_messages, ref_file_ids
