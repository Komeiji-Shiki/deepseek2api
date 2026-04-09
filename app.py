from __future__ import annotations

import asyncio
import base64
import ctypes
import json
import logging
import random
import re
import struct
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curl_cffi import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from wasmtime import Linker, Module, Store

from account_pool import AccountLease, AccountPool
from tool_calling import (
    XmlToolCallStreamParser,
    build_tool_system_prompt,
    extract_tool_calls_from_text,
    normalize_openai_messages,
    normalize_text_content,
    normalize_tool_definitions,
    prepend_system_instruction,
    render_assistant_think_block,
    render_tool_calls_as_xml_blocks,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepseek2api")

CONFIG_PATH = Path("config.json")
WASM_GLOB_PATTERN = "sha3_wasm_bg.*.wasm"
KEEP_ALIVE_TIMEOUT = 5
CLAUDE_DEFAULT_MODEL = "claude-sonnet-4-20250514"

DEEPSEEK_HOST = "chat.deepseek.com"
DEEPSEEK_LOGIN_URL = f"https://{DEEPSEEK_HOST}/api/v0/users/login"
DEEPSEEK_CREATE_SESSION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create"
DEEPSEEK_DELETE_SESSION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat_session/delete"
DEEPSEEK_CREATE_POW_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/create_pow_challenge"
DEEPSEEK_COMPLETION_URL = f"https://{DEEPSEEK_HOST}/api/v0/chat/completion"

BASE_HEADERS = {
    "Host": DEEPSEEK_HOST,
    "User-Agent": "DeepSeek/1.8.0 Android/35",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "Content-Type": "application/json",
    "x-client-platform": "android",
    "x-client-version": "1.8.0",
    "x-client-locale": "zh_CN",
    "accept-charset": "UTF-8",
}

OPENAI_MODELS = [
    {
        "id": "deepseek-default-chat",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepseek",
        "permission": [],
    },
    {
        "id": "deepseek-default-chat-search",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepseek",
        "permission": [],
    },
    {
        "id": "deepseek-default-reasoner",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepseek",
        "permission": [],
    },
    {
        "id": "deepseek-default-reasoner-search",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepseek",
        "permission": [],
    },
    {
        "id": "deepseek-expert-chat",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepseek",
        "permission": [],
    },
    {
        "id": "deepseek-expert-reasoner",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepseek",
        "permission": [],
    },
    {
        "id": "deepseek-expert-chat-search",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepseek",
        "permission": [],
    },
    {
        "id": "deepseek-expert-reasoner-search",
        "object": "model",
        "created": 1677610602,
        "owned_by": "deepseek",
        "permission": [],
    },
]

CLAUDE_MODELS = [
    {
        "id": "claude-sonnet-4-20250514",
        "object": "model",
        "created": 1715635200,
        "owned_by": "anthropic",
    },
    {
        "id": "claude-sonnet-4-20250514-fast",
        "object": "model",
        "created": 1715635200,
        "owned_by": "anthropic",
    },
    {
        "id": "claude-sonnet-4-20250514-slow",
        "object": "model",
        "created": 1715635200,
        "owned_by": "anthropic",
    },
]


def load_config() -> dict[str, Any]:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as exc:
        logger.warning("[load_config] 无法读取配置文件: %s", exc)
        return {}


CONFIG: dict[str, Any] = load_config()
CONFIG_WRITE_LOCK = asyncio.Lock()
ACCOUNT_POOL = AccountPool(CONFIG.get("accounts", []))
templates = Jinja2Templates(directory="templates")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http_client = requests.AsyncSession()
    try:
        yield
    finally:
        await app.state.http_client.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
)


@dataclass(slots=True)
class AuthContext:
    use_config_token: bool
    deepseek_token: str
    account_lease: AccountLease | None = None
    tried_accounts: set[str] = field(default_factory=set)

    @property
    def account(self) -> dict[str, Any] | None:
        return self.account_lease.account if self.account_lease else None


@dataclass(slots=True)
class CompletionContext:
    response: Any | None
    session_id: str
    prompt: str
    output_model: str
    thinking_enabled: bool
    search_enabled: bool
    expert_enabled: bool


class SyncResponseAsyncAdapter:
    """Wraps a sync curl_cffi streaming Response to provide async line iteration.

    curl_cffi's AsyncSession has known issues with long-running SSE streams
    (premature connection drops after ~20-30s). The sync Session does not have
    this problem. This adapter lets us use a sync streaming response while
    keeping the rest of the codebase async.
    """

    def __init__(self, sync_response):
        self._response = sync_response
        self._closed = False

    async def aiter_lines(self, decode_unicode=False, delimiter=None):
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def _reader():
            try:
                for raw_line in self._response.iter_lines():
                    loop.call_soon_threadsafe(q.put_nowait, raw_line)
            except Exception as exc:
                loop.call_soon_threadsafe(q.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            yield item

    async def aclose(self):
        if not self._closed:
            self._closed = True
            try:
                self._response.close()
            except Exception:
                pass


def get_http_client(request: Request):
    return request.app.state.http_client


async def save_config() -> None:
    try:
        async with CONFIG_WRITE_LOCK:
            with CONFIG_PATH.open("w", encoding="utf-8") as file:
                json.dump(CONFIG, file, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("[save_config] 写入配置失败: %s", exc)


def resolve_wasm_path(
    base_dir: str = ".",
    pattern: str = WASM_GLOB_PATTERN,
) -> str:
    candidates = sorted(
        Path(base_dir).glob(pattern),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"未找到 WASM 文件，期望匹配: {Path(base_dir) / pattern}")
    return str(candidates[0])


def extract_bearer_token(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: missing Bearer token.",
        )
    return auth_header.replace("Bearer ", "", 1).strip()


def ensure_authorized(request: Request) -> str:
    bearer_token = extract_bearer_token(request)
    if bearer_token:
        return bearer_token
    raise HTTPException(status_code=401, detail="Unauthorized.")


def get_auth_headers(auth_ctx: AuthContext) -> dict[str, str]:
    return {**BASE_HEADERS, "authorization": f"Bearer {auth_ctx.deepseek_token}"}


def estimate_tokens(value: Any) -> int:
    text = normalize_text_content(value)
    return max(1, len(text) // 4) if text else 0


def build_usage(prompt: str, reasoning: str, content: str) -> dict[str, Any]:
    prompt_tokens = estimate_tokens(prompt)
    reasoning_tokens = estimate_tokens(reasoning)
    completion_text_tokens = estimate_tokens(content)
    completion_tokens = reasoning_tokens + completion_text_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "completion_tokens_details": {
            "reasoning_tokens": reasoning_tokens,
        },
    }


def resolve_deepseek_model_features(model: str) -> tuple[bool, bool, bool]:
    normalized = model.strip().lower()
    # default 模式: chat (无推理, 无搜索)
    if normalized in {"deepseek-default-chat", "deepseek-chat", "deepseek-v3"}:
        return False, False, False
    # default 模式: chat + 搜索
    if normalized in {"deepseek-default-chat-search", "deepseek-chat-search", "deepseek-v3-search"}:
        return False, True, False
    # default 模式: reasoner (推理)
    if normalized in {"deepseek-default-reasoner", "deepseek-reasoner", "deepseek-r1"}:
        return True, False, False
    # default 模式: reasoner + 搜索
    if normalized in {"deepseek-default-reasoner-search", "deepseek-reasoner-search", "deepseek-r1-search"}:
        return True, True, False
    # expert 模式: chat
    if normalized in {"deepseek-expert-chat", "deepseek-expert"}:
        return False, False, True
    # expert 模式: reasoner
    if normalized in {"deepseek-expert-reasoner"}:
        return True, False, True
    # expert 模式: chat + 搜索
    if normalized in {"deepseek-expert-chat-search", "deepseek-expert-search"}:
        return False, True, True
    # expert 模式: reasoner + 搜索
    if normalized in {"deepseek-expert-reasoner-search"}:
        return True, True, True
    raise HTTPException(status_code=503, detail=f"Model '{model}' is not available.")


def map_claude_model_to_deepseek(model: str) -> str:
    mapping = CONFIG.get(
        "claude_model_mapping",
        {
            "fast": "deepseek-chat",
            "slow": "deepseek-chat",
        },
    )
    normalized = model.lower()
    if any(flag in normalized for flag in ("opus", "reasoner", "slow")):
        return mapping.get("slow", "deepseek-chat")
    return mapping.get("fast", "deepseek-chat")


def messages_prepare(messages: list[dict[str, Any]], *, thinking_enabled: bool = False) -> str:
    processed: list[dict[str, str]] = []
    first_system_seen = False
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(normalize_text_content(item))
            text = "\n".join(part for part in parts if part)
        else:
            text = normalize_text_content(content)

        if role == "system":
            if not first_system_seen:
                first_system_seen = True
                text = f"<system_instructions>{text}</system_instructions>"
            role = "user"

        processed.append({"role": role, "text": text})

    if not processed:
        return ""

    merged = [processed[0]]
    for message in processed[1:]:
        if message["role"] == merged[-1]["role"]:
            merged[-1]["text"] += "\n\n" + message["text"]
        else:
            merged.append(message)

    parts: list[str] = []
    for index, block in enumerate(merged):
        role = block["role"]
        text = block["text"]
        if role == "assistant":
            if thinking_enabled:
                parts.append(f"<｜Assistant｜><｜end▁of▁thinking｜>{text}<｜end▁of▁sentence｜>")
            else:
                parts.append(f"<｜Assistant｜>{text}<｜end▁of▁sentence｜>")
        elif role == "user":
            if index > 0:
                parts.append(f"<｜User｜>{text}")
            else:
                parts.append(text)
        else:
            parts.append(text)

    final_prompt = "".join(parts)
    final_prompt = re.sub(r"!\[(.*?)\]\((.*?)\)", r"[\1](\2)", final_prompt)
    return final_prompt


def is_finished_status_chunk(chunk: dict[str, Any]) -> bool:
    """判断是否为整体响应完成的状态事件。

    只有 response/status 或 response/quasi_status 为 FINISHED 时才算完成。
    fragment 级别的 status（如 response/fragments/-2/status）不算整体完成。
    """
    if not isinstance(chunk, dict):
        return False

    if str(chunk.get("status", "")).strip().upper() == "FINISHED":
        return True

    path = str(chunk.get("p", "")).strip().lower()
    value = chunk.get("v")

    # 只有 response/status 才算整体完成，fragment 级别的 status 不算
    if path in ("response/status", "response/quasi_status") and isinstance(value, str):
        return value.strip().upper() == "FINISHED"

    # BATCH 操作中检查是否有 response/status 或 quasi_status 为 FINISHED
    if isinstance(value, list) and path == "response" and chunk.get("o") == "BATCH":
        for item in value:
            if not isinstance(item, dict):
                continue
            item_path = str(item.get("p", "")).strip().lower()
            item_value = item.get("v")
            if item_path in ("status", "quasi_status") and isinstance(item_value, str):
                if item_value.strip().upper() == "FINISHED":
                    return True
        return False

    return False


def _is_claude_tool_result_message(message: dict[str, Any]) -> bool:
    if str(message.get("role", "")).strip() != "user":
        return False

    content = message.get("content", "")
    if not isinstance(content, list) or not content:
        return False

    has_tool_result = False
    for block in content:
        if not isinstance(block, dict):
            return False
        if str(block.get("type", "")).strip() != "tool_result":
            return False
        has_tool_result = True

    return has_tool_result


def _has_claude_tool_use(message: dict[str, Any]) -> bool:
    if str(message.get("role", "")).strip() != "assistant":
        return False

    content = message.get("content", "")
    if not isinstance(content, list):
        return False

    return any(
        isinstance(block, dict) and str(block.get("type", "")).strip() == "tool_use"
        for block in content
    )


def _collect_claude_active_tool_assistant_indexes(
    messages: list[dict[str, Any]],
) -> set[int]:
    active_indexes: set[int] = set()
    in_active_chain = False

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            break

        is_tool_result_message = _is_claude_tool_result_message(message)
        has_tool_use = _has_claude_tool_use(message)

        if not in_active_chain:
            if is_tool_result_message:
                in_active_chain = True
            elif has_tool_use:
                in_active_chain = True
                active_indexes.add(index)
                continue
            else:
                break

        if is_tool_result_message:
            continue
        if has_tool_use:
            active_indexes.add(index)
            continue
        break

    return active_indexes


def normalize_claude_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_tool_assistant_indexes = _collect_claude_active_tool_assistant_indexes(messages)
    normalized: list[dict[str, Any]] = []

    for index, message in enumerate(messages):
        role = str(message.get("role", "")).strip()
        content = message.get("content", "")
        if not isinstance(content, list):
            normalized.append({"role": role, "content": normalize_text_content(content)})
            continue

        parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        should_include_thinking = index in active_tool_assistant_indexes

        for block in content:
            if not isinstance(block, dict):
                parts.append(normalize_text_content(block))
                continue

            block_type = str(block.get("type", "")).strip()
            if block_type == "text":
                parts.append(str(block.get("text", "")))
            elif block_type == "thinking":
                if should_include_thinking:
                    thinking_block = render_assistant_think_block(block.get("thinking", ""))
                    if thinking_block:
                        parts.append(thinking_block)
            elif block_type == "tool_result":
                tool_use_id = str(block.get("tool_use_id", "unknown")).strip() or "unknown"
                tool_result = normalize_text_content(block.get("content", ""))
                parts.append(
                    f"Tool result (tool_use_id={tool_use_id}):\n{tool_result}"
                )
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "name": block.get("name", ""),
                        "arguments": block.get("input", {}) if isinstance(block.get("input", {}), dict) else {},
                    }
                )
            else:
                parts.append(normalize_text_content(block))

        if tool_calls:
            parts.append(render_tool_calls_as_xml_blocks(tool_calls))

        normalized.append(
            {
                "role": role,
                "content": "\n\n".join(part for part in parts if part),
            }
        )

    return normalized


async def login_deepseek_via_account(request: Request, account: dict[str, Any]) -> str:
    email = str(account.get("email", "")).strip()
    mobile = str(account.get("mobile", "")).strip()
    password = str(account.get("password", "")).strip()

    if not password or (not email and not mobile):
        raise HTTPException(
            status_code=400,
            detail="账号缺少必要的登录信息（必须提供 email 或 mobile 以及 password）",
        )

    if email:
        payload = {
            "email": email,
            "password": password,
            "device_id": "deepseek_to_api",
            "os": "android",
        }
    else:
        payload = {
            "mobile": mobile,
            "area_code": None,
            "password": password,
            "device_id": "deepseek_to_api",
            "os": "android",
        }

    response = None
    try:
        response = await get_http_client(request).post(
            DEEPSEEK_LOGIN_URL,
            headers=BASE_HEADERS,
            json=payload,
            timeout=30,
            impersonate="safari15_3",
        )
    except Exception as exc:
        logger.error("[login_deepseek_via_account] 登录请求异常: %s", exc)
        raise HTTPException(status_code=500, detail="Account login failed: request error") from exc

    try:
        data = response.json()
    except Exception as exc:
        logger.error("[login_deepseek_via_account] JSON 解析失败: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Account login failed: invalid JSON response",
        ) from exc
    finally:
        await response.aclose()

    logger.info(
        "[login_deepseek_via_account] 登录响应, status=%s, code=%s, msg=%s, body=%s",
        response.status_code,
        data.get("code"),
        data.get("msg"),
        str(data)[:500],
    )

    if response.status_code != 200:
        logger.error(
            "[login_deepseek_via_account] 登录失败, status=%s, body=%s",
            response.status_code,
            str(data)[:300],
        )
        raise HTTPException(status_code=500, detail="Account login failed.")

    biz_data = (
        data.get("data", {})
        .get("biz_data", {})
    )
    user_data = biz_data.get("user", {})
    new_token = str(user_data.get("token", "")).strip()
    if not new_token:
        logger.error("[login_deepseek_via_account] 登录响应缺少 token: %s", data)
        raise HTTPException(status_code=500, detail="Account login failed: missing token")

    account["token"] = new_token
    await save_config()
    return new_token


async def assign_account_lease(request: Request, auth_ctx: AuthContext) -> bool:
    while True:
        lease = await ACCOUNT_POOL.acquire(auth_ctx.tried_accounts)
        if lease is None:
            return False

        auth_ctx.account_lease = lease
        account = lease.account
        token = str(account.get("token", "")).strip()

        try:
            if not token:
                token = await login_deepseek_via_account(request, account)
            auth_ctx.deepseek_token = token
            return True
        except Exception as exc:
            logger.error(
                "[assign_account_lease] 账号 %s 不可用: %s",
                lease.identifier,
                exc,
            )
            auth_ctx.tried_accounts.add(lease.identifier)
            await lease.release()
            auth_ctx.account_lease = None


async def switch_account(request: Request, auth_ctx: AuthContext) -> bool:
    if not auth_ctx.use_config_token:
        return False

    if auth_ctx.account_lease is not None:
        auth_ctx.tried_accounts.add(auth_ctx.account_lease.identifier)
        await auth_ctx.account_lease.release()
        auth_ctx.account_lease = None

    return await assign_account_lease(request, auth_ctx)


async def determine_mode_and_token(request: Request) -> AuthContext:
    bearer_token = extract_bearer_token(request)
    config_keys = CONFIG.get("keys", [])

    if bearer_token in config_keys:
        logger.info(
            "[determine_mode_and_token] 使用账号池模式, bearer_token=%s..., 配置keys数量=%d",
            bearer_token[:8],
            len(config_keys),
        )
        if not ACCOUNT_POOL.has_accounts():
            raise HTTPException(
                status_code=503,
                detail="No accounts configured in config.json.",
            )

        auth_ctx = AuthContext(use_config_token=True, deepseek_token="")
        if not await assign_account_lease(request, auth_ctx):
            # 区分是"所有账号忙"还是"所有账号都不可用"
            total_accounts = ACCOUNT_POOL.size()
            raise HTTPException(
                status_code=503,
                detail=f"All {total_accounts} account(s) are currently busy or unavailable.",
            )
        logger.info(
            "[determine_mode_and_token] 分配到账号, token=%s...",
            auth_ctx.deepseek_token[:8],
        )
        return auth_ctx

    logger.info(
        "[determine_mode_and_token] 使用直连token模式, bearer_token=%s...",
        bearer_token[:8],
    )
    return AuthContext(use_config_token=False, deepseek_token=bearer_token)


async def release_auth_context(auth_ctx: AuthContext | None) -> None:
    if auth_ctx is None or auth_ctx.account_lease is None:
        return
    await auth_ctx.account_lease.release()
    auth_ctx.account_lease = None


async def create_session(
    request: Request,
    auth_ctx: AuthContext,
    max_attempts: int = 3,
) -> str | None:
    attempts = 0
    while attempts < max_attempts:
        response = None
        data: dict[str, Any] = {}
        try:
            response = await get_http_client(request).post(
                DEEPSEEK_CREATE_SESSION_URL,
                headers=get_auth_headers(auth_ctx),
                json={"agent": "chat"},
                timeout=30,
                impersonate="safari15_3",
            )
            data = response.json()
        except Exception as exc:
            logger.warning("[create_session] 请求异常: %s", exc)
        finally:
            if response is not None:
                await response.aclose()

        if response is not None and response.status_code == 200 and data.get("code") == 0:
            biz_data = data.get("data", {}).get("biz_data", {})
            # 新版 API 返回 biz_data.chat_session.id，兼容旧版 biz_data.id
            session_id = (
                biz_data.get("chat_session", {}).get("id")
                or biz_data.get("id")
            )
            return session_id

        logger.warning(
            "[create_session] 创建会话失败, status=%s, code=%s, msg=%s, 完整响应=%s, token=%s...",
            getattr(response, "status_code", None),
            data.get("code"),
            data.get("msg"),
            str(data)[:500],
            auth_ctx.deepseek_token[:8],
        )

        attempts += 1
        if auth_ctx.use_config_token and await switch_account(request, auth_ctx):
            continue

        await asyncio.sleep(1)

    return None


async def delete_chat_session(
    request: Request,
    auth_ctx: AuthContext,
    session_id: str,
    max_attempts: int = 2,
) -> bool:
    if not session_id:
        return False

    payload = {"chat_session_id": session_id, "id": session_id}
    attempts = 0
    while attempts < max_attempts:
        response = None
        try:
            response = await get_http_client(request).post(
                DEEPSEEK_DELETE_SESSION_URL,
                headers=get_auth_headers(auth_ctx),
                json=payload,
                timeout=20,
                impersonate="safari15_3",
            )
            try:
                data = response.json()
            except Exception:
                data = {}

            if response.status_code == 200 and (
                data.get("code") == 0 or data.get("success") is True
            ):
                logger.info("[delete_chat_session] 会话删除成功: %s", session_id)
                return True

            logger.warning(
                "[delete_chat_session] 删除失败, session=%s, status=%s, body=%s",
                session_id,
                response.status_code,
                str(data)[:300],
            )
        except Exception as exc:
            logger.warning(
                "[delete_chat_session] 请求异常(session=%s): %s",
                session_id,
                exc,
            )
        finally:
            if response is not None:
                await response.aclose()

        attempts += 1
        await asyncio.sleep(0.2)

    return False


def compute_pow_answer(
    algorithm: str,
    challenge_str: str,
    salt: str,
    difficulty: int,
    expire_at: int,
    signature: str,
    target_path: str,
    wasm_path: str,
) -> int | None:
    if algorithm != "DeepSeekHashV1":
        raise ValueError(f"不支持的算法：{algorithm}")

    prefix = f"{salt}_{expire_at}_"
    store = Store()
    linker = Linker(store.engine)

    with open(wasm_path, "rb") as file:
        wasm_bytes = file.read()

    module = Module(store.engine, wasm_bytes)
    instance = linker.instantiate(store, module)
    exports = instance.exports(store)

    try:
        memory = exports["memory"]
        add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
        alloc = exports["__wbindgen_export_0"]
        wasm_solve = exports["wasm_solve"]
    except KeyError as exc:
        raise RuntimeError(f"缺少 wasm 导出函数: {exc}") from exc

    def write_memory(offset: int, data: bytes) -> None:
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        ctypes.memmove(base_addr + offset, data, len(data))

    def read_memory(offset: int, size: int) -> bytes:
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        return ctypes.string_at(base_addr + offset, size)

    def encode_string(text: str) -> tuple[int, int]:
        data = text.encode("utf-8")
        ptr_value = alloc(store, len(data), 1)
        ptr = int(ptr_value.value) if hasattr(ptr_value, "value") else int(ptr_value)
        write_memory(ptr, data)
        return ptr, len(data)

    retptr = add_to_stack(store, -16)
    ptr_challenge, len_challenge = encode_string(challenge_str)
    ptr_prefix, len_prefix = encode_string(prefix)

    wasm_solve(
        store,
        retptr,
        ptr_challenge,
        len_challenge,
        ptr_prefix,
        len_prefix,
        float(difficulty),
    )

    status_bytes = read_memory(retptr, 4)
    if len(status_bytes) != 4:
        add_to_stack(store, 16)
        raise RuntimeError("读取状态字节失败")
    status = struct.unpack("<i", status_bytes)[0]

    value_bytes = read_memory(retptr + 8, 8)
    if len(value_bytes) != 8:
        add_to_stack(store, 16)
        raise RuntimeError("读取结果字节失败")
    value = struct.unpack("<d", value_bytes)[0]

    add_to_stack(store, 16)
    if status == 0:
        return None
    return int(value)


async def get_pow_response(
    request: Request,
    auth_ctx: AuthContext,
    max_attempts: int = 3,
) -> str | None:
    attempts = 0
    while attempts < max_attempts:
        response = None
        data: dict[str, Any] = {}
        try:
            response = await get_http_client(request).post(
                DEEPSEEK_CREATE_POW_URL,
                headers=get_auth_headers(auth_ctx),
                json={"target_path": "/api/v0/chat/completion"},
                timeout=30,
                impersonate="safari15_3",
            )
            data = response.json()
        except Exception as exc:
            logger.warning("[get_pow_response] 请求异常: %s", exc)
        finally:
            if response is not None:
                await response.aclose()

        if response is not None and response.status_code == 200 and data.get("code") == 0:
            challenge = (
                data.get("data", {})
                .get("biz_data", {})
                .get("challenge", {})
            )
            difficulty = challenge.get("difficulty", 144000)
            expire_at = challenge.get("expire_at", 1680000000)
            try:
                wasm_path = resolve_wasm_path()
                answer = await asyncio.to_thread(
                    compute_pow_answer,
                    challenge["algorithm"],
                    challenge["challenge"],
                    challenge["salt"],
                    difficulty,
                    expire_at,
                    challenge["signature"],
                    challenge["target_path"],
                    wasm_path,
                )
            except Exception as exc:
                logger.error("[get_pow_response] PoW 计算异常: %s", exc)
                answer = None

            if answer is None:
                attempts += 1
                await asyncio.sleep(0.5)
                continue

            pow_dict = {
                "algorithm": challenge["algorithm"],
                "challenge": challenge["challenge"],
                "salt": challenge["salt"],
                "answer": answer,
                "signature": challenge["signature"],
                "target_path": challenge["target_path"],
            }
            pow_str = json.dumps(pow_dict, separators=(",", ":"), ensure_ascii=False)
            return base64.b64encode(pow_str.encode("utf-8")).decode("utf-8").rstrip()

        logger.warning(
            "[get_pow_response] 获取 PoW 失败, status=%s, code=%s, msg=%s",
            getattr(response, "status_code", None),
            data.get("code"),
            data.get("msg"),
        )

        attempts += 1
        if auth_ctx.use_config_token and await switch_account(request, auth_ctx):
            continue

        await asyncio.sleep(1)

    return None


def _sync_completion_post(
    headers: dict[str, str],
    payload: dict[str, Any],
):
    """Synchronous streaming POST to DeepSeek completion endpoint.

    Using sync requests avoids curl_cffi AsyncSession's premature stream drops.
    """
    return requests.post(
        DEEPSEEK_COMPLETION_URL,
        headers=headers,
        json=payload,
        stream=True,
        impersonate="safari15_3",
    )


async def call_completion_endpoint(
    request: Request,
    headers: dict[str, str],
    payload: dict[str, Any],
    max_attempts: int = 3,
):
    attempts = 0
    while attempts < max_attempts:
        try:
            response = await asyncio.to_thread(
                _sync_completion_post, headers, payload,
            )
        except Exception as exc:
            logger.warning("[call_completion_endpoint] 请求异常: %s", exc)
            attempts += 1
            await asyncio.sleep(1)
            continue

        if response.status_code == 200:
            return SyncResponseAsyncAdapter(response)

        logger.warning(
            "[call_completion_endpoint] 调用失败, status=%s",
            response.status_code,
        )
        response.close()
        attempts += 1
        await asyncio.sleep(1)

    return None


def _format_search_tool_call(fragment: dict[str, Any], is_first: bool = False) -> str | None:
    """将 TOOL_SEARCH fragment 转换为思维链中的工具调用文本格式"""
    frag_type = str(fragment.get("type", "")).strip().upper()
    if frag_type != "TOOL_SEARCH":
        return None
    
    queries = fragment.get("queries", [])
    if not queries or not isinstance(queries, list):
        return None
    
    # 提取所有查询
    query_list = []
    for q in queries:
        if isinstance(q, dict) and q.get("query"):
            query_list.append(q["query"])
    
    if not query_list:
        return None
    
    # 构建工具调用格式
    tool_input = {
        "query": query_list[0] if query_list else "",
    }
    
    # 如果有多个查询，尝试从查询内容推断 time_range
    if len(query_list) > 1:
        combined = " ".join(query_list)
        if any(kw in combined for kw in ["今日", "今天", "最新", "latest", "today"]):
            tool_input["time_range"] = "day"
        elif any(kw in combined for kw in ["本周", "这周", "week"]):
            tool_input["time_range"] = "week"
        elif any(kw in combined for kw in ["本月", "这月", "month"]):
            tool_input["time_range"] = "month"
    
    prefix = "\n" if is_first else ""
    return f"{prefix}「调用工具: search 输入内容: {json.dumps(tool_input, ensure_ascii=False)}」\n"


def _is_citation_or_reference(text: str) -> bool:
    return text.startswith("[citation:") or text.startswith("[reference:")


def _emit_fragment_events(
    fragment: dict[str, Any],
    *,
    search_enabled: bool,
    current_event_type: str,
    is_first_tool_call: bool = False,
) -> tuple[list[dict[str, str]], str, bool]:
    """统一处理单个 fragment，返回要 yield 的事件列表。

    Returns:
        (events, updated_current_event_type, updated_is_first_tool_call)
    """
    events: list[dict[str, str]] = []
    frag_type = str(fragment.get("type", "")).strip().upper()

    if frag_type == "TIP":
        return events, current_event_type, is_first_tool_call

    # TOOL_SEARCH — 搜索查询
    if frag_type == "TOOL_SEARCH" and search_enabled:
        tool_call_text = _format_search_tool_call(fragment, is_first=is_first_tool_call)
        if tool_call_text:
            events.append({"type": "thinking", "content": tool_call_text})
            is_first_tool_call = False
        content = fragment.get("content", "")
        if isinstance(content, str) and content and not _is_citation_or_reference(content):
            events.append({"type": current_event_type, "content": content})
        return events, current_event_type, is_first_tool_call

    # TOOL_OPEN — 打开/浏览网页
    if frag_type == "TOOL_OPEN" and search_enabled:
        prefix = "\n" if is_first_tool_call else ""
        result = fragment.get("result")
        ref = fragment.get("reference")
        if isinstance(result, dict) and result:
            url = result.get("url", "")
            title = result.get("title", "")
            site_name = result.get("site_name", "")
            if title:
                source = site_name or "网页"
                events.append({"type": "thinking", "content": f"{prefix}「浏览: [{source}] {title[:40]}{'...' if len(title) > 40 else ''}」\n"})
                is_first_tool_call = False
            elif url:
                events.append({"type": "thinking", "content": f"{prefix}「打开: {url[:60]}{'...' if len(url) > 60 else ''}」\n"})
                is_first_tool_call = False
        elif isinstance(ref, dict):
            events.append({"type": "thinking", "content": f"{prefix}「继续浏览页面...」\n"})
            is_first_tool_call = False
        return events, current_event_type, is_first_tool_call

    # TOOL_FIND — 页面内查找
    if frag_type == "TOOL_FIND" and search_enabled:
        prefix = "\n" if is_first_tool_call else ""
        pattern = fragment.get("pattern", "")
        if pattern:
            events.append({"type": "thinking", "content": f"{prefix}「页面内查找: \"{pattern}\"」\n"})
            is_first_tool_call = False
        return events, current_event_type, is_first_tool_call

    # THINK — 思维链
    if frag_type == "THINK":
        current_event_type = "thinking"
        content = fragment.get("content", "")
        if isinstance(content, str) and content:
            events.append({"type": "thinking", "content": content})
            is_first_tool_call = True
        return events, current_event_type, is_first_tool_call

    # RESPONSE — 最终回复
    if frag_type == "RESPONSE":
        current_event_type = "text"
        content = fragment.get("content", "")
        if isinstance(content, str) and content:
            if not (search_enabled and _is_citation_or_reference(content)):
                events.append({"type": "text", "content": content})
                is_first_tool_call = True
        return events, current_event_type, is_first_tool_call

    # 未知类型 — 按内容输出
    current_event_type = "thinking" if frag_type == "THINK" else "text"
    content = fragment.get("content", "")
    if isinstance(content, str) and content:
        if not (search_enabled and _is_citation_or_reference(content)):
            events.append({"type": current_event_type, "content": content})
    return events, current_event_type, is_first_tool_call


async def iter_deepseek_events(
    response,
    *,
    search_enabled: bool,
):
    current_event_type = "text"
    is_first_tool_call = True
    event_count = 0

    async for raw_line in response.aiter_lines(decode_unicode=False):
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="ignore")
        else:
            line = str(raw_line)

        if not line:
            continue
        if not line.startswith("data:"):
            continue

        data_str = line[5:].strip()
        if data_str == "[DONE]":
            logger.info("[iter_deepseek_events] 收到 [DONE]")
            return

        try:
            chunk = json.loads(data_str)
        except Exception as exc:
            logger.warning("[iter_deepseek_events] JSON 解析失败: %s", exc)
            raise RuntimeError("Failed to parse upstream event.") from exc

        event_count += 1
        if event_count <= 10 or event_count % 50 == 0:
            logger.info("[iter_deepseek_events] 事件 #%d, p=%s", event_count, chunk.get("p", "null"))

        if is_finished_status_chunk(chunk):
            logger.info("[iter_deepseek_events] 收到 FINISHED 状态")
            return

        value = chunk.get("v")
        path = str(chunk.get("p", "")).strip()
        path_lower = path.lower()

        # 新版 API: v 是 dict 且包含 response → 从 fragments 检测内容类型
        if isinstance(value, dict):
            response_obj = value.get("response")
            if isinstance(response_obj, dict):
                fragments = response_obj.get("fragments", [])
                if fragments:
                    last_fragment = fragments[-1]
                    fragment_events, current_event_type, is_first_tool_call = _emit_fragment_events(
                        last_fragment,
                        search_enabled=search_enabled,
                        current_event_type=current_event_type,
                        is_first_tool_call=is_first_tool_call,
                    )
                    for event in fragment_events:
                        yield event
            continue

        # 新版 API: 新 fragment 追加 → 类型切换 (THINK → RESPONSE)
        if isinstance(value, list) and path == "response/fragments":
            for fragment in value:
                if not isinstance(fragment, dict):
                    continue
                fragment_events, current_event_type, is_first_tool_call = _emit_fragment_events(
                    fragment,
                    search_enabled=search_enabled,
                    current_event_type=current_event_type,
                    is_first_tool_call=is_first_tool_call,
                )
                for event in fragment_events:
                    yield event
            continue

        # 新版 API: BATCH 操作（包含多个 fragment 更新）
        if isinstance(value, list) and path == "response" and chunk.get("o") == "BATCH":
            for operation in value:
                if not isinstance(operation, dict):
                    continue
                op_path = str(operation.get("p", "")).strip()
                op_value = operation.get("v")
                # 处理 fragments 追加操作
                if op_path == "fragments" and isinstance(op_value, list):
                    for fragment in op_value:
                        if not isinstance(fragment, dict):
                            continue
                        fragment_events, current_event_type, is_first_tool_call = _emit_fragment_events(
                            fragment,
                            search_enabled=search_enabled,
                            current_event_type=current_event_type,
                            is_first_tool_call=is_first_tool_call,
                        )
                        for event in fragment_events:
                            yield event
                # 处理其他类型的 BATCH 操作（如 has_pending_fragment）
                elif op_path == "has_pending_fragment":
                    # 跳过状态更新
                    continue
            continue

        # 跳过非字符串值 (float elapsed_secs, int token_usage 等)
        if not isinstance(value, str):
            continue

        # 跳过搜索状态
        if path_lower == "response/search_status":
            continue

        # 旧版 API 兼容: thinking_content 路径
        if path_lower.endswith("thinking_content"):
            current_event_type = "thinking"
            yield {"type": "thinking", "content": value}
            continue

        # 有路径时，仅当路径指向 content 才输出
        if path:
            if path_lower.endswith("/content") or path_lower == "response/content":
                yield {"type": current_event_type, "content": value}
            # 其他路径 (elapsed_secs, status, token_usage 等) 跳过
            continue

        # 无路径的字符串值 → 内容 token
        if search_enabled and _is_citation_or_reference(value):
            continue
        yield {"type": current_event_type, "content": value}


async def collect_completion_output(
    completion: CompletionContext,
) -> tuple[str, str]:
    reasoning_parts: list[str] = []
    text_parts: list[str] = []

    async for event in iter_deepseek_events(
        completion.response,
        search_enabled=completion.search_enabled,
    ):
        if event["type"] == "thinking":
            if completion.thinking_enabled:
                reasoning_parts.append(event["content"])
        else:
            text_parts.append(event["content"])

    return "".join(reasoning_parts), "".join(text_parts)


async def cleanup_completion(
    request: Request,
    auth_ctx: AuthContext | None,
    completion: CompletionContext | None = None,
) -> None:
    if completion is not None and completion.response is not None:
        response = completion.response
        completion.response = None
        try:
            await response.aclose()
        except Exception as exc:
            logger.warning("[cleanup_completion] 关闭上游响应异常: %s", exc)

    if completion is not None and completion.session_id:
        try:
            await delete_chat_session(request, auth_ctx, completion.session_id)
        except Exception as exc:
            logger.warning(
                "[cleanup_completion] 删除会话异常(session=%s): %s",
                completion.session_id,
                exc,
            )
        completion.session_id = ""

    await release_auth_context(auth_ctx)


async def start_deepseek_completion(
    request: Request,
    auth_ctx: AuthContext,
    *,
    deepseek_model: str,
    messages: list[dict[str, Any]],
    output_model: str,
) -> CompletionContext:
    thinking_enabled, search_enabled, expert_enabled = resolve_deepseek_model_features(deepseek_model)
    final_prompt = messages_prepare(messages, thinking_enabled=thinking_enabled)

    session_id = await create_session(request, auth_ctx)
    if not session_id and auth_ctx.use_config_token and auth_ctx.account is not None:
        logger.info(
            "[start_deepseek_completion] 会话创建失败，尝试重新登录刷新 token, 账号=%s, 旧token=%s...",
            auth_ctx.account.get("email", "unknown"),
            auth_ctx.deepseek_token[:8],
        )
        try:
            new_token = await login_deepseek_via_account(request, auth_ctx.account)
            auth_ctx.deepseek_token = new_token
            logger.info(
                "[start_deepseek_completion] 重新登录成功, 新token=%s...",
                new_token[:8],
            )
            session_id = await create_session(request, auth_ctx)
            if session_id:
                logger.info(
                    "[start_deepseek_completion] 重登录后创建会话成功, session_id=%s",
                    session_id,
                )
            else:
                logger.warning("[start_deepseek_completion] 重登录后创建会话仍然失败")
        except Exception as exc:
            logger.warning("[start_deepseek_completion] 重新登录失败: %s", exc, exc_info=True)

    if not session_id:
        raise HTTPException(status_code=401, detail="invalid token.")

    pow_response = await get_pow_response(request, auth_ctx)
    if not pow_response:
        raise HTTPException(
            status_code=401,
            detail="Failed to get PoW (invalid token or unknown error).",
        )

    headers = {**get_auth_headers(auth_ctx), "x-ds-pow-response": pow_response}
    payload = {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "model_type": "expert" if expert_enabled else "default",
        "prompt": final_prompt,
        "ref_file_ids": [],
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
        "preempt": False,
    }

    response = await call_completion_endpoint(request, headers, payload, max_attempts=3)
    if response is None:
        raise HTTPException(status_code=500, detail="Failed to get completion.")

    return CompletionContext(
        response=response,
        session_id=session_id,
        prompt=final_prompt,
        output_model=output_model,
        thinking_enabled=thinking_enabled,
        search_enabled=search_enabled,
        expert_enabled=expert_enabled,
    )


def build_openai_response_payload(
    completion: CompletionContext,
    created_time: int,
    reasoning: str,
    content: str,
    tool_calls: list[Any],
) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": "assistant",
    }

    if tool_calls:
        message["content"] = None
        message["tool_calls"] = [tool_call.to_openai_dict() for tool_call in tool_calls]
    else:
        message["content"] = content

    if reasoning:
        message["reasoning_content"] = reasoning

    return {
        "id": completion.session_id,
        "object": "chat.completion",
        "created": created_time,
        "model": completion.output_model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": build_usage(completion.prompt, reasoning, content),
    }


def build_claude_response_payload(
    model: str,
    input_tokens: int,
    reasoning: str,
    content: str,
    tool_calls: list[Any],
) -> dict[str, Any]:
    response_content: list[dict[str, Any]] = []

    if reasoning:
        response_content.append(
            {
                "type": "thinking",
                "thinking": reasoning,
            }
        )

    if tool_calls:
        response_content.extend(tool_call.to_anthropic_dict() for tool_call in tool_calls)
        stop_reason = "tool_use"
    else:
        response_content.append(
            {
                "type": "text",
                "text": content or "",
            }
        )
        stop_reason = "end_turn"

    return {
        "id": f"msg_{int(time.time())}_{random.randint(1000, 9999)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": response_content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": estimate_tokens(reasoning + content),
        },
    }


def serialize_openai_chunk(
    *,
    completion: CompletionContext,
    created_time: int,
    choices: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "id": completion.session_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": completion.output_model,
        "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def format_tool_result_message(tool_name: str, tool_call_id: str, content: Any) -> str:
    normalized = normalize_text_content(content)
    return (
        f"Tool result for {tool_name} "
        f"(tool_call_id={tool_call_id or 'unknown'}):\n{normalized}"
    )


def strip_xml_tool_call_blocks(text: str) -> str:
    if not text:
        return text

    cleaned = re.sub(
        r'<<<tool_call>>>.*?<<</tool_call>>>',
        '',
        text,
        flags=re.DOTALL,
    )
    cleaned = re.sub(
        r'<tool_call>.*?</tool_call>',
        '',
        cleaned,
        flags=re.DOTALL,
    )
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def inject_tool_prompt(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_choice: Any = None,
) -> list[dict[str, Any]]:
    if not tools:
        return messages
    instruction = build_tool_system_prompt(tools, tool_choice)
    return prepend_system_instruction(messages, instruction)


@app.get("/v1/models")
def list_models():
    return JSONResponse(content={"object": "list", "data": OPENAI_MODELS}, status_code=200)


@app.get("/anthropic/v1/models")
def list_claude_models():
    return JSONResponse(content={"object": "list", "data": CLAUDE_MODELS}, status_code=200)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    auth_ctx: AuthContext | None = None
    completion: CompletionContext | None = None
    cleanup_is_owned_by_stream = False

    try:
        auth_ctx = await determine_mode_and_token(request)
        req_data = await request.json()

        model = str(req_data.get("model", "")).strip()
        raw_messages = req_data.get("messages", [])
        if not model or not raw_messages:
            raise HTTPException(
                status_code=400,
                detail="Request must include 'model' and 'messages'.",
            )

        tools = normalize_tool_definitions(req_data.get("tools") or [])
        tool_choice = req_data.get("tool_choice")
        logger.info(
            "[chat_completions] 收到工具定义 %d 个: %s",
            len(tools),
            [t["name"] for t in tools],
        )
        normalized_messages = normalize_openai_messages(raw_messages)
        normalized_messages = inject_tool_prompt(normalized_messages, tools, tool_choice)

        completion = await start_deepseek_completion(
            request,
            auth_ctx,
            deepseek_model=model,
            messages=normalized_messages,
            output_model=model,
        )

        created_time = int(time.time())
        streaming = bool(req_data.get("stream", False))

        if streaming:
            cleanup_is_owned_by_stream = True

            async def openai_live_stream():
                first_chunk_sent = False
                final_reasoning = ""
                final_content = ""
                reasoning_tool_parser = XmlToolCallStreamParser(tools) if tools else None
                content_tool_parser = XmlToolCallStreamParser(tools) if tools else None
                emitted_tool_call_count = 0
                has_tool_calls = False
                logger.info(
                    "[openai_live_stream] 流式开始: tools=%d个, thinking_enabled=%s, tool_names=%s",
                    len(tools),
                    completion.thinking_enabled,
                    [t["name"] for t in tools],
                )

                def build_delta(extra: dict[str, Any]) -> dict[str, Any]:
                    nonlocal first_chunk_sent
                    delta = dict(extra)
                    if not first_chunk_sent:
                        delta = {"role": "assistant", **delta}
                        first_chunk_sent = True
                    return delta

                async def emit_tool_call(tool_call: Any) -> str:
                    return serialize_openai_chunk(
                        completion=completion,
                        created_time=created_time,
                        choices=[
                            {
                                "index": 0,
                                "delta": build_delta(
                                    {
                                        "tool_calls": [
                                            {
                                                "index": emitted_tool_call_count,
                                                **tool_call.to_openai_dict(),
                                            }
                                        ]
                                    }
                                ),
                            }
                        ],
                    )

                try:
                    async for event in iter_deepseek_events(
                        completion.response,
                        search_enabled=completion.search_enabled,
                    ):
                        event_type = event["type"]
                        event_content = event["content"]

                        if event_type == "thinking":
                            if not completion.thinking_enabled:
                                continue

                            if not tools:
                                final_reasoning += event_content
                                yield serialize_openai_chunk(
                                    completion=completion,
                                    created_time=created_time,
                                    choices=[
                                        {
                                            "index": 0,
                                            "delta": build_delta(
                                                {"reasoning_content": event_content}
                                            ),
                                        }
                                    ],
                                )
                                continue

                            text_chunks, parsed_tool_calls = reasoning_tool_parser.feed(event_content)

                            for text_chunk in text_chunks:
                                if not text_chunk:
                                    continue
                                final_reasoning += text_chunk
                                yield serialize_openai_chunk(
                                    completion=completion,
                                    created_time=created_time,
                                    choices=[
                                        {
                                            "index": 0,
                                            "delta": build_delta(
                                                {"reasoning_content": text_chunk}
                                            ),
                                        }
                                    ],
                                )

                            for tool_call in parsed_tool_calls:
                                has_tool_calls = True
                                yield await emit_tool_call(tool_call)
                                emitted_tool_call_count += 1

                            continue

                        if not tools:
                            final_content += event_content
                            yield serialize_openai_chunk(
                                completion=completion,
                                created_time=created_time,
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": build_delta({"content": event_content}),
                                    }
                                ],
                            )
                            continue

                        text_chunks, parsed_tool_calls = content_tool_parser.feed(event_content)

                        for text_chunk in text_chunks:
                            if not text_chunk:
                                continue
                            final_content += text_chunk
                            yield serialize_openai_chunk(
                                completion=completion,
                                created_time=created_time,
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": build_delta({"content": text_chunk}),
                                    }
                                ],
                            )

                        for tool_call in parsed_tool_calls:
                            has_tool_calls = True
                            yield await emit_tool_call(tool_call)
                            emitted_tool_call_count += 1

                    if tools:
                        logger.info(
                            "[openai_live_stream] 流式事件结束, 开始finish: reasoning长度=%d, content长度=%d, 已发出tool_calls=%d",
                            len(final_reasoning),
                            len(final_content),
                            emitted_tool_call_count,
                        )
                        remaining_reasoning_chunks, remaining_reasoning_tool_calls = reasoning_tool_parser.finish()
                        for text_chunk in remaining_reasoning_chunks:
                            if not text_chunk:
                                continue
                            final_reasoning += text_chunk
                            yield serialize_openai_chunk(
                                completion=completion,
                                created_time=created_time,
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": build_delta(
                                            {"reasoning_content": text_chunk}
                                        ),
                                    }
                                ],
                            )

                        for tool_call in remaining_reasoning_tool_calls:
                            has_tool_calls = True
                            yield await emit_tool_call(tool_call)
                            emitted_tool_call_count += 1

                        remaining_content_chunks, remaining_content_tool_calls = content_tool_parser.finish()
                        for text_chunk in remaining_content_chunks:
                            if not text_chunk:
                                continue
                            final_content += text_chunk
                            yield serialize_openai_chunk(
                                completion=completion,
                                created_time=created_time,
                                choices=[
                                    {
                                        "index": 0,
                                        "delta": build_delta({"content": text_chunk}),
                                    }
                                ],
                            )

                        for tool_call in remaining_content_tool_calls:
                            has_tool_calls = True
                            yield await emit_tool_call(tool_call)
                            emitted_tool_call_count += 1

                    logger.info(
                        "[openai_live_stream] 流式完成: has_tool_calls=%s, emitted_tool_call_count=%d",
                        has_tool_calls,
                        emitted_tool_call_count,
                    )
                    usage = build_usage(completion.prompt, final_reasoning, final_content)
                    yield serialize_openai_chunk(
                        completion=completion,
                        created_time=created_time,
                        choices=[
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls" if has_tool_calls else "stop",
                            }
                        ],
                        usage=usage,
                    )
                    yield "data: [DONE]\n\n"
                except Exception:
                    logger.exception("[chat_completions] 流式响应异常")
                finally:
                    await cleanup_completion(request, auth_ctx, completion)

            return StreamingResponse(
                openai_live_stream(),
                media_type="text/event-stream",
                headers={"Content-Type": "text/event-stream"},
            )

        final_reasoning, final_content = await collect_completion_output(completion)
        logger.info(
            "[chat_completions] 非流式收集完毕: reasoning长度=%d, content长度=%d",
            len(final_reasoning),
            len(final_content),
        )
        tool_calls = []
        if tools:
            combined_tool_source = "\n".join(
                part for part in (final_reasoning, final_content) if part
            )
            logger.info(
                "[chat_completions] 合并后工具源文本长度=%d, 前200字符: %s",
                len(combined_tool_source),
                repr(combined_tool_source[:200]),
            )
            tool_calls = extract_tool_calls_from_text(combined_tool_source, tools)
            logger.info(
                "[chat_completions] 工具调用解析结果: %d 个, names=%s",
                len(tool_calls),
                [tc.name for tc in tool_calls],
            )
            if tool_calls:
                final_reasoning = strip_xml_tool_call_blocks(final_reasoning)
                final_content = strip_xml_tool_call_blocks(final_content)

        result = build_openai_response_payload(
            completion=completion,
            created_time=created_time,
            reasoning=final_reasoning,
            content=final_content,
            tool_calls=tool_calls,
        )
        return JSONResponse(content=result, status_code=200)

    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
    except Exception:
        logger.exception("[chat_completions] 未知异常")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})
    finally:
        if not cleanup_is_owned_by_stream:
            await cleanup_completion(request, auth_ctx, completion)


@app.post("/anthropic/v1/messages")
async def claude_messages(request: Request):
    auth_ctx: AuthContext | None = None
    completion: CompletionContext | None = None
    cleanup_is_owned_by_stream = False

    try:
        auth_ctx = await determine_mode_and_token(request)
        req_data = await request.json()

        model = str(req_data.get("model", "")).strip() or CLAUDE_DEFAULT_MODEL
        raw_messages = req_data.get("messages", [])
        if not raw_messages:
            raise HTTPException(
                status_code=400,
                detail="Request must include 'messages'.",
            )

        tools = normalize_tool_definitions(req_data.get("tools") or [])
        tool_choice = req_data.get("tool_choice")
        system_prompt = normalize_text_content(req_data.get("system", ""))

        normalized_messages = normalize_claude_messages(raw_messages)
        if system_prompt:
            normalized_messages = prepend_system_instruction(normalized_messages, system_prompt)
        normalized_messages = inject_tool_prompt(normalized_messages, tools, tool_choice)

        completion = await start_deepseek_completion(
            request,
            auth_ctx,
            deepseek_model=map_claude_model_to_deepseek(model),
            messages=normalized_messages,
            output_model=model,
        )

        final_reasoning, final_content = await collect_completion_output(completion)
        tool_calls = []
        if tools:
            combined_tool_source = "\n".join(
                part for part in (final_reasoning, final_content) if part
            )
            tool_calls = extract_tool_calls_from_text(combined_tool_source, tools)
            if tool_calls:
                final_reasoning = strip_xml_tool_call_blocks(final_reasoning)
                final_content = strip_xml_tool_call_blocks(final_content)

        input_tokens = estimate_tokens(
            json.dumps(
                {
                    "system": system_prompt,
                    "messages": normalized_messages,
                    "tools": tools,
                },
                ensure_ascii=False,
            )
        )

        if bool(req_data.get("stream", False)):
            cleanup_is_owned_by_stream = True

            async def claude_stream():
                try:
                    message_id = f"msg_{int(time.time())}_{random.randint(1000, 9999)}"
                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "message_start",
                                "message": {
                                    "id": message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "model": model,
                                    "content": [],
                                    "stop_reason": None,
                                    "stop_sequence": None,
                                    "usage": {
                                        "input_tokens": input_tokens,
                                        "output_tokens": 0,
                                    },
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )

                    block_index = 0
                    if tool_calls:
                        for tool_call in tool_calls:
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        "type": "content_block_start",
                                        "index": block_index,
                                        "content_block": tool_call.to_anthropic_dict(),
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )
                            yield (
                                "data: "
                                + json.dumps(
                                    {
                                        "type": "content_block_stop",
                                        "index": block_index,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )
                            block_index += 1

                        stop_reason = "tool_use"
                    else:
                        text_block = final_content or ""
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "content_block_start",
                                    "index": 0,
                                    "content_block": {"type": "text", "text": ""},
                                },
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        )
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": text_block},
                                },
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        )
                        yield (
                            "data: "
                            + json.dumps(
                                {
                                    "type": "content_block_stop",
                                    "index": 0,
                                },
                                ensure_ascii=False,
                            )
                            + "\n\n"
                        )
                        stop_reason = "end_turn"

                    yield (
                        "data: "
                        + json.dumps(
                            {
                                "type": "message_delta",
                                "delta": {
                                    "stop_reason": stop_reason,
                                    "stop_sequence": None,
                                },
                                "usage": {
                                    "output_tokens": estimate_tokens(final_reasoning + final_content),
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n\n"
                    )
                    yield "data: " + json.dumps({"type": "message_stop"}, ensure_ascii=False) + "\n\n"
                finally:
                    await cleanup_completion(request, auth_ctx, completion)

            return StreamingResponse(
                claude_stream(),
                media_type="text/event-stream",
                headers={"Content-Type": "text/event-stream"},
            )

        result = build_claude_response_payload(
            model=model,
            input_tokens=input_tokens,
            reasoning=final_reasoning,
            content=final_content,
            tool_calls=tool_calls,
        )
        return JSONResponse(content=result, status_code=200)

    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": "invalid_request_error", "message": exc.detail}},
        )
    except Exception:
        logger.exception("[claude_messages] 未知异常")
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "api_error", "message": "Internal Server Error"}},
        )
    finally:
        if not cleanup_is_owned_by_stream:
            await cleanup_completion(request, auth_ctx, completion)


@app.post("/anthropic/v1/messages/count_tokens")
async def claude_count_tokens(request: Request):
    try:
        ensure_authorized(request)
        req_data = await request.json()

        model = str(req_data.get("model", "")).strip() or CLAUDE_DEFAULT_MODEL
        messages = normalize_claude_messages(req_data.get("messages", []))
        system_prompt = normalize_text_content(req_data.get("system", ""))
        tools = normalize_tool_definitions(req_data.get("tools") or [])

        if not model or not messages:
            raise HTTPException(
                status_code=400,
                detail="Request must include 'model' and 'messages'.",
            )

        input_tokens = 0
        if system_prompt:
            input_tokens += estimate_tokens(system_prompt)

        for message in messages:
            input_tokens += 2
            input_tokens += estimate_tokens(message.get("content", ""))

        for tool in tools:
            input_tokens += estimate_tokens(tool.get("name", ""))
            input_tokens += estimate_tokens(tool.get("description", ""))
            input_tokens += estimate_tokens(
                json.dumps(tool.get("input_schema", {}), ensure_ascii=False)
            )

        return JSONResponse(
            content={"input_tokens": max(1, input_tokens)},
            status_code=200,
        )

    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": "invalid_request_error", "message": exc.detail}},
        )
    except Exception:
        logger.exception("[claude_count_tokens] 未知异常")
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "api_error", "message": "Internal Server Error"}},
        )


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5001)
