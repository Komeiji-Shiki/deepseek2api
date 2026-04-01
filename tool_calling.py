from __future__ import annotations

import ast
import html
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable


XML_TOOL_CALL_PREFIX = "<<<tool_call>>>"
XML_TOOL_CALL_SUFFIX = "<<</tool_call>>>"
LEGACY_XML_TOOL_CALL_PREFIX = "<tool_call>"
LEGACY_XML_TOOL_CALL_SUFFIX = "</tool_call>"

TOOL_CALL_PREFIXES = (
    XML_TOOL_CALL_PREFIX,
    LEGACY_XML_TOOL_CALL_PREFIX,
)
TOOL_CALL_SUFFIX_BY_PREFIX = {
    XML_TOOL_CALL_PREFIX: XML_TOOL_CALL_SUFFIX,
    LEGACY_XML_TOOL_CALL_PREFIX: LEGACY_XML_TOOL_CALL_SUFFIX,
}

TOOL_CALL_JSON_START_PATTERN = re.compile(r'\{\s*"tool_calls"\s*:', re.DOTALL)
XML_TOOL_CALL_PATTERN = re.compile(
    r'<<<tool_call>>>\s*name:\s*(.*?)\s*arguments:\s*(.*?)\s*<<</tool_call>>>',
    re.DOTALL | re.IGNORECASE,
)
LEGACY_XML_TOOL_CALL_PATTERN = re.compile(
    r'<tool_call>\s*<name>(.*?)</name>\s*<arguments>(.*?)</arguments>\s*</tool_call>',
    re.DOTALL | re.IGNORECASE,
)
STRICT_WRAPPER_PATTERN = re.compile(
    r"<<<\s*tool_call\s*>>>(?P<body>[\s\S]*?)<<<\s*/tool_call\s*>>>",
    re.IGNORECASE,
)
GENERIC_TAG_BLOCK_PATTERN = re.compile(
    r"<(?P<tag>tool_call|tool|function_call|invoke|call_tool)\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</(?P=tag)>",
    re.IGNORECASE,
)
SELF_CLOSING_TAG_PATTERN = re.compile(
    r"<(?P<tag>tool_call|tool|function_call|invoke|call_tool)\b(?P<attrs>[^>]*)/>",
    re.IGNORECASE,
)
SIMPLE_NAME_ARGS_PATTERN = re.compile(
    r"name:\s*(?P<name>[^\n\r]+?)\s*arguments:\s*(?P<args>.*?)(?=(?:\n\s*name:)|$)",
    re.IGNORECASE | re.DOTALL,
)
CHINESE_TOOL_CALL_PATTERN = re.compile(
    r"「\s*调用工具\s*[:：]\s*(?P<name>[^\s，,」]+)\s*输入内容\s*[:：]\s*(?P<args>[\s\S]*?)\s*」",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:24]}")

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }

    def to_anthropic_dict(self) -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.call_id,
            "name": self.name,
            "input": self.arguments,
        }


def _safe_json_loads(text: Any) -> Any:
    if text is None:
        return None
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None

    candidate = html.unescape(text).strip()
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except Exception:
        pass

    try:
        return ast.literal_eval(candidate)
    except Exception:
        return None


def parse_tool_call_arguments(arguments_text: str) -> dict[str, Any] | None:
    normalized = _normalize_arguments_dict(arguments_text)
    return normalized if isinstance(normalized, dict) else None


def _normalize_arguments_dict(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}

    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, list):
        return {"items": arguments}

    if isinstance(arguments, str):
        stripped = html.unescape(arguments).strip()
        if not stripped:
            return {}
        parsed = _safe_json_loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"items": parsed}
        return {"input": stripped}

    return {"input": str(arguments)}


def _coerce_tool_name_and_arguments(
    raw_tool_call: Any,
) -> tuple[str, dict[str, Any]] | None:
    if isinstance(raw_tool_call, ToolCall):
        return raw_tool_call.name, raw_tool_call.arguments

    if not isinstance(raw_tool_call, dict):
        return None

    function = raw_tool_call.get("function", {})
    if not isinstance(function, dict):
        function = {}

    tool_name = (
        str(raw_tool_call.get("name", "")).strip()
        or str(function.get("name", "")).strip()
    )
    if not tool_name:
        return None

    raw_arguments = (
        raw_tool_call.get("arguments")
        if "arguments" in raw_tool_call
        else raw_tool_call.get("input")
    )
    if raw_arguments is None:
        raw_arguments = function.get("arguments", {})

    return tool_name, _normalize_arguments_dict(raw_arguments)


def render_xml_tool_call_block(name: str, arguments: dict[str, Any]) -> str:
    arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
    return "\n".join(
        [
            XML_TOOL_CALL_PREFIX,
            f"name: {name}",
            f"arguments: {arguments_json}",
            XML_TOOL_CALL_SUFFIX,
        ]
    )


def render_tool_calls_as_xml_blocks(raw_tool_calls: Iterable[Any]) -> str:
    blocks: list[str] = []

    for raw_tool_call in raw_tool_calls:
        coerced = _coerce_tool_name_and_arguments(raw_tool_call)
        if coerced is None:
            continue
        tool_name, arguments = coerced
        blocks.append(render_xml_tool_call_block(tool_name, arguments))

    return "\n\n".join(blocks)


def _find_first_tool_call_prefix(text: str) -> tuple[int, str | None]:
    matches = [
        (text.find(prefix), prefix)
        for prefix in TOOL_CALL_PREFIXES
        if text.find(prefix) != -1
    ]
    if not matches:
        return -1, None
    index, prefix = min(matches, key=lambda item: item[0])
    return index, prefix


def _find_pending_tool_call_prefix_fragment(text: str) -> str:
    best_fragment = ""

    for prefix in TOOL_CALL_PREFIXES:
        max_size = min(len(text), len(prefix) - 1)
        for size in range(max_size, 0, -1):
            candidate = prefix[:size]
            if text.endswith(candidate) and size > len(best_fragment):
                best_fragment = candidate
                break

    return best_fragment


def normalize_text_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "tool_result":
                    parts.append(str(item.get("content", "")))
                elif item_type == "tool_use":
                    parts.append(
                        json.dumps(
                            {
                                "type": "tool_use",
                                "name": item.get("name"),
                                "input": item.get("input", {}),
                            },
                            ensure_ascii=False,
                        )
                    )
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)

    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", ""))
        if "content" in content:
            return str(content.get("content", ""))
        return json.dumps(content, ensure_ascii=False)

    return str(content)


def render_assistant_think_block(content: Any) -> str:
    text = normalize_text_content(content).strip()
    if not text:
        return ""
    return f"<think>{text}</think>"


def normalize_tool_definitions(tools: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            function = tool["function"]
            name = str(function.get("name", "")).strip()
            description = str(function.get("description", "")).strip()
            input_schema = function.get("parameters") or {}
        else:
            name = str(tool.get("name", "")).strip()
            description = str(tool.get("description", "")).strip()
            input_schema = tool.get("input_schema") or {}

        if not name or name in seen_names:
            continue

        normalized.append(
            {
                "name": name,
                "description": description,
                "input_schema": input_schema if isinstance(input_schema, dict) else {},
            }
        )
        seen_names.add(name)

    return normalized


def build_tool_system_prompt(
    tool_definitions: list[dict[str, Any]],
    tool_choice: Any = None,
) -> str:
    tool_blocks: list[str] = []
    for tool in tool_definitions:
        schema_json = json.dumps(
            tool.get("input_schema", {}),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        tool_blocks.append(
            "\n".join(
                [
                    f"工具名：{tool['name']}",
                    f"说明：{tool.get('description', '') or '无说明'}",
                    f"参数 JSON Schema：{schema_json}",
                ]
            )
        )

    tool_choice_instruction = "按需调用工具，不需要就直接回答。"
    if tool_choice == "required":
        tool_choice_instruction = "你必须先至少调用一个工具，再继续回答。"
    elif isinstance(tool_choice, dict):
        function_name = (
            tool_choice.get("function", {}).get("name")
            if isinstance(tool_choice.get("function"), dict)
            else ""
        )
        function_name = str(function_name).strip()
        if function_name:
            tool_choice_instruction = f"你必须先调用工具「{function_name}」，再继续回答。"
    elif tool_choice == "none":
        tool_choice_instruction = "不要调用任何工具，直接正常回答。"

    return "\n\n".join(
        [
            "你可以使用下面这些工具：",
            "\n\n".join(tool_blocks) if tool_blocks else "当前没有可用工具。",
            "正文输出必须遵守下面的规则：",
            "1. 需要调用工具时，正文只能输出 XML 工具块，前后不能夹带任何别的文字。",
            "2. 不需要调用工具时，正文直接正常回答，不能包含工具块。",
            "3. 思考、分析、计划都写在思维链里，不要混进工具块。",
            tool_choice_instruction,
            "工具调用格式如下：",
            XML_TOOL_CALL_PREFIX,
            "name: 工具名",
            'arguments: {"参数名":"参数值"}',
            XML_TOOL_CALL_SUFFIX,
            "arguments 必须是合法 JSON 对象。",
            "可以连续输出多个工具块。",
            "不要加 Markdown 代码块。",
        ]
    )


def prepend_system_instruction(
    messages: list[dict[str, Any]],
    instruction: str,
) -> list[dict[str, Any]]:
    if not instruction:
        return [message.copy() for message in messages]

    merged = [message.copy() for message in messages]
    if merged and merged[0].get("role") == "system":
        existing = normalize_text_content(merged[0].get("content"))
        merged[0]["content"] = (
            f"{instruction}\n\n{existing}" if existing else instruction
        )
    else:
        merged.insert(0, {"role": "system", "content": instruction})
    return merged


def _collect_openai_active_tool_assistant_indexes(
    messages: list[dict[str, Any]],
) -> set[int]:
    active_indexes: set[int] = set()
    in_active_chain = False

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            break

        role = str(message.get("role", "")).strip()
        is_tool_message = role == "tool"
        has_tool_calls = (
            role == "assistant"
            and isinstance(message.get("tool_calls"), list)
            and bool(message.get("tool_calls"))
        )

        if not in_active_chain:
            if is_tool_message:
                in_active_chain = True
            else:
                break

        if is_tool_message:
            continue
        if has_tool_calls:
            active_indexes.add(index)
            continue
        break

    return active_indexes


def normalize_openai_messages(messages: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_messages = [message for message in messages if isinstance(message, dict)]
    active_tool_assistant_indexes = _collect_openai_active_tool_assistant_indexes(raw_messages)

    normalized: list[dict[str, Any]] = []
    tool_name_by_id: dict[str, str] = {}

    for index, message in enumerate(raw_messages):
        role = str(message.get("role", "")).strip()
        content_text = normalize_text_content(message.get("content"))
        reasoning_text = ""
        if role == "assistant" and index in active_tool_assistant_indexes:
            assistant_reasoning = message.get("reasoning_content")
            if assistant_reasoning is None:
                assistant_reasoning = message.get("thinking_content")
            reasoning_text = render_assistant_think_block(assistant_reasoning)

        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            serialized_tool_calls: list[dict[str, Any]] = []

            for raw_call in message["tool_calls"]:
                coerced = _coerce_tool_name_and_arguments(raw_call)
                if coerced is None:
                    continue

                tool_name, arguments = coerced
                tool_call_id = str(raw_call.get("id", "")).strip()
                if tool_call_id:
                    tool_name_by_id[tool_call_id] = tool_name

                serialized_tool_calls.append(
                    {
                        "name": tool_name,
                        "arguments": arguments,
                    }
                )

            assistant_parts: list[str] = []
            if reasoning_text:
                assistant_parts.append(reasoning_text)
            if content_text:
                assistant_parts.append(content_text)

            xml_tool_blocks = render_tool_calls_as_xml_blocks(serialized_tool_calls)
            if xml_tool_blocks:
                assistant_parts.append(xml_tool_blocks)

            normalized.append(
                {
                    "role": "assistant",
                    "content": "\n\n".join(part for part in assistant_parts if part),
                }
            )
            continue

        if role == "tool":
            tool_call_id = str(message.get("tool_call_id", "")).strip()
            tool_name = tool_name_by_id.get(tool_call_id, "unknown_tool")
            tool_result_text = normalize_text_content(message.get("content"))
            normalized.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name} "
                        f"(tool_call_id={tool_call_id or 'unknown'}):\n{tool_result_text}"
                    ),
                }
            )
            continue

        normalized.append({"role": role, "content": content_text})

    return normalized


def _iter_tool_call_json_candidates(response_text: str) -> Iterable[dict[str, Any]]:
    stripped = response_text.strip()
    if not stripped:
        return []

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []

    possible_texts = [stripped]
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            possible_texts.append("\n".join(lines[1:-1]).strip())

    for text in possible_texts:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            candidates.append(parsed)

    for match in TOOL_CALL_JSON_START_PATTERN.finditer(response_text):
        sub_text = response_text[match.start():].lstrip()
        try:
            parsed, _ = decoder.raw_decode(sub_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("tool_calls"), list):
            candidates.append(parsed)

    return candidates


def _resolve_tool_name(raw_name: str, allowed_names: set[str]) -> str:
    name = (raw_name or "").strip()
    if not name or not allowed_names:
        return name

    if name in allowed_names:
        return name

    lowered = name.lower()

    exact_ci = [item for item in allowed_names if item.lower() == lowered]
    if len(exact_ci) == 1:
        return exact_ci[0]

    suffix = f"_{lowered}"
    suffix_matches = [item for item in allowed_names if item.lower().endswith(suffix)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    contains_matches = [item for item in allowed_names if lowered in item.lower()]
    if len(contains_matches) == 1:
        return contains_matches[0]
    if len(contains_matches) > 1:
        contains_matches.sort(key=len)
        return contains_matches[0]

    return name


def _extract_name_from_attrs(attrs: str) -> str:
    name_match = re.search(
        r'(?:name|tool|function)\s*=\s*([\'"])(.*?)\1',
        attrs,
        re.IGNORECASE | re.DOTALL,
    )
    return (name_match.group(2) or "").strip() if name_match else ""


def _extract_args_from_attrs(attrs: str) -> Any:
    args_match = re.search(
        r'(?:arguments|args|input|params)\s*=\s*([\'"])(.*?)\1',
        attrs,
        re.IGNORECASE | re.DOTALL,
    )
    return args_match.group(2) if args_match else None


def parse_xml_tool_calls(
    response_text: str,
    tool_definitions: list[dict[str, Any]],
) -> list[ToolCall]:
    allowed_names = {tool["name"] for tool in tool_definitions}
    if not response_text.strip() or not allowed_names:
        return []

    parsed_calls: list[ToolCall] = []
    seen_signatures: set[tuple[str, str]] = set()

    def append_call(raw_name: str, raw_arguments: Any) -> None:
        resolved_name = _resolve_tool_name(raw_name, allowed_names)
        if resolved_name not in allowed_names:
            return

        arguments = _normalize_arguments_dict(raw_arguments)
        signature = (
            resolved_name,
            json.dumps(arguments, ensure_ascii=False, sort_keys=True),
        )
        if signature in seen_signatures:
            return

        seen_signatures.add(signature)
        parsed_calls.append(ToolCall(name=resolved_name, arguments=arguments))

    for wrapper_match in STRICT_WRAPPER_PATTERN.finditer(response_text):
        wrapper_body = html.unescape(wrapper_match.group("body") or "").strip()
        found_inside_wrapper = False

        tool_tag_pattern = re.compile(
            r"<tool\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</tool>",
            re.IGNORECASE,
        )
        for tool_match in tool_tag_pattern.finditer(wrapper_body):
            raw_name = _extract_name_from_attrs(tool_match.group("attrs") or "")
            raw_arguments = html.unescape(tool_match.group("body") or "").strip()
            append_call(raw_name, raw_arguments)
            found_inside_wrapper = True

        if found_inside_wrapper:
            continue

        simple_matches = list(SIMPLE_NAME_ARGS_PATTERN.finditer(wrapper_body))
        if simple_matches:
            for match in simple_matches:
                append_call(match.group("name"), match.group("args"))
            continue

    for raw_name, raw_arguments in XML_TOOL_CALL_PATTERN.findall(response_text):
        append_call(raw_name, raw_arguments)

    for raw_name, raw_arguments in LEGACY_XML_TOOL_CALL_PATTERN.findall(response_text):
        append_call(raw_name, raw_arguments)

    for block_match in GENERIC_TAG_BLOCK_PATTERN.finditer(response_text):
        tag = (block_match.group("tag") or "").lower()
        attrs = html.unescape(block_match.group("attrs") or "")
        body = html.unescape(block_match.group("body") or "").strip()

        raw_name = _extract_name_from_attrs(attrs)
        raw_arguments: Any = _extract_args_from_attrs(attrs)

        if body:
            body_name_match = re.search(
                r"<(?:name|tool|function)>([\s\S]*?)</(?:name|tool|function)>",
                body,
                re.IGNORECASE,
            )
            if body_name_match and not raw_name:
                raw_name = body_name_match.group(1).strip()

            body_args_match = re.search(
                r"<(?:arguments|args|input|params)>([\s\S]*?)</(?:arguments|args|input|params)>",
                body,
                re.IGNORECASE,
            )
            if body_args_match:
                raw_arguments = body_args_match.group(1).strip()
            elif raw_arguments is None:
                parsed_body = _safe_json_loads(body)
                if isinstance(parsed_body, dict):
                    if not raw_name:
                        raw_name = (
                            parsed_body.get("name")
                            or parsed_body.get("tool")
                            or parsed_body.get("tool_name")
                            or parsed_body.get("function")
                            or parsed_body.get("function_name")
                            or ""
                        )
                    raw_arguments = parsed_body.get(
                        "arguments",
                        parsed_body.get(
                            "args",
                            parsed_body.get(
                                "input",
                                parsed_body.get("params", {}),
                            ),
                        ),
                    )
                elif tag == "tool" and raw_name:
                    raw_arguments = body
                elif tag != "tool_call":
                    raw_arguments = {"input": body}

        append_call(raw_name, raw_arguments)

    for self_match in SELF_CLOSING_TAG_PATTERN.finditer(response_text):
        attrs = html.unescape(self_match.group("attrs") or "")
        raw_name = _extract_name_from_attrs(attrs)
        raw_arguments = _extract_args_from_attrs(attrs)
        append_call(raw_name, raw_arguments)

    for chinese_match in CHINESE_TOOL_CALL_PATTERN.finditer(response_text):
        append_call(chinese_match.group("name"), chinese_match.group("args"))

    return parsed_calls


def extract_tool_calls_from_text(
    response_text: str,
    tool_definitions: list[dict[str, Any]],
) -> list[ToolCall]:
    xml_tool_calls = parse_xml_tool_calls(response_text, tool_definitions)
    if xml_tool_calls:
        return xml_tool_calls

    allowed_names = {tool["name"] for tool in tool_definitions}
    if not response_text.strip() or not allowed_names:
        return []

    for candidate in _iter_tool_call_json_candidates(response_text):
        parsed_calls: list[ToolCall] = []

        for raw_call in candidate.get("tool_calls", []):
            if not isinstance(raw_call, dict):
                continue

            function_data = raw_call.get("function", {})
            if not isinstance(function_data, dict):
                function_data = {}

            tool_name = (
                str(raw_call.get("name", "")).strip()
                or str(function_data.get("name", "")).strip()
            )
            tool_name = _resolve_tool_name(tool_name, allowed_names)
            if tool_name not in allowed_names:
                continue

            raw_arguments = (
                raw_call.get("arguments")
                if "arguments" in raw_call
                else raw_call.get("input")
            )
            if raw_arguments is None:
                raw_arguments = function_data.get("arguments", {})

            arguments = _normalize_arguments_dict(raw_arguments)

            call_id = (
                str(raw_call.get("id", "")).strip()
                or str(raw_call.get("tool_call_id", "")).strip()
                or f"call_{uuid.uuid4().hex[:24]}"
            )

            parsed_calls.append(
                ToolCall(name=tool_name, arguments=arguments, call_id=call_id)
            )

        if parsed_calls:
            return parsed_calls

    return []


class XmlToolCallStreamParser:
    def __init__(self, tool_definitions: list[dict[str, Any]]):
        self.tool_definitions = tool_definitions
        self.pending_buffer = ""
        self.xml_mode = False
        self.xml_prefix = ""
        self.xml_buffer = ""

    def feed(self, new_content: str) -> tuple[list[str], list[ToolCall]]:
        if not new_content:
            return [], []

        text_chunks: list[str] = []
        parsed_tool_calls: list[ToolCall] = []

        buffer = self.pending_buffer + new_content
        self.pending_buffer = ""

        while buffer:
            if self.xml_mode:
                self.xml_buffer += buffer
                buffer = ""

                suffix = TOOL_CALL_SUFFIX_BY_PREFIX[self.xml_prefix]
                suffix_index = self.xml_buffer.find(suffix)
                if suffix_index == -1:
                    break

                block_end = suffix_index + len(suffix)
                complete_block = self.xml_buffer[:block_end]
                buffer = self.xml_buffer[block_end:]

                tool_calls = parse_xml_tool_calls(complete_block, self.tool_definitions)
                if tool_calls:
                    parsed_tool_calls.extend(tool_calls)
                else:
                    text_chunks.append(complete_block)

                self.xml_mode = False
                self.xml_prefix = ""
                self.xml_buffer = ""
                continue

            prefix_index, prefix = _find_first_tool_call_prefix(buffer)
            if prefix is not None:
                # Guard: if a shorter prefix matched partway through the
                # buffer, check whether the buffer so far could still be the
                # beginning of a *longer* prefix.  If yes, keep buffering
                # instead of locking in the shorter match.
                if prefix_index > 0:
                    defer_match = False
                    for longer_prefix in TOOL_CALL_PREFIXES:
                        if len(longer_prefix) <= len(prefix):
                            continue
                        check_len = min(len(buffer), len(longer_prefix))
                        if buffer[:check_len] == longer_prefix[:check_len] and len(buffer) < len(longer_prefix):
                            defer_match = True
                            break
                    if defer_match:
                        self.pending_buffer = buffer
                        buffer = ""
                        continue

                if prefix_index > 0:
                    text_chunks.append(buffer[:prefix_index])

                self.xml_mode = True
                self.xml_prefix = prefix
                self.xml_buffer = ""
                buffer = buffer[prefix_index:]
                continue

            pending_fragment = _find_pending_tool_call_prefix_fragment(buffer)
            if pending_fragment:
                emit_text = buffer[:-len(pending_fragment)]
                if emit_text:
                    text_chunks.append(emit_text)
                self.pending_buffer = pending_fragment
                buffer = ""
                continue

            text_chunks.append(buffer)
            buffer = ""

        return text_chunks, parsed_tool_calls

    def finish(self) -> tuple[list[str], list[ToolCall]]:
        text_chunks: list[str] = []
        parsed_tool_calls: list[ToolCall] = []

        if self.pending_buffer:
            text_chunks.append(self.pending_buffer)
            self.pending_buffer = ""

        if self.xml_buffer:
            tool_calls = parse_xml_tool_calls(self.xml_buffer, self.tool_definitions)
            if tool_calls:
                parsed_tool_calls.extend(tool_calls)
            else:
                text_chunks.append(self.xml_buffer)

        self.xml_mode = False
        self.xml_prefix = ""
        self.xml_buffer = ""

        return text_chunks, parsed_tool_calls