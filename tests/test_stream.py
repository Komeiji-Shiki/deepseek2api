"""测试完整流式响应 - 捕获全部事件"""
import json
import base64
import struct
import ctypes
import glob
import os
from curl_cffi import requests
from wasmtime import Linker, Module, Store

CONFIG = json.load(open("config.json", encoding="utf-8"))
DEEPSEEK_HOST = "chat.deepseek.com"

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

account = CONFIG["accounts"][0]
login_resp = requests.post(
    f"https://{DEEPSEEK_HOST}/api/v0/users/login",
    headers=BASE_HEADERS,
    json={"email": account["email"], "password": account["password"], "device_id": "deepseek_to_api", "os": "android"},
    timeout=30, impersonate="safari15_3",
)
token = login_resp.json()["data"]["biz_data"]["user"]["token"]
auth_headers = {**BASE_HEADERS, "authorization": f"Bearer {token}"}

session_resp = requests.post(
    f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create",
    headers=auth_headers, json={"agent": "chat"},
    timeout=30, impersonate="safari15_3",
)
session_id = session_resp.json()["data"]["biz_data"]["chat_session"]["id"]

pow_resp = requests.post(
    f"https://{DEEPSEEK_HOST}/api/v0/chat/create_pow_challenge",
    headers=auth_headers, json={"target_path": "/api/v0/chat/completion"},
    timeout=30, impersonate="safari15_3",
)
challenge = pow_resp.json()["data"]["biz_data"]["challenge"]

def compute_pow(challenge_str, salt, difficulty, expire_at):
    prefix = f"{salt}_{expire_at}_"
    store = Store()
    linker = Linker(store.engine)
    wasm_files = sorted(glob.glob("sha3_wasm_bg.*.wasm"), key=lambda f: os.path.getmtime(f), reverse=True)
    with open(wasm_files[0], "rb") as f:
        wasm_bytes = f.read()
    module = Module(store.engine, wasm_bytes)
    instance = linker.instantiate(store, module)
    exports = instance.exports(store)
    memory = exports["memory"]
    add_to_stack = exports["__wbindgen_add_to_stack_pointer"]
    alloc = exports["__wbindgen_export_0"]
    wasm_solve = exports["wasm_solve"]
    def write_memory(offset, data):
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        ctypes.memmove(base_addr + offset, data, len(data))
    def read_memory(offset, size):
        base_addr = ctypes.cast(memory.data_ptr(store), ctypes.c_void_p).value
        return ctypes.string_at(base_addr + offset, size)
    def encode_string(text):
        d = text.encode("utf-8")
        ptr_value = alloc(store, len(d), 1)
        ptr = int(ptr_value.value) if hasattr(ptr_value, "value") else int(ptr_value)
        write_memory(ptr, d)
        return ptr, len(d)
    retptr = add_to_stack(store, -16)
    ptr_c, len_c = encode_string(challenge_str)
    ptr_p, len_p = encode_string(prefix)
    wasm_solve(store, retptr, ptr_c, len_c, ptr_p, len_p, float(difficulty))
    status = struct.unpack("<i", read_memory(retptr, 4))[0]
    value = struct.unpack("<d", read_memory(retptr + 8, 8))[0]
    add_to_stack(store, 16)
    return int(value) if status != 0 else None

answer = compute_pow(challenge["challenge"], challenge["salt"], challenge.get("difficulty", 144000), challenge.get("expire_at", 1680000000))
pow_dict = {"algorithm": challenge["algorithm"], "challenge": challenge["challenge"], "salt": challenge["salt"], "answer": answer, "signature": challenge["signature"], "target_path": challenge["target_path"]}
pow_response = base64.b64encode(json.dumps(pow_dict, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).decode("utf-8").rstrip()

comp_headers = {**auth_headers, "x-ds-pow-response": pow_response}
comp_resp = requests.post(
    f"https://{DEEPSEEK_HOST}/api/v0/chat/completion",
    headers=comp_headers,
    json={"chat_session_id": session_id, "parent_message_id": None, "prompt": "1+1=?",
          "ref_file_ids": [], "thinking_enabled": True, "search_enabled": False},
    stream=True, impersonate="safari15_3",
)

output_lines = [f"status_code: {comp_resp.status_code}\n"]
count = 0
for raw_line in comp_resp.iter_lines():
    line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
    if not line or not line.startswith("data:"):
        continue
    data_str = line[5:].strip()
    if data_str == "[DONE]":
        output_lines.append("[DONE]")
        break
    try:
        chunk = json.loads(data_str)
    except:
        output_lines.append(f"[PARSE ERROR] {data_str[:200]}")
        continue
    output_lines.append(json.dumps(chunk, ensure_ascii=False))
    count += 1

comp_resp.close()
output_lines.append(f"\n总共 {count} 个事件")

with open("stream_output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"完成! 写入 {count} 个事件到 stream_output.txt")