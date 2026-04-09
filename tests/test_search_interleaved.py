"""测试 DeepSeek 交错思考搜索模式 - 观察多次调用工具的返回格式"""
import json
import base64
import struct
import ctypes
import glob
import os
import sys
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

def login_and_get_token(account):
    """登录获取 token"""
    print(f"[登录] 使用账号: {account.get('email', account.get('mobile', 'unknown'))}")
    login_resp = requests.post(
        f"https://{DEEPSEEK_HOST}/api/v0/users/login",
        headers=BASE_HEADERS,
        json={"email": account["email"], "password": account["password"], "device_id": "deepseek_to_api", "os": "android"},
        timeout=30, impersonate="safari15_3",
    )
    if login_resp.status_code != 200:
        print(f"[登录失败] status={login_resp.status_code}, body={login_resp.text[:500]}")
        sys.exit(1)
    data = login_resp.json()
    if data.get("code") != 0:
        print(f"[登录失败] code={data.get('code')}, msg={data.get('msg')}")
        sys.exit(1)
    token = data["data"]["biz_data"]["user"]["token"]
    print(f"[登录成功] token={token[:16]}...")
    return token

def create_session(auth_headers):
    """创建会话"""
    print("[创建会话] ...")
    session_resp = requests.post(
        f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create",
        headers=auth_headers, json={"agent": "chat"},
        timeout=30, impersonate="safari15_3",
    )
    if session_resp.status_code != 200:
        print(f"[创建会话失败] status={session_resp.status_code}")
        sys.exit(1)
    data = session_resp.json()
    session_id = data["data"]["biz_data"]["chat_session"]["id"]
    print(f"[会话创建成功] session_id={session_id}")
    return session_id

def get_pow_response(auth_headers):
    """获取 PoW 响应"""
    print("[获取 PoW] ...")
    pow_resp = requests.post(
        f"https://{DEEPSEEK_HOST}/api/v0/chat/create_pow_challenge",
        headers=auth_headers, json={"target_path": "/api/v0/chat/completion"},
        timeout=30, impersonate="safari15_3",
    )
    if pow_resp.status_code != 200:
        print(f"[PoW 失败] status={pow_resp.status_code}")
        sys.exit(1)
    challenge = pow_resp.json()["data"]["biz_data"]["challenge"]
    answer = compute_pow(
        challenge["challenge"], 
        challenge["salt"], 
        challenge.get("difficulty", 144000), 
        challenge.get("expire_at", 1680000000)
    )
    pow_dict = {
        "algorithm": challenge["algorithm"], 
        "challenge": challenge["challenge"], 
        "salt": challenge["salt"], 
        "answer": answer, 
        "signature": challenge["signature"], 
        "target_path": challenge["target_path"]
    }
    pow_response = base64.b64encode(
        json.dumps(pow_dict, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).decode("utf-8").rstrip()
    print(f"[PoW 计算成功] answer={answer}")
    return pow_response

def parse_and_format_chunk(chunk, index):
    """解析并格式化 chunk 以便观察"""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"[事件 #{index}]")
    lines.append(f"{'='*60}")
    
    # 原始 JSON
    lines.append(f"\n[原始 JSON]")
    lines.append(json.dumps(chunk, ensure_ascii=False, indent=2))
    
    # 解析关键字段
    p = chunk.get("p", "")
    v = chunk.get("v")
    status = chunk.get("status")
    
    lines.append(f"\n[字段解析]")
    lines.append(f"  p (path): {p!r}")
    lines.append(f"  v (value) 类型: {type(v).__name__}")
    
    if status:
        lines.append(f"  status: {status!r}")
    
    # 详细解析 v 字段
    if isinstance(v, dict):
        lines.append(f"\n  [v 是字典]")
        for k, val in v.items():
            val_type = type(val).__name__
            val_preview = str(val)[:200] if not isinstance(val, (dict, list)) else f"[{val_type}]"
            lines.append(f"    {k}: {val_type} = {val_preview}")
            
        # 特别关注 response 字段
        if "response" in v:
            resp = v["response"]
            if isinstance(resp, dict):
                lines.append(f"\n    [response 详情]")
                for k, val in resp.items():
                    if k == "fragments" and isinstance(val, list):
                        lines.append(f"      fragments ({len(val)} 个):")
                        for i, frag in enumerate(val):
                            frag_type = frag.get("type", "unknown") if isinstance(frag, dict) else "?"
                            frag_content = str(frag.get("content", ""))[:100] if isinstance(frag, dict) else str(frag)[:100]
                            lines.append(f"        [{i}] type={frag_type}, content={frag_content!r}")
                    else:
                        lines.append(f"      {k}: {str(val)[:200]}")
    
    elif isinstance(v, list):
        lines.append(f"\n  [v 是列表，长度 {len(v)}]")
        for i, item in enumerate(v[:5]):  # 只显示前5个
            item_preview = json.dumps(item, ensure_ascii=False)[:200] if isinstance(item, (dict, list)) else str(item)[:200]
            lines.append(f"    [{i}] {item_preview}")
        if len(v) > 5:
            lines.append(f"    ... 还有 {len(v) - 5} 个元素")
    
    elif isinstance(v, str):
        lines.append(f"\n  [v 是字符串]")
        lines.append(f"    内容: {v[:500]!r}")
        if len(v) > 500:
            lines.append(f"    ... 共 {len(v)} 字符")
    
    else:
        lines.append(f"\n  [v 值]")
        lines.append(f"    {v!r}")
    
    return "\n".join(lines)

def main():
    # 使用第一个账号
    account = CONFIG["accounts"][0]
    token = login_and_get_token(account)
    auth_headers = {**BASE_HEADERS, "authorization": f"Bearer {token}"}
    
    session_id = create_session(auth_headers)
    pow_response = get_pow_response(auth_headers)
    
    # 使用一个需要搜索的提示词 - 结合当前时间的事件
    prompt = "最新的科技新闻有哪些？请搜索并总结。"
    
    comp_headers = {**auth_headers, "x-ds-pow-response": pow_response}
    
    print(f"\n{'#'*60}")
    print(f"# 开始测试交错思考搜索模式")
    print(f"# 提示词: {prompt}")
    print(f"# 配置: thinking_enabled=True, search_enabled=True")
    print(f"{'#'*60}\n")
    
    comp_resp = requests.post(
        f"https://{DEEPSEEK_HOST}/api/v0/chat/completion",
        headers=comp_headers,
        json={
            "chat_session_id": session_id, 
            "parent_message_id": None, 
            "prompt": prompt,
            "ref_file_ids": [], 
            "thinking_enabled": True, 
            "search_enabled": True
        },
        stream=True, 
        impersonate="safari15_3",
    )
    
    print(f"[响应状态] {comp_resp.status_code}")
    
    output_lines = []
    output_lines.append(f"测试时间: {json.dumps(os.path.getctime, default=str) if hasattr(os, 'getctime') else 'N/A'}")
    output_lines.append(f"提示词: {prompt}")
    output_lines.append(f"配置: thinking_enabled=True, search_enabled=True")
    output_lines.append(f"响应状态码: {comp_resp.status_code}")
    output_lines.append("")
    
    event_count = 0
    thinking_events = []
    text_events = []
    search_events = []
    other_events = []
    
    for raw_line in comp_resp.iter_lines():
        line = raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else str(raw_line)
        if not line:
            continue
            
        if not line.startswith("data:"):
            output_lines.append(f"[非数据行] {line[:200]}")
            continue
            
        data_str = line[5:].strip()
        
        if data_str == "[DONE]":
            output_lines.append("\n" + "="*60)
            output_lines.append("[DONE]")
            output_lines.append("="*60)
            break
            
        try:
            chunk = json.loads(data_str)
        except Exception as e:
            output_lines.append(f"\n[解析错误] {e}: {data_str[:200]}")
            continue
        
        event_count += 1
        formatted = parse_and_format_chunk(chunk, event_count)
        output_lines.append(formatted)
        
        # 分类统计
        p = str(chunk.get("p", "")).lower()
        v = chunk.get("v")
        
        # 检测事件类型
        if isinstance(v, dict) and "response" in v:
            resp = v.get("response", {})
            if isinstance(resp, dict) and "fragments" in resp:
                fragments = resp.get("fragments", [])
                if fragments:
                    last_frag = fragments[-1]
                    if isinstance(last_frag, dict):
                        frag_type = str(last_frag.get("type", "")).upper()
                        if frag_type == "THINK":
                            thinking_events.append(event_count)
                        elif frag_type == "RESPONSE":
                            text_events.append(event_count)
                        elif frag_type == "TIP":
                            search_events.append(event_count)
        
        if "search" in p:
            search_events.append(event_count)
    
    comp_resp.close()
    
    # 统计信息
    output_lines.append(f"\n{'='*60}")
    output_lines.append("[统计信息]")
    output_lines.append(f"{'='*60}")
    output_lines.append(f"总事件数: {event_count}")
    output_lines.append(f"思考相关事件: {len(thinking_events)} 个 (索引: {thinking_events[:10]}{'...' if len(thinking_events) > 10 else ''})")
    output_lines.append(f"文本相关事件: {len(text_events)} 个 (索引: {text_events[:10]}{'...' if len(text_events) > 10 else ''})")
    output_lines.append(f"搜索相关事件: {len(search_events)} 个 (索引: {search_events[:10]}{'...' if len(search_events) > 10 else ''})")
    
    # 保存到文件
    output_file = "search_interleaved_output.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    
    print(f"\n[完成] 共捕获 {event_count} 个事件")
    print(f"[保存] 输出已写入 {output_file}")
    print(f"[统计] 思考事件: {len(thinking_events)}, 文本事件: {len(text_events)}, 搜索事件: {len(search_events)}")

if __name__ == "__main__":
    main()