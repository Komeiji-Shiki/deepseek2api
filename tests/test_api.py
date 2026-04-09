"""快速诊断 DeepSeek API 连通性"""
import json
from curl_cffi import requests

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
email = account["email"]
password = account["password"]

print(f"=== 测试账号: {email} ===")

# Step 1: 登录
print("\n[Step 1] 登录...")
login_resp = requests.post(
    f"https://{DEEPSEEK_HOST}/api/v0/users/login",
    headers=BASE_HEADERS,
    json={"email": email, "password": password, "device_id": "deepseek_to_api", "os": "android"},
    timeout=30,
    impersonate="safari15_3",
)
login_data = login_resp.json()
print(f"  status_code: {login_resp.status_code}")
print(f"  code: {login_data.get('code')}")
print(f"  msg: {login_data.get('msg')}")

token = login_data.get("data", {}).get("biz_data", {}).get("user", {}).get("token", "")
if not token:
    print("  登录失败，拿不到 token!")
    exit(1)
print(f"  token: {token[:12]}...")

auth_headers = {**BASE_HEADERS, "authorization": f"Bearer {token}"}

# Step 2: 创建会话
print("\n[Step 2] 创建会话...")
session_resp = requests.post(
    f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create",
    headers=auth_headers,
    json={"agent": "chat"},
    timeout=30,
    impersonate="safari15_3",
)
session_data = session_resp.json()
print(f"  status_code: {session_resp.status_code}")
print(f"  完整响应: {json.dumps(session_data, ensure_ascii=False, indent=2)[:1000]}")

# Step 3: 尝试不同的请求参数
print("\n[Step 3] 尝试带额外参数创建会话...")
session_resp2 = requests.post(
    f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create",
    headers=auth_headers,
    json={"agent": "chat", "device_id": "deepseek_to_api", "os": "android"},
    timeout=30,
    impersonate="safari15_3",
)
session_data2 = session_resp2.json()
print(f"  status_code: {session_resp2.status_code}")
print(f"  完整响应: {json.dumps(session_data2, ensure_ascii=False, indent=2)[:1000]}")

# Step 4: 尝试 web 端 User-Agent
print("\n[Step 4] 尝试 web 端 UA...")
web_headers = {
    "Host": DEEPSEEK_HOST,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Encoding": "gzip",
    "Content-Type": "application/json",
    "authorization": f"Bearer {token}",
}
session_resp3 = requests.post(
    f"https://{DEEPSEEK_HOST}/api/v0/chat_session/create",
    headers=web_headers,
    json={"agent": "chat"},
    timeout=30,
    impersonate="chrome134",
)
session_data3 = session_resp3.json()
print(f"  status_code: {session_resp3.status_code}")
print(f"  完整响应: {json.dumps(session_data3, ensure_ascii=False, indent=2)[:1000]}")