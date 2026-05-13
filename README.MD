# DeepSeek2API

支持高速流式输出、多轮对话、深度思考、联网搜索、专家模式、工具调用（Function Calling）、Vision 视觉识别、文件上传，多路账号自动轮换。

与 OpenAI / Claude 接口完全兼容。

## 目录

* [免责声明](#免责声明)
* [新特性](#新特性)
* [接入准备](#接入准备)
  * [多账号接入](#多账号接入)
* [Docker部署](#Docker部署)
* [Docker-compose部署](#Docker-compose部署)
* [原生部署](#原生部署)
* [接口列表](#接口列表)
  * [模型列表](#模型列表)
  * [对话补全](#对话补全)
  * [工具调用](#工具调用)
  * [Vision 视觉](#Vision-视觉)
  * [文件上传](#文件上传)
  * [Claude API](#claude-api)
* [注意事项](#注意事项)
  * [Nginx反代优化](#Nginx反代优化)
* [Star History](#star-history)
* [鸣谢](#鸣谢)

## 免责声明

**逆向API是不稳定的，建议前往DeepSeek官方 https://platform.deepseek.com/ 付费使用API，避免封禁的风险。**

**本组织和个人不接受任何资金捐助和交易，此项目是纯粹研究交流学习性质！**

> ⚠️ 近期 DeepSeek 加强了对第三方客户端的检测。本项目已针对 `device_id` 指纹做出防护（每个账号自动生成独立随机 device_id），并与其他同类项目（如 ds2api-main）在指纹上产生差异化。但封杀策略随时可能调整，使用需自担风险。

## 新特性

- 🛡️ **device_id 随机化**：每个账号启动时自动生成 64 字节随机 device_id，与官方客户端格式一致，避免指纹被检测
- 🖼️ **Vision 视觉识别**：支持 OpenAI 格式的 `image_url` 图片输入（data: URL），自动上传至 DeepSeek 并关联到对话
- 📁 **文件上传**：`/v1/files` 端点支持 multipart 文件上传，兼容 OpenAI Files API
- 🔄 **Prompt 格式对齐**：与 DeepSeek 官方 prompt 标记格式保持一致

## 接入准备

一个或多个 DeepSeek 账号

### 多账号接入

目前同个账号同时只能有*一路*输出，你可以通过提供多个账号并使用。

每次请求服务会从中挑选一个。

## Docker部署

请准备一台具有公网IP的服务器并将5001端口开放。

配置 config.json

在 `deepseek` 目录下，创建 config.json文件
```json
{
  "keys": [
    "key1",
    "key2"
  ],
  "accounts": [
    {
      "email": "example1@example.com",
      "password": "password1",
      "token": "",
      "device_id": ""
    },
    {
      "email": "example2@example.com",
      "password": "password2",
      "token": "",
      "device_id": ""
    },
    {
      "mobile": "12345678901",
      "password": "password3",
      "token": "",
      "device_id": ""
    }
  ]
}
```
 * `keys` - 你的 API 鉴权密钥
 * `accounts` - DeepSeek 账号列表，支持多个账号轮换
 * `device_id` - 留空即可，**服务启动时会自动为每个账号生成独立的随机 device_id 并持久化**，无需手动填写

拉取镜像并启动服务。

```shell
docker run -d -p 5001:5001 -v "$(pwd)/config.json:/app/config.json" --name deepseek2api ghcr.io/komeiji-shiki/deepseek2api:latest
```

查看服务实时日志

```shell
docker logs -f deepseek2api
```

重启服务

```shell
docker restart deepseek2api
```

停止服务

```shell
docker stop deepseek2api
```

## Docker-compose部署

拉取该项目
```shell
git clone https://github.com/Komeiji-Shiki/deepseek2api.git
cd deepseek2api
```
配置 config.json

在当前目录下，修改 config.json 文件
 * `keys` - 你的 API 鉴权密钥
 * `accounts` - DeepSeek 账号列表
 * `device_id` - 留空即可，自动生成

启动服务

```shell
docker-compose up -d
```

查看服务实时日志

```shell
docker logs -f deepseek2api
```

重启服务

```shell
docker restart deepseek2api
```

停止服务

```shell
docker stop deepseek2api
```

## 原生部署

请准备一台具有公网IP的服务器并将5001端口开放。

请先安装好 Python 环境并且配置好环境变量，确认 python 命令可用。

安装依赖

```shell
git clone https://github.com/Komeiji-Shiki/deepseek2api.git
cd deepseek2api
pip install -r requirements.txt
```

初次运行前，先下载 WASM（PoW 计算所需）：

```shell
python fetch_wasm.py
```

配置 config.json

在当前目录下，修改 config.json 文件
 * `keys` - 你的 API 鉴权密钥
 * `accounts` - DeepSeek 账号列表
 * `device_id` - 留空即可，启动时自动生成随机 device_id 并持久化到配置文件

启动服务

```shell
python app.py
```

使用 nohup 启动

```shell
nohup uvicorn app:app --host 0.0.0.0 --port 5001 > uvicorn.log 2>&1 &
```

## 接口列表

目前支持与 OpenAI 兼容的接口，可自行使用与 OpenAI 或其他兼容的客户端接入。

### 模型列表

获取模型列表接口

**GET /v1/models**

响应数据:
```json
{
  "data": [
    {
      "created": 1677610602,
      "id": "deepseek-chat",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-reasoner",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-chat-search",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-reasoner-search",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-expert",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-expert-reasoner",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-expert-search",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    },
    {
      "created": 1677610602,
      "id": "deepseek-expert-reasoner-search",
      "object": "model",
      "owned_by": "deepseek",
      "permission": []
    }
  ],
  "object": "list"
}
```

**支持的模型列表：**

**快速模式（默认）：**

| 模型 ID | 说明 |
|---------|------|
| `deepseek-chat` / `deepseek-v3` | 标准对话模型 |
| `deepseek-reasoner` / `deepseek-r1` | 思考模型 |
| `deepseek-chat-search` / `deepseek-v3-search` | 标准模型 + 联网搜索 |
| `deepseek-reasoner-search` / `deepseek-r1-search` | 深度思考 + 联网搜索 |

**专家模式：**

| 模型 ID | 说明 |
|---------|------|
| `deepseek-expert` | 专家模式对话 |
| `deepseek-expert-reasoner` | 专家模式 + 深度思考 |
| `deepseek-expert-search` | 专家模式 + 联网搜索 |
| `deepseek-expert-reasoner-search` | 专家模式 + 深度思考 + 联网搜索 |

### 对话补全

对话补全接口，与 OpenAI 的 [chat-completions-api](https://platform.openai.com/docs/guides/text-generation/chat-completions-api) 兼容。

**POST /v1/chat/completions**

header 需要设置 Authorization 头部：

```
Authorization: Bearer [keys 中的任意值 或 DeepSeek Token]
```

请求数据：
```json
{
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "user",
            "content": "你是谁？"
        }
    ],
    "stream": false
}
```

响应数据：
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "您好！我是由深度求索公司开发的智能助手。",
        "reasoning_content": "嗯，用户问我是谁...",
        "role": "assistant"
      }
    }
  ],
  "created": 1738598554,
  "id": "bc223da6-f091-4687-9f59-b9f2a917bd49",
  "model": "deepseek-chat",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 37,
    "prompt_tokens": 4,
    "total_tokens": 41,
    "completion_tokens_details": {
      "reasoning_tokens": 12
    }
  }
}
```

### 工具调用

支持 OpenAI 格式的 Function Calling / Tools，模型会在需要时自动调用工具。

请求示例：
```json
{
    "model": "deepseek-reasoner",
    "messages": [{"role": "user", "content": "北京天气如何？"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "获取指定城市的天气信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "城市名称"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
}
```

工具调用结果会以 `tool_calls` 形式返回，后续对话需要传递 `tool` 角色的消息来返回工具执行结果。

### Vision 视觉

支持 OpenAI 格式的 `image_url` 图片输入。服务会自动将图片上传至 DeepSeek 并关联到对话上下文。

请求示例：
```json
{
    "model": "deepseek-chat",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgo..."
                    }
                }
            ]
        }
    ]
}
```

> 支持 data: URL 格式（base64），不支持外部 HTTP URL。

### 文件上传

**POST /v1/files**

支持 multipart/form-data 文件上传，兼容 OpenAI Files API。

```shell
curl -X POST http://localhost:5001/v1/files \
  -H "Authorization: Bearer key1" \
  -F "file=@document.pdf" \
  -F "purpose=user_data"
```

响应：
```json
{
  "id": "file_xxx",
  "object": "file",
  "bytes": 1024,
  "created_at": 1700000000,
  "filename": "document.pdf",
  "purpose": "user_data",
  "status": "uploaded"
}
```

### Claude API

除 OpenAI 兼容接口外，还提供 Claude API 兼容接口：

- **GET** `/anthropic/v1/models` - 获取模型列表
- **POST** `/anthropic/v1/messages` - 对话补全
- **POST** `/anthropic/v1/messages/count_tokens` - Token 计数

支持 Claude 格式的 `tools`、`thinking`、`tool_use` 等功能，同样支持 Vision 视觉输入。

## 注意事项

### Nginx反代优化

如果您正在使用Nginx反向代理deepseek2api，请添加以下配置项优化流的输出效果。

```nginx
proxy_buffering off;
chunked_transfer_encoding on;
tcp_nopush on;
tcp_nodelay on;
keepalive_timeout 120;
```

### Token统计

符合 OpenAI 接口规范，包含 `reasoning_tokens` 细分统计：

```json
"usage": {
  "completion_tokens": 37,
  "prompt_tokens": 4,
  "total_tokens": 41,
  "completion_tokens_details": {
    "reasoning_tokens": 12
  }
}
```

### device_id 指纹

为防止被 DeepSeek 根据设备指纹封杀，服务启动时会自动为每个配置的账号生成独立的随机 `device_id`（64 字节 Base64 编码，与官方客户端格式一致）。`device_id` 会自动写入 `config.json`，重启后保持不变。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Komeiji-Shiki/deepseek2api&type=Date)](https://star-history.com/#Komeiji-Shiki/deepseek2api&Date)

## 鸣谢

本项目部分代码参考了 [deepseek2api](https://github.com/iidamie/deepseek2api) 和 [ds2api](https://github.com/CJackHwang/ds2api)，在此表示感谢。
