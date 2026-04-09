"""测试搜索工具调用格式转换"""
import json


def _format_search_tool_call(fragment: dict) -> str | None:
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
        "time_range": "day",  # 默认值
        "max_results": 10,    # 默认值
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
    
    return f"「调用工具: tavily-mcp_tavily_search 输入内容: {json.dumps(tool_input, ensure_ascii=False)}」"


# 测试数据
test_fragments = [
    {
        "id": 3,
        "type": "TOOL_SEARCH",
        "status": "WIP",
        "content": None,
        "queries": [
            {"query": "2026年4月 科技新闻 最新"}
        ],
        "results": [],
        "stage_id": 1
    },
    {
        "id": 4,
        "type": "TOOL_SEARCH",
        "status": "WIP",
        "content": None,
        "queries": [
            {"query": "2026年4月9日 科技新闻"}
        ],
        "results": [],
        "stage_id": 1
    },
    {
        "id": 5,
        "type": "TOOL_SEARCH",
        "status": "WIP",
        "content": None,
        "queries": [
            {"query": "人工智能 最新进展"}
        ],
        "results": [],
        "stage_id": 1
    },
    # 非搜索类型的 fragment
    {
        "id": 10,
        "type": "THINK",
        "content": "用户想了解最新科技新闻",
        "elapsed_secs": None,
        "references": [],
        "stage_id": 1
    },
    # 没有 queries 的搜索
    {
        "id": 11,
        "type": "TOOL_SEARCH",
        "status": "WIP",
        "content": None,
        "queries": [],
        "results": [],
        "stage_id": 1
    }
]

print("=" * 60)
print("测试搜索工具调用格式转换")
print("=" * 60)

for i, fragment in enumerate(test_fragments, 1):
    print(f"\n[测试 #{i}]")
    print(f"Fragment type: {fragment.get('type')}")
    print(f"Queries: {fragment.get('queries', [])}")
    
    result = _format_search_tool_call(fragment)
    
    if result:
        print(f"输出: {result}")
    else:
        print("输出: None (不转换)")
    
    print("-" * 40)

print("\n" + "=" * 60)
print("测试完成!")
print("=" * 60)