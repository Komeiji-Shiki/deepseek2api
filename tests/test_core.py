import asyncio
import json
import unittest

from fastapi.testclient import TestClient

import app
from account_pool import AccountPool
from tool_calling import (
    XmlToolCallStreamParser,
    build_tool_system_prompt,
    extract_tool_calls_from_text,
    normalize_openai_messages,
    normalize_tool_definitions,
)


class AccountPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_account_pool_never_leases_same_account_twice_concurrently(self):
        pool = AccountPool(
            [
                {"email": "a@example.com", "password": "x"},
                {"email": "b@example.com", "password": "x"},
            ]
        )

        leases = []

        async def worker():
            lease = await pool.acquire()
            if lease is None:
                return None
            leases.append(lease)
            await asyncio.sleep(0.02)
            return lease.identifier

        acquired = [item for item in await asyncio.gather(*[worker() for _ in range(6)]) if item]

        self.assertEqual(len(acquired), 2)
        self.assertEqual(len(set(acquired)), 2)

        for lease in leases:
            await lease.release()

        lease1 = await pool.acquire()
        lease2 = await pool.acquire()
        self.assertIsNotNone(lease1)
        self.assertIsNotNone(lease2)
        self.assertNotEqual(lease1.identifier, lease2.identifier)

        await lease1.release()
        await lease2.release()

    async def test_account_pool_respects_excluded_accounts(self):
        pool = AccountPool(
            [
                {"email": "a@example.com", "password": "x"},
                {"email": "b@example.com", "password": "x"},
            ]
        )

        lease = await pool.acquire({"a@example.com"})
        self.assertIsNotNone(lease)
        self.assertEqual(lease.identifier, "b@example.com")
        await lease.release()


class ToolCallingTests(unittest.TestCase):
    def test_normalize_openai_tools_definition(self):
        tools = normalize_tool_definitions(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather data",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        )

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "get_weather")
        self.assertIn("city", tools[0]["input_schema"]["properties"])

    def test_extract_tool_calls_from_json_text(self):
        tools = normalize_tool_definitions(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather data",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        )

        parsed = extract_tool_calls_from_text(
            '{"tool_calls":[{"name":"get_weather","arguments":{"city":"beijing"}}]}',
            tools,
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].name, "get_weather")
        self.assertEqual(parsed[0].arguments, {"city": "beijing"})

        openai_payload = parsed[0].to_openai_dict()
        self.assertEqual(openai_payload["type"], "function")
        self.assertEqual(openai_payload["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(openai_payload["function"]["arguments"]),
            {"city": "beijing"},
        )

    def test_extract_tool_calls_from_xml_text(self):
        tools = normalize_tool_definitions(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather data",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        )

        parsed = extract_tool_calls_from_text(
            '<<<tool_call>>>\nname: get_weather\narguments: {"city":"beijing"}\n<<</tool_call>>>',
            tools,
        )

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].name, "get_weather")
        self.assertEqual(parsed[0].arguments, {"city": "beijing"})

    def test_normalize_openai_messages_with_tool_history(self):
        messages = normalize_openai_messages(
            [
                {"role": "user", "content": "查天气"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "先思考，再调用工具",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"beijing"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temp":26}',
                },
            ]
        )

        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertTrue(messages[1]["content"].startswith("<think>先思考，再调用工具</think>"))
        self.assertIn("<<<tool_call>>>", messages[1]["content"])
        self.assertIn("name: get_weather", messages[1]["content"])
        self.assertIn('arguments: {"city":"beijing"}', messages[1]["content"])
        self.assertEqual(messages[2]["role"], "user")
        self.assertIn("Tool result for get_weather", messages[2]["content"])
        self.assertIn('{"temp":26}', messages[2]["content"])

    def test_normalize_openai_messages_does_not_replay_old_round_reasoning(self):
        messages = normalize_openai_messages(
            [
                {"role": "user", "content": "查天气"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "旧轮次的思考",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"beijing"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temp":26}',
                },
                {"role": "assistant", "content": "北京 26 度"},
                {"role": "user", "content": "继续聊别的"},
            ]
        )

        self.assertEqual(messages[1]["role"], "assistant")
        self.assertNotIn("<think>旧轮次的思考</think>", messages[1]["content"])
        self.assertIn("<<<tool_call>>>", messages[1]["content"])

    def test_normalize_claude_messages_with_thinking_and_tool_use_history(self):
        messages = app.normalize_claude_messages(
            [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "先思考，再调用工具",
                        },
                        {
                            "type": "tool_use",
                            "name": "get_weather",
                            "input": {"city": "beijing"},
                        },
                    ],
                }
            ]
        )

        self.assertEqual(messages[0]["role"], "assistant")
        self.assertTrue(messages[0]["content"].startswith("<think>先思考，再调用工具</think>"))
        self.assertIn("<<<tool_call>>>", messages[0]["content"])
        self.assertIn("name: get_weather", messages[0]["content"])
        self.assertIn('arguments: {"city":"beijing"}', messages[0]["content"])

    def test_normalize_claude_messages_does_not_replay_old_round_thinking(self):
        messages = app.normalize_claude_messages(
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "查天气"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "旧轮次的思考",
                        },
                        {
                            "type": "tool_use",
                            "name": "get_weather",
                            "input": {"city": "beijing"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": '{"temp":26}',
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "北京 26 度"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "继续聊别的"}],
                },
            ]
        )

        self.assertEqual(messages[1]["role"], "assistant")
        self.assertNotIn("<think>旧轮次的思考</think>", messages[1]["content"])
        self.assertIn("<<<tool_call>>>", messages[1]["content"])

    def test_build_tool_system_prompt_contains_tool_choice_rule(self):
        tools = normalize_tool_definitions(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather data",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )

        prompt = build_tool_system_prompt(tools, tool_choice="required")
        self.assertIn("你必须先至少调用一个工具", prompt)
        self.assertIn("get_weather", prompt)
        self.assertIn("<<<tool_call>>>", prompt)

    def test_xml_tool_call_stream_parser_keeps_plain_text_streaming(self):
        tools = normalize_tool_definitions(
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather data",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
        )

        parser = XmlToolCallStreamParser(tools)

        text_chunks, tool_calls = parser.feed("h")
        self.assertEqual(text_chunks, ["h"])
        self.assertEqual(tool_calls, [])

        text_chunks, tool_calls = parser.feed("ello ")
        self.assertEqual(text_chunks, ["ello "])
        self.assertEqual(tool_calls, [])

        text_chunks, tool_calls = parser.feed(
            "<<<tool_call>>>\nname: get_weather\narguments: "
        )
        self.assertEqual(text_chunks, [])
        self.assertEqual(tool_calls, [])

        text_chunks, tool_calls = parser.feed('{"city":"beijing"}\n<<</tool_call>>>')
        self.assertEqual(text_chunks, [])
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "get_weather")
        self.assertEqual(tool_calls[0].arguments, {"city": "beijing"})


class StreamParsingTests(unittest.IsolatedAsyncioTestCase):
    async def test_collect_completion_output_keeps_thinking_type_when_path_is_missing(self):
        class FakeResponse:
            async def aiter_lines(self, decode_unicode=False, delimiter=None):
                if decode_unicode:
                    raise NotImplementedError("decode_unicode not supported")
                yield 'data: {"p":"response/thinking_content","v":"we"}'.encode("utf-8")
                yield 'data: {"v":"think"}'.encode("utf-8")
                yield 'data: {"p":"response/content","v":"answer"}'.encode("utf-8")
                yield 'data: {"p":"response/status","v":"FINISHED"}'.encode("utf-8")

            async def aclose(self):
                return None

        completion = app.CompletionContext(
            response=FakeResponse(),
            session_id="sess_test",
            prompt="prompt",
            output_model="deepseek-reasoner",
            thinking_enabled=True,
            search_enabled=False,
        )

        reasoning, content = await app.collect_completion_output(completion)
        self.assertEqual(reasoning, "wethink")
        self.assertEqual(content, "answer")


class AppRouteTests(unittest.TestCase):
    def test_chat_completions_non_stream_path_handles_async_line_iteration(self):
        class FakeResponse:
            async def aiter_lines(self, decode_unicode=False, delimiter=None):
                if decode_unicode:
                    raise NotImplementedError("decode_unicode not supported")
                yield 'data: {"p":"response/content","v":"hello"}'.encode("utf-8")
                yield 'data: {"p":"response/status","v":"FINISHED"}'.encode("utf-8")

            async def aclose(self):
                return None

        async def fake_determine_mode_and_token(request):
            return app.AuthContext(use_config_token=False, deepseek_token="test-token")

        async def fake_start_deepseek_completion(
            request,
            auth_ctx,
            *,
            deepseek_model,
            messages,
            output_model,
        ):
            return app.CompletionContext(
                response=FakeResponse(),
                session_id="sess_test",
                prompt="prompt",
                output_model=output_model,
                thinking_enabled=False,
                search_enabled=False,
            )

        async def fake_cleanup_completion(request, auth_ctx, completion=None):
            return None

        original_determine = app.determine_mode_and_token
        original_start = app.start_deepseek_completion
        original_cleanup = app.cleanup_completion

        try:
            app.determine_mode_and_token = fake_determine_mode_and_token
            app.start_deepseek_completion = fake_start_deepseek_completion
            app.cleanup_completion = fake_cleanup_completion

            client = TestClient(app.app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-token"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["choices"][0]["message"]["content"], "hello")
            self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        finally:
            app.determine_mode_and_token = original_determine
            app.start_deepseek_completion = original_start
            app.cleanup_completion = original_cleanup

    def test_chat_completions_non_stream_extracts_xml_tool_call_from_reasoning_content(self):
        class FakeResponse:
            async def aiter_lines(self, decode_unicode=False, delimiter=None):
                if decode_unicode:
                    raise NotImplementedError("decode_unicode not supported")
                yield (
                    'data: {"p":"response/thinking_content","v":"<<<tool_call>>>\\nname: get_weather\\narguments: {\\"city\\":\\"beijing\\"}\\n<<</tool_call>>>"}'
                ).encode("utf-8")
                yield 'data: {"p":"response/status","v":"FINISHED"}'.encode("utf-8")

            async def aclose(self):
                return None

        async def fake_determine_mode_and_token(request):
            return app.AuthContext(use_config_token=False, deepseek_token="test-token")

        async def fake_start_deepseek_completion(
            request,
            auth_ctx,
            *,
            deepseek_model,
            messages,
            output_model,
        ):
            return app.CompletionContext(
                response=FakeResponse(),
                session_id="sess_test",
                prompt="prompt",
                output_model=output_model,
                thinking_enabled=True,
                search_enabled=False,
            )

        async def fake_cleanup_completion(request, auth_ctx, completion=None):
            return None

        original_determine = app.determine_mode_and_token
        original_start = app.start_deepseek_completion
        original_cleanup = app.cleanup_completion

        try:
            app.determine_mode_and_token = fake_determine_mode_and_token
            app.start_deepseek_completion = fake_start_deepseek_completion
            app.cleanup_completion = fake_cleanup_completion

            client = TestClient(app.app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": "9.8-9.11 是多少"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather data",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                    "stream": False,
                },
                headers={"Authorization": "Bearer test-token"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            message = payload["choices"][0]["message"]
            self.assertEqual(payload["choices"][0]["finish_reason"], "tool_calls")
            self.assertIsNone(message["content"])
            self.assertEqual(message["tool_calls"][0]["function"]["name"], "get_weather")
            self.assertEqual(
                json.loads(message["tool_calls"][0]["function"]["arguments"]),
                {"city": "beijing"},
            )
        finally:
            app.determine_mode_and_token = original_determine
            app.start_deepseek_completion = original_start
            app.cleanup_completion = original_cleanup

    def test_chat_completions_stream_extracts_xml_tool_call_from_reasoning_content(self):
        class FakeResponse:
            async def aiter_lines(self, decode_unicode=False, delimiter=None):
                if decode_unicode:
                    raise NotImplementedError("decode_unicode not supported")
                yield (
                    'data: {"p":"response/thinking_content","v":"<<<tool_call>>>\\nname: get_weather\\narguments: {\\"city\\":\\"beijing\\"}\\n<<</tool_call>>>"}'
                ).encode("utf-8")
                yield 'data: {"p":"response/status","v":"FINISHED"}'.encode("utf-8")

            async def aclose(self):
                return None

        async def fake_determine_mode_and_token(request):
            return app.AuthContext(use_config_token=False, deepseek_token="test-token")

        async def fake_start_deepseek_completion(
            request,
            auth_ctx,
            *,
            deepseek_model,
            messages,
            output_model,
        ):
            return app.CompletionContext(
                response=FakeResponse(),
                session_id="sess_test",
                prompt="prompt",
                output_model=output_model,
                thinking_enabled=True,
                search_enabled=False,
            )

        async def fake_cleanup_completion(request, auth_ctx, completion=None):
            return None

        original_determine = app.determine_mode_and_token
        original_start = app.start_deepseek_completion
        original_cleanup = app.cleanup_completion

        try:
            app.determine_mode_and_token = fake_determine_mode_and_token
            app.start_deepseek_completion = fake_start_deepseek_completion
            app.cleanup_completion = fake_cleanup_completion

            client = TestClient(app.app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-reasoner",
                    "messages": [{"role": "user", "content": "9.8-9.11 是多少"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "description": "Get weather data",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"city": {"type": "string"}},
                                    "required": ["city"],
                                },
                            },
                        }
                    ],
                    "stream": True,
                },
                headers={"Authorization": "Bearer test-token"},
            )

            self.assertEqual(response.status_code, 200)
            body = response.text
            self.assertIn('"tool_calls"', body)
            self.assertIn('"finish_reason": "tool_calls"', body)
            self.assertNotIn("<<<tool_call>>>", body)
        finally:
            app.determine_mode_and_token = original_determine
            app.start_deepseek_completion = original_start
            app.cleanup_completion = original_cleanup


if __name__ == "__main__":
    unittest.main()