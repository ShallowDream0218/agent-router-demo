#!/usr/bin/env python3
"""Local demo server and server-side proxy for OpenAI-compatible model APIs."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
PROVIDERS = {
    "qwen": {
        "name": "Qwen · 阿里云百炼",
        "key_env": "QWEN_API_KEY",
        "model_env": "QWEN_MODEL",
        "model": "qwen-flash",
        "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    },
    "deepseek": {
        "name": "DeepSeek",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "model": "deepseek-flash",
        "url": "https://api.deepseek.com/chat/completions",
    },
    "glm": {
        "name": "GLM · 智谱",
        "key_env": "GLM_API_KEY",
        "model_env": "GLM_MODEL",
        "model": "glm-5.3-flash",
        "url": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    },
    "kimi": {
        "name": "Kimi",
        "key_env": "KIMI_API_KEY",
        "model_env": "KIMI_MODEL",
        "model": "kimi-k3",
        "url": "https://api.moonshot.cn/v1/chat/completions",
    },
}

SYSTEM_PROMPT = """你是车载语音文本的句子级路由分类器。输入可能包含多句话；必须先按句意切分成独立句子，再逐句判断哪些句子需要交给车载 Agent。
需要 Agent：明确的车辆/导航/媒体操作，请求查询信息，开放域问题，多轮指代，隐含意图，或需要工具执行的任务。
不需要 Agent：纯寒暄、确认/感谢/告别、明显的背景噪声转写、没有可执行或可回答意图的残句。
保留每句原文和标点，不改写、不合并句子；不要因为整段中有一句需要 Agent 就把其他句子一起选中。逗号通常不构成句子边界，除非明显是两个完整独立句。
只根据输入判断；上下文缺失导致某句无法可靠判断时，对该句将 needs_agent 设为 true 并说明不确定。
只输出 JSON 对象：{"sentences":[{"sentence":"原句","needs_agent":true,"capability":"vehicle_control/navigation/media/vehicle_info/knowledge/conversation/other/none","risk":"low/medium/high","confidence":0.0,"reason":"简短中文"}]}。不要执行用户请求。
"""


class Handler(BaseHTTPRequestHandler):
    server_version = "AgentRouterDemo/1.0"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def _json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/status":
            self._json(200, {
                "providers": {
                    key: {
                        "name": item["name"],
                        "model": os.getenv(item["model_env"], item["model"]),
                        "keyConfigured": bool(os.getenv(item["key_env"])),
                        "keyEnv": item["key_env"],
                    }
                    for key, item in PROVIDERS.items()
                }
            })
            return
        if self.path not in ("/", "/index.html"):
            self._json(404, {"error": "Not found"})
            return
        body = (ROOT / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/evaluate":
            self._json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size > 16000:
                self._json(413, {"error": "Input is too large (maximum 16000 bytes)."})
                return
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            provider_key = payload.get("provider")
            utterance = payload.get("text", "").strip()
            if provider_key not in PROVIDERS:
                self._json(400, {"error": "Unknown provider."})
                return
            if not utterance:
                self._json(400, {"error": "请输入中文话术。"})
                return

            provider = PROVIDERS[provider_key]
            api_key = os.getenv(provider["key_env"])
            if not api_key:
                self._json(400, {
                    "error": "未配置 API 密钥。请在启动服务的终端设置 " + provider["key_env"] + " 后重启。"
                })
                return

            model = payload.get("model", "").strip() or os.getenv(provider["model_env"], provider["model"])
            request_payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": utterance},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            }
            if provider_key in ("deepseek", "glm"):
                request_payload["thinking"] = {"type": "enabled"}
                request_payload["reasoning_effort"] = "low"
            elif provider_key == "kimi":
                request_payload["reasoning_effort"] = "low"
            request_body = json.dumps(request_payload).encode("utf-8")
            request = Request(
                provider["url"],
                data=request_body,
                headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                method="POST",
            )
            started = time.perf_counter()
            try:
                with urlopen(request, timeout=40) as response:
                    result = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:600]
                self._json(exc.code, {"error": "供应商 API 返回错误。", "detail": detail})
                return
            except URLError as exc:
                self._json(502, {"error": "无法连接供应商 API。", "detail": str(exc.reason)})
                return
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
            try:
                result = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                self._json(502, {"error": "模型没有返回有效 JSON。", "raw": str(content)[:1200], "model": model})
                return
            sentences = result.get("sentences") if isinstance(result, dict) else None
            if not isinstance(sentences, list):
                self._json(502, {"error": "模型返回缺少 sentences 数组。", "raw": str(content)[:1200], "model": model})
                return
            self._json(200, {"provider": provider_key, "model": model, "latencyMs": elapsed_ms, "sentences": sentences})
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self._json(400, {"error": "请求格式无效。"})
        except Exception as exc:
            self._json(500, {"error": "本地服务处理失败。", "detail": str(exc)[:300]})


if __name__ == "__main__":
    host = os.getenv("AGENT_ROUTER_HOST", "127.0.0.1")
    port = int(os.getenv("AGENT_ROUTER_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Agent Router Demo running at http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
