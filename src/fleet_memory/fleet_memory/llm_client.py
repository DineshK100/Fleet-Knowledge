"""Minimal client for OpenAI-compatible chat-completion endpoints.

Works with local servers such as **Ollama** (``http://localhost:11434/v1``),
**vLLM** (``http://localhost:8000/v1``) or LM Studio, and with hosted
OpenAI-compatible APIs. Only the Python standard library is used so the ROS
nodes have no extra dependencies.

Shared by the VLM event writer (images + text) and the LLM policy adapter
(text only).
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import List, Optional, Sequence


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, base_url: str = 'http://localhost:11434/v1', model: str = 'qwen2.5vl:7b',
                 api_key_env: str = 'OPENAI_API_KEY', timeout_s: float = 60.0,
                 temperature: float = 0.1, max_tokens: int = 400, json_mode: bool = True):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = os.environ.get(api_key_env, '') if api_key_env else ''
        self.timeout_s = float(timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.json_mode = bool(json_mode)

    def chat(self, system: str, user_text: str, images_jpeg: Sequence[bytes] = ()) -> str:
        """Send one chat turn (optionally with JPEG images) and return the reply text."""
        content: List[dict] = [{'type': 'text', 'text': user_text}]
        for img in images_jpeg:
            b64 = base64.b64encode(img).decode('ascii')
            content.append({'type': 'image_url',
                            'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': content if images_jpeg else user_text},
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
        }
        if self.json_mode:
            payload['response_format'] = {'type': 'json_object'}
        try:
            return self._post(payload)
        except LLMError as exc:
            # Some servers reject response_format; retry once without it.
            if self.json_mode and 'response_format' in str(exc):
                payload.pop('response_format', None)
                return self._post(payload)
            raise

    def _post(self, payload: dict) -> str:
        req = urllib.request.Request(
            self.base_url + '/chat/completions',
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json',
                     **({'Authorization': f'Bearer {self.api_key}'} if self.api_key else {})},
            method='POST')
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', 'replace')[:500]
            raise LLMError(f'HTTP {exc.code}: {detail}') from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LLMError(f'connection failed: {exc}') from exc
        try:
            return body['choices'][0]['message']['content'] or ''
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f'unexpected response: {str(body)[:300]}') from exc


_FENCE_RE = re.compile(r'```(?:json)?\s*(.*?)```', re.S)


def extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from a model reply (handles code fences / chatter)."""
    if not text:
        return None
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find('{')
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find('{', start + 1)
    return None
