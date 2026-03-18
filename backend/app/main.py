import os
import re
from venv import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv
from typing import Any, Dict, List, Literal, TypedDict
import uuid
from urllib.parse import urlparse
# SSE
import json
from typing import AsyncIterator
from fastapi.responses import StreamingResponse
from fastapi import HTTPException, UploadFile, File, Form

from app.utils import rag as rag_utils

load_dotenv()

app = FastAPI()

Role = Literal["system", "user", "assistant"]
SYSTEM_PROMPT = "You are a helpful assistant."
MAX_HISTORY_MESSAGES = 20

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
    对前端请求和后端响应进行类型验证和自动序列化。
    BaseModel 可以保证请求体中必须包含 message 字段，类型为 str。
    自动生成 OpenAPI 文档的请求示例。
"""
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

"""
    定义消息格式，包含角色和内容。
"""
class Message(TypedDict):
    role: Role
    content: str

"""
    定义新会话响应格式，包含会话 ID。
    定义历史
"""
SESSIONS: Dict[str, List[Message]] = {}

class NewSessionResponse(BaseModel):
    session_id: str

class SessionStateResponse(BaseModel):
    session_id: str
    messages: list[ChatRequest]  

def get_or_create_session_messages(session_id: str) -> List[Message]:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return SESSIONS[session_id]

"""
    多伦对话，根据会话 ID 进行文本生成。
"""
class MultiChatRequest(BaseModel):
    session_id: str
    message: str

class MultiChatResponse(BaseModel):
    reply: str
    session_id: str
    messages_count: int

class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_id: str = ''
    history: list[dict] = []

"""
    RAG 查询请求
"""
class RagUploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int

class AgentRequest(BaseModel):
    query: str

class AgentResponse(BaseModel):
    reply: str
    steps: list[dict]

HTTP_GET_ALLOWLIST = {"api.open-meteo.com", "geocoding-api.open-meteo.com"}
TOOL_WHITELIST = {"calc", "get_weather", "http_get"}
DEFAULT_CITY = os.getenv("DEFAULT_WEATHER_CITY", "北京")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "计算器，支持 + - * /",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "op": {"type": "string", "enum": ["+", "-", "*", "/"]},
                },
                "required": ["a", "b", "op"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询城市天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_get",
            "description": "发起 GET 请求，仅允许白名单域名",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
]

def _as_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def validate_tool_args(name: str, args: dict) -> tuple[bool, str]:
    if name == "calc":
        a = _as_number(args.get("a"))
        b = _as_number(args.get("b"))
        op = args.get("op")
        if a is None or b is None:
            return False, "a/b 必须是数字"
        if abs(a) > 1e9 or abs(b) > 1e9:
            return False, "数值超出范围"
        if op not in {"+", "-", "*", "/"}:
            return False, "op 非法"
        if op == "/" and b == 0:
            return False, "除数不能为 0"
        return True, ""
    if name == "get_weather":
        city = str(args.get("city", "")).strip()
        if not city or len(city) > 30:
            return False, "city 非法"
        return True, ""
    if name == "http_get":
        url = str(args.get("url", "")).strip()
        if not url:
            return False, "url 不能为空"
        parsed = urlparse(url)
        if parsed.scheme != "https":
            return False, "仅允许 https"
        if parsed.hostname not in HTTP_GET_ALLOWLIST:
            return False, "域名不在白名单"
        if len(url) > 2048:
            return False, "url 过长"
        return True, ""
    return False, "未知工具"

def _extract_city(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"([\u4e00-\u9fff]{2,10})(市|省|区|县)", text)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return ""

def _is_weather_intent(text: str) -> bool:
    if not text:
        return False
    return "天气" in text or "气温" in text or "温度" in text

def _format_weather(city: str, result: dict) -> str:
    if "error" in result:
        return f"{city}天气查询失败：{result['error']}"
    temp = result.get("temperature")
    wind = result.get("windspeed")
    code = result.get("weathercode")
    return f"{city}当前天气：温度 {temp}℃，风速 {wind}，天气码 {code}。"

def tool_calc(a: float, b: float, op: str) -> dict:
    if op == "+":
        return {"result": a + b}
    if op == "-":
        return {"result": a - b}
    if op == "*":
        return {"result": a * b}
    if op == "/":
        return {"result": a / b}
    return {"error": "invalid op"}

async def tool_get_weather(city: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "zh", "format": "json"},
            )
            geo.raise_for_status()
            geo_data = geo.json()
            if not geo_data.get("results"):
                return {"error": f"城市未找到: {city}"}
            loc = geo_data["results"][0]
            lat, lon = loc["latitude"], loc["longitude"]
            weather = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current_weather": "true",
                    "timezone": "Asia/Shanghai",
                },
            )
            weather.raise_for_status()
            w = weather.json().get("current_weather", {})
        return {
            "city": city,
            "latitude": lat,
            "longitude": lon,
            "temperature": w.get("temperature"),
            "windspeed": w.get("windspeed"),
            "weathercode": w.get("weathercode"),
        }
    except httpx.ReadTimeout:
        return {"error": "天气查询超时"}
    except httpx.RequestError as e:
        return {"error": f"天气查询失败: {str(e)}"}

async def tool_http_get(url: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
        return {"status": resp.status_code, "body": resp.text[:500]}
    except httpx.ReadTimeout:
        return {"error": "http_get 超时"}
    except httpx.RequestError as e:
        return {"error": f"http_get 失败: {str(e)}"}

async def dispatch_tool(name: str, args: dict) -> dict:
    if name == "calc":
        return tool_calc(float(args["a"]), float(args["b"]), str(args["op"]))
    if name == "get_weather":
        return await tool_get_weather(str(args["city"]))
    if name == "http_get":
        return await tool_http_get(str(args["url"]))
    raise HTTPException(status_code=400, detail=f"unknown tool: {name}")

"""
    异步操作，调用 OpenAI API 进行文本生成。
"""
async def call_llm(messages: List[Message]) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        last = messages[-1]["content"] if messages else ""
        return f"本地模拟回复（未配置 OPENAI_API_KEY），你说的是：{last}"
    
    payload = {
        "model": model,
        "messages": messages
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
        response = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        # response.raise_for_status()
        # data = response.json()
        from fastapi import HTTPException

        # 429 额度问题，还是频繁请求问题排查
        try:
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError:
            detail = response.text
            retry_after = response.headers.get("retry-after")
            if response.status_code == 429:
                raise HTTPException(status_code=429, detail={"error": detail, "retry_after": retry_after})
            raise HTTPException(status_code=response.status_code, detail={"error": detail})
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail={"error": "LLM 请求超时"})
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail={"error": str(e)})

    return data["choices"][0]["message"]["content"]

"""
    流式调用(SSE Server-Sent Events) LLM的函数，返回一个异步迭代器，每次迭代返回一个 token。
    AsyncIterator: 异步生成器，配合yeild使用
"""
async def stream_llm(messages: List[Message]) -> AsyncIterator[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{base_url}/chat/completions", headers=headers, json=payload) as resp:
            if resp.status_code >= 400:
                body = await resp.aread()
                raise HTTPException(status_code=resp.status_code, detail=body.decode("utf-8", errors="ignore"))
            
            """ 真实返回格式
                data: {delta token1}
                data: {delta token2}
                data: {delta token3}
                data: [DONE]
            """
            async for line in resp.aiter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue


                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                data = json.loads(data_str)
                choice = (data.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                chunk = delta.get("content")

                if not chunk:
                    # 兼容一些实现会放在 message.content
                    msg = choice.get("message") or {}
                    chunk = msg.get("content")

                if chunk:
                    yield chunk

"""
    RAG 的存储与工具函数
"""



@app.get("/api/health")
async def health():
    return {"status": "ok"}

"""
    简单的聊天接口，直接调用 OpenAI API 进行文本生成。
"""
@app.post("/api/chat/simple", response_model=ChatResponse)
async def chat(req: ChatRequest):
    reply = await call_llm([{"role": "user", "content": req.message}])
    return ChatResponse(reply=reply)

"""
    创建新会话接口，返回一个唯一的会话 ID。
"""
@app.post("/api/session/new", response_model=NewSessionResponse)
async def new_session():
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return NewSessionResponse(session_id=session_id)

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    return {"session_id": session_id, "messages": SESSIONS.get(session_id, [])}

"""
    多伦对话接口，根据会话 ID 进行文本生成。
"""
@app.post("/api/chat", response_model=MultiChatResponse)
async def chat(req: MultiChatRequest):
    session_id = req.session_id
    messages = get_or_create_session_messages(session_id)
    messages.append({"role": "user", "content": req.message})
    reply = await call_llm(messages)
    messages.append({"role": "assistant", "content": reply})
    return MultiChatResponse(reply=reply, session_id=session_id, messages_count=len(messages))

"""
    SSE 路由对话
"""
@app.post("/api/chat/stream")
async def chat_stream(req: MultiChatRequest):
    session_id = req.session_id
    messages = get_or_create_session_messages(session_id)
    messages.append({"role": "user", "content": req.message})

    # 历史剪裁，token 超限 = 报错
    trimmed = [messages[0]] + messages[-MAX_HISTORY_MESSAGES:]

    async def event_gen():
        parts: List[str] = []
        try:
            async for chunk in stream_llm(trimmed):
                parts.append(chunk)
                yield f"event: delta\ndata: {chunk}\n\n"
        except HTTPException as e:
            yield f"event: error\ndata: {json.dumps(e.detail, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return
        
        reply = "".join(parts)
        messages.append({"role": "assistant", "content": reply})
        yield "event: done\ndata: [DONE]\n\n"


    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection":"keep-alive"})

@app.post("/api/agent", response_model=AgentResponse)
async def agent(req: AgentRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        raise HTTPException(status_code=500, detail="缺少 OPENAI_API_KEY")

    system = (
        "你是一个会调用工具的助手。"
        "只有在无法直接回答时才调用工具。"
        "http_get 仅用于白名单域名。"
        "不要臆造参数。"
    )
    weather_intent = _is_weather_intent(req.query)
    city = _extract_city(req.query)
    if weather_intent and not city:
        city = DEFAULT_CITY

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": req.query},
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
        try:
            first = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": {"type": "function", "function": {"name": "get_weather"}} if weather_intent else "auto",
                },
            )
            first.raise_for_status()
        except httpx.ReadTimeout:
            raise HTTPException(status_code=504, detail="LLM 请求超时")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=str(e))
        data = first.json()
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        steps: list[dict] = []

        if weather_intent and not tool_calls:
            result = await tool_get_weather(city)
            return AgentResponse(
                reply=_format_weather(city, result),
                steps=[{"name": "get_weather", "arguments": {"city": city}, "result": result}],
            )

        if tool_calls:
            messages.append(msg)
            for call in tool_calls:
                name = call.get("function", {}).get("name", "")
                args: dict = {}
                if name not in TOOL_WHITELIST:
                    result = {"error": f"工具未授权: {name}"}
                else:
                    raw_args = call.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        args = {}
                    if name == "get_weather":
                        if not isinstance(args, dict):
                            args = {}
                        if not str(args.get("city", "")).strip():
                            args["city"] = city
                    ok, err = validate_tool_args(name, args if isinstance(args, dict) else {})
                    if not ok:
                        result = {"error": err}
                    else:
                        result = await dispatch_tool(name, args)

                steps.append({"name": name, "arguments": args, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            try:
                second = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={"model": model, "messages": messages},
                )
                second.raise_for_status()
            except httpx.ReadTimeout:
                raise HTTPException(status_code=504, detail="LLM 请求超时")
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=str(e))
            final = second.json()["choices"][0]["message"]["content"]
            return AgentResponse(reply=final, steps=steps)

        final_text = msg.get("content") or ""
        return AgentResponse(reply=final_text, steps=steps)

@app.post("/api/agent/stream")
async def agent_stream(req: AgentRequest):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        raise HTTPException(status_code=500, detail="缺少 OPENAI_API_KEY")

    system = (
        "你是一个会调用工具的助手。"
        "只有在无法直接回答时才调用工具。"
        "http_get 仅用于白名单域名。"
        "不要臆造参数。"
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": req.query},
    ]

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    weather_intent = _is_weather_intent(req.query)
    city = _extract_city(req.query)
    if weather_intent and not city:
        city = DEFAULT_CITY

    async def event_gen():
        async with httpx.AsyncClient(base_url=base_url, timeout=60) as client:
            try:
                first = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": TOOLS,
                        "tool_choice": {"type": "function", "function": {"name": "get_weather"}} if weather_intent else "auto",
                    },
                )
                if first.status_code >= 400:
                    yield f"event: error\ndata: {first.text}\n\n"
                    yield "event: done\ndata: [DONE]\n\n"
                    return
                data = first.json()
                msg = data["choices"][0]["message"]
                tool_calls = msg.get("tool_calls") or []
            except httpx.ReadTimeout:
                yield "event: error\ndata: LLM 请求超时\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return
            except httpx.RequestError as e:
                yield f"event: error\ndata: {str(e)}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return

            if weather_intent and not tool_calls:
                result = await tool_get_weather(city)
                text = _format_weather(city, result)
                yield f"event: delta\ndata: {text}\n\n"
                yield "event: done\ndata: [DONE]\n\n"
                return

            if tool_calls:
                messages.append(msg)
                for call in tool_calls:
                    name = call.get("function", {}).get("name", "")
                    args: dict = {}
                    if name not in TOOL_WHITELIST:
                        result = {"error": f"工具未授权: {name}"}
                    else:
                        raw_args = call.get("function", {}).get("arguments", "{}")
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                        except json.JSONDecodeError:
                            args = {}
                        if name == "get_weather":
                            if not isinstance(args, dict):
                                args = {}
                            if not str(args.get("city", "")).strip():
                                args["city"] = city
                        ok, err = validate_tool_args(name, args if isinstance(args, dict) else {})
                        if not ok:
                            result = {"error": err}
                        else:
                            result = await dispatch_tool(name, args)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id", ""),
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )

        try:
            async for chunk in stream_llm(messages):
                yield f"event: delta\ndata: {chunk}\n\n"
        except HTTPException as e:
            yield f"event: error\ndata: {json.dumps(e.detail, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection":"keep-alive"})

@app.post("/api/rag/upload", response_model=RagUploadResponse)
async def rag_upload(
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    chunk_overlap: int = Form(120),
):
    rag_utils.ensure_rag_dirs()

    doc_id = str(uuid.uuid4())
    doc_folder = rag_utils.UPLOAD_DIR / doc_id
    doc_folder.mkdir(parents=True, exist_ok=True)

    raw = await file.read()
    saved_path = doc_folder / file.filename
    saved_path.write_bytes(raw)

    text = rag_utils.extract_text(file.filename, raw)
    parts = rag_utils.chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    if not parts:
        raise HTTPException(status_code=400, detail="文档为空或无法解析出文本")

    # 将文本列表转成向量列表，
    chunk_texts = [p[2] for p in parts]
    embs = await rag_utils.embed_texts(chunk_texts)
    embs = [rag_utils.normalize(v) for v in embs]

    docs = rag_utils.load_docs()
    docs.append(rag_utils.make_doc_item(doc_id=doc_id, filename=file.filename, size=len(raw)))
    rag_utils.save_docs(docs)

    for idx, ((start, end, piece), emb) in enumerate(zip(parts, embs)):
        chunk_id = f"{doc_id}_{idx}"
        rag_utils.append_chunk(
            {
                "doc_id": doc_id,
                "filename": file.filename,
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "text": piece,
                "embedding": emb,
            }
        )
    
    return RagUploadResponse(doc_id=doc_id, filename=file.filename, chunks_count=len(parts))

@app.post("/api/rag/query/stream")
async def rag_query_stream(req: RagQueryRequest):
    search_query = rag_utils.build_search_query(req.query, req.history)
    q_emb = (await rag_utils.embed_texts([search_query]))[0]
    q_emb = rag_utils.normalize(q_emb)

    rows = rag_utils.iter_chunks()
    scored: List[dict] = []
    for r in rows:
        if req.doc_id and r["doc_id"] != req.doc_id:
            continue
        score = rag_utils.dot(q_emb, r["embedding"])
        scored.append(
            {
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "chunk_id": r["chunk_id"],
                "score": float(score),
                "text": r["text"],
            }
        )

    scored.sort(key=lambda x: x["score"], reverse=True)
    hits = scored[: max(1, req.top_k)]
    messages = rag_utils.build_rag_messages_with_history(req.query, hits, req.history)

    async def event_gen():
        try:
            async for chunk in stream_llm(messages):
                yield f"event: delta\ndata: {chunk}\n\n"
        except HTTPException as e:
            yield f"event: error\ndata: {json.dumps(e.detail, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection":"keep-alive"})
