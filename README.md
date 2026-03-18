# 启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8088 --env-file .env


# 通用 AI / LLM 基础知识点

- 基本概念：
  - 大语言模型（LLM）、Token、上下文窗口、Prompt
  - Chat Completion / Completion 的区别
  - System / User / Assistant / Tool 多种角色
- 能力边界：
  - 生成式 vs 检索式：为什么需要 RAG
  - Hallucination（幻觉）、对策（RAG、约束输出、验证）
- 常用术语：
  - Zero-shot / Few-shot
  - Temperature、Top_p
  - Function Calling / Tool Use / Agent

# 后端（Python + FastAPI）知识点

- FastAPI 基础：
  - 路由定义： @app.get/post ，路径参数、Query 参数、Body 参数
  - Pydantic 模型：请求/响应数据校验
  - 中间件 / 依赖注入（依赖注入可后看）
- 异步编程：
  - async def / await
  - 异步 HTTP 调用（httpx / aiohttp）
- LLM API 调用：
  - Chat Completion 请求体结构： model 、 messages 、 stream 、 tools 等
  - Streaming：如何以流形式读取响应（chunk / SSE）
  - 错误处理：超时、限流、Token 不足、自定义重试
- 文件上传与处理：
  - FastAPI 文件上传： UploadFile
  - 临时文件存储 / 持久存储
  - PDF / 文本解析：pypdf 等常见库（视需要选用）

# Chat / 多轮文本生成知识点

- 会话管理：
  - 如何保存上下文：内存 / Redis / DB
  - 截断历史：只保留最近 N 条消息，避免超过上下文窗口
  - 简单的用户体系（session token / user id）
- Streaming：
  - HTTP 分块传输 / SSE 基本概念
  - FastAPI 中实现流式响应
  - 前端如何逐块渲染（ReadableStream / EventSource）
- Prompt 设计：
  - System Prompt 负责设定助手人格 / 风格
  - 针对多轮对话如何在历史中嵌入关键指令

# RAG 知识库（文档上传 + 向量检索）知识点

- 文档处理：
  - 文本提取：PDF → 文本，Markdown / HTML 转纯文本
  - 文本清洗：去除多余空格、页眉页脚等噪音
- 文本切分（Chunking）：
  - 固定长度切分（按字符 / 按句子），配合滑动窗口 + 重叠
  - 按语义切分（参考段落、标题）
  - 设计 chunk 大小和重叠大小的 trade-off
- Embedding（文本向量）：
  - 什么是向量表示？余弦相似度 / 点积
  - 调用 Embedding 模型接口：输入文本列表 → 输出向量数组
  - 归一化 / 存储向量
- 向量检索：
  - 最近邻搜索（kNN）
  - 本地向量库：Faiss / Chroma 基本使用
  - 索引结构：平面索引 / IVF / HNSW（了解即可）
- RAG Prompt 组装：
  - 通用模板：把检索到的片段插入到系统提示中
  - 给模型明确指令：只能基于上下文回答，不知道就说不知道
- （可选）Rerank 重排序：
  - 用 Reranker 模型对检索结果进行重新排序
  - 提升答案相关性

  兼容

# Agent / Function Calling 知识点

- Function Calling 协议：
  - 定义 tools / functions ：名称、描述、入参 Schema（JSON Schema）
  - LLM 输出 tool_calls ：包含 function.name 和 arguments
  - 你需要解析 arguments，调用本地函数，返回结果
- 工具设计：
  - 工具类型：计算（确定性）、查询类（HTTP）、写操作（下单 / 写库）
  - 工具粒度：尽量“原子化”，描述清晰，避免过于宽泛
- Agent 流程：
  - 单轮工具调用：用户输入 → LLM 选择工具 → 调用工具 → 再次调用 LLM 输出最终答案
  - 多轮工具调用（可选）：循环执行，直到 LLM 不再要求 tool
- 可靠性 / 安全：
  - 工具参数校验（防止 LLM 传一些奇怪参数）
  - 对外部 API 调用做超时和错误处理
  - 对敏感操作工具（比如“删除文件”）谨慎开放
  
# AIGC 加分模块（文生图 / 文生文 / 智能代码生成）知识点

- 文生图：
  - Diffusion / Stable Diffusion 概念（了解即可）
  - 常见参数：prompt / negative_prompt / steps / cfg_scale
  - 图片返回方式：URL / base64 / 二进制流
- 文生文（写作助手）：
  - 多种任务：总结、翻译、润色、风格改写、续写
  - Prompt Pattern：指令 + 示例（Few-shot）+ 约束（字数、语气）
  - 模板封装：把任务类型抽象成接口参数
- 智能代码生成：
  - 代码模型特点：更严格的格式、偏向补全
  - Prompt 设计：
    - 明确语言、框架、约束（不要加解释，只要代码）
    - 可加已有代码上下文（比如函数签名）
  - 安全性：不直接执行模型生成的代码（仅展示）