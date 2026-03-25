# ChronosAgent

**ChronosAgent** 是一个高性能多模态代理框架，具有分层时间衰减记忆、动态推理协调和自包含数字人物特性。

作为专为“数字生命”设计的通用智能体框架，ChronosAgent 引入了受生物启发的层级化时间感知记忆引擎，使智能体拥有纵贯线性的时间记忆能力，并支持深度的认知推理控制。

---

## 核心特点

### 1. 多级记忆引擎
核心逻辑位于 `hierarchical_memory_db.py`。Chronos 实现了从 **短期上下文 -> 日常摘要 -> 周/月/年级凝固** 的记忆演变过程。这让智能体能够像人类一样，记得住一年前的约定，同时模糊处理无关紧要的细节。

### 2. 确定性推理调度 
深度集成 Qwen-Reasoning 协议，通过前端实时控制 Thinking Budget。用户可以手动干预 AI 的思考深度，在“敏捷响应”与“深度逻辑推演”之间无缝切换。

### 3. 物理化人格内核 (Identity Encapsulation)
通过 `role_meta.json` 实现人格与数据的物理解耦。System Prompt、推理参数与层级记忆库 100% 独立存储，角色可以像“存档”一样被轻松迁移、备份与克隆。

---

## 技术特性 

### 后端架构
基于 **FastAPI** 与 **WebSocket** 的异步高并发架构，支持大规模流式数据传输。

### 适配器层
**QwenNativeAdapter** 深度适配阿里云 DashScope 接口，支持 65k Token 输出。

### 前端交互
风格化 WebUI，支持 Markdown 实时高亮、思维链折叠显示以及图片 Base64 并发处理。

### 存储系统
**ChromaDB** 向量底座 + 结构化 **JSON Metadata** 管理。

---
