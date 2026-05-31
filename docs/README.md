# LLM RolePlay

一个功能强大的大语言模型角色扮演对话系统，支持分层长期记忆管理、深度回忆功能、现代化 Web UI、多模型选择、多模态对话等特性。

## 特性

- 多角色支持: 创建和管理多个个性化角色
- 分层记忆管理: 日/周/月/年四级记忆总结，角色能够"记住"长期对话
- 深度回忆功能: 支持三种回忆模式（关闭/正常/增强），让角色灵活回忆不同时期的记忆
- LLM 智能总结: 使用 Qwen3.5-Flash 思考模式生成高质量总结，配合专用 Prompt 模板
- WebSocket 流式对话: 实时流式响应
- 多模型支持: 通义千问系列模型（Qwen3、Qwen3.5、Qwen3.6、Qwen3.7 等）及 DeepSeek 系列模型
- 多模态对话: 支持图片上传和视觉理解
- 深度思考: 支持模型深度思考模式
- 联网搜索: 集成网络搜索能力
- 对话统计: 详细的模型和角色使用统计
- 知识库功能: 支持知识库管理和文档引用

## 技术栈

### 后端
- Web 框架: FastAPI 0.104.1
- ASGI 服务器: Uvicorn 0.24.0
- 数据验证: Pydantic 2.5.0

### 大模型与向量数据库
- LLM 服务: 阿里云通义千问 (DashScope)
- 向量数据库: ChromaDB 0.4.18
- 嵌入模型: BAAI/bge-large-zh-v1.5
- 深度学习框架: PyTorch 2.1.0

### 前端
- UI 库: Ant Design 5.15.0
- Markdown 渲染: Marked 12.0.0
- 代码高亮: Highlight.js 11.9.0
- 图表: Chart.js 4.4.1

## 安装

### 环境要求
- Python 3.8 或更高版本
- 阿里云通义千问 API Key

### 步骤

1. 克隆仓库
```bash
git clone <repository-url>
cd llm-roleplay
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

或者使用 pip 和 pyproject.toml:
```bash
pip install -e .
```

3. 配置 API Key

首次运行时，系统会提示您输入阿里云通义千问 API Key。您也可以手动创建 `data/credentials.json` 文件:

```json
{
  "api_key": "your-dashscope-api-key"
}
```

## 快速开始

详细使用方法可参考[项目开发文档](./项目功能文档.md)

### Web UI 模式（推荐）

启动 Web 服务器:
```bash
python main.py
```

然后在浏览器中访问: http://localhost:8000

## 使用说明

### Web UI 使用

1. 创建角色
   - 点击左侧"新建角色"按钮
   - 填写角色名称和设定（System Prompt）
   - 可选：上传头像
   - 配置高级参数（Temperature、Top P 等）

2. 开始对话
   - 从角色列表选择一个角色
   - 在输入框中输入消息
   - 可选：上传图片、选择模型、开启深度思考/联网搜索
   - 选择深度回忆模式（点击按钮循环切换：关→正常→增强→关）
   - 点击"发送"或按 Enter

3. 深度回忆功能
   
   系统提供两种触发深度回忆的方式和三种回忆模式：
   
   自动意图检测: 当您输入包含以下关键词时，系统会自动触发深度回忆：
   - "你还记得...吗"
   - "仔细回忆"
   - "很久以前"
   - "上次...说"
   - "最初的"
   - "咱们第一次"
   - "详细想想"
   
   三种深度回忆模式:
   
   | 回忆模式 | 描述 | 检索范围 | 适用场景 |
   |---------|------|---------|---------|
   | 关闭 | 不进行深度回忆，仅使用短期记忆 | 无 | 日常闲聊、简单对话 |
   | 正常 | 从分层总结库检索 | 日/周/月/年总结库 | 一般回忆需求，平衡精度和性能 |
   | 增强 | 从原始记录库检索 | 最近半年的原始对话记录 | 需要精确回忆细节时 |

4. LLM 智能总结
   
   v3.2.2 版本引入了全新的 LLM 总结算法：
   
   - 模型: Qwen3.5-Flash
   - 模式: 思考模式 (enable_think=True)
   - 思考预算: 4096 tokens
   
   为四个层级分别设计了专用的 Prompt 模板：
   - 日级总结：突出重要事件和情感变化（200-400字）
   - 周级总结：提炼本周核心主题和重要发展（300-500字）
   - 月级总结：提炼本月核心事件和成长变化（400-600字）
   - 年级总结：提炼本年核心主题和重大变化（500-800字）
   
   系统会优先使用 LLM 生成总结，失败时自动回退到原非 LLM 算法，确保稳定性。

5. 管理对话
   - 创建新对话
   - 切换历史对话
   - 重命名或删除对话

6. 查看统计
   - 点击左侧"个人中心"
   - 进入"统计信息"查看模型和角色使用数据

## 项目结构

```
project_root/
├── app/                      # Web 后端核心目录 (FastAPI)
│   ├── __init__.py
│   ├── main.py               # FastAPI 实例初始化与路由挂载
│   ├── exceptions.py         # 自定义异常类
│   ├── api/                  # API 路由控制器
│   │   ├── endpoints.py      # 常规 HTTP 接口 (角色管理、设置、统计等)
│   │   └── websockets.py     # WebSocket 全双工通信接口 (实时对话)
│   ├── config/               # 配置管理模块
│   │   ├── __init__.py
│   │   ├── models.py         # 配置数据模型
│   │   └── settings.py       # 全局设置与环境配置
│   ├── schemas/              # Pydantic 数据验证模型
│   │   ├── chat_schema.py    # WebSocket 消息输入输出模型
│   │   └── role_schema.py    # 角色信息验证模型
│   ├── services/             # 业务逻辑层
│   │   ├── __init__.py
│   │   ├── auth_manager.py   # 认证与凭证管理
│   │   ├── chat_service.py   # 对话服务 (对接 RoleplaySession)
│   │   ├── data_service.py   # 通用数据服务
│   │   ├── role_manager.py   # 角色管理服务
│   │   ├── stats_service.py  # 统计数据服务
│   │   └── user_manager.py   # 用户管理服务
│   ├── middleware/           # 中间件
│   │   └── error_handler.py  # 全局异常处理器
│   ├── static/               # 前端静态资源
│   │   ├── css/
│   │   │   └── style.css     # 前端样式文件
│   │   └── js/
│   │       ├── app.js            # 入口文件：事件绑定初始化
│   │       └── modules/          # 前端模块化目录
│   │           ├── state.js          # 全局状态树 + DOM 缓存
│   │           ├── utils.js          # 纯工具函数
│   │           ├── websocket.js      # WebSocket 连接管理
│   │           ├── roles.js          # 角色列表 + 对话管理 + 侧边栏
│   │           ├── chat-messages.js  # 消息气泡渲染 + 历史加载
│   │           ├── input-ui.js       # 输入气泡 + 发送 + 功能开关
│   │           ├── image-cropper.js  # 图片上传 + Canvas 裁剪器
│   │           ├── model.js          # 模型选择器 + 启动序列
│   │           ├── stats.js          # 统计页面 (Chart.js)
│   │           ├── knowledge-base.js # 知识库 CRUD + 文档管理
│   │           └── doc-selector.js   # 文档选择器 + 引用 Tag
│   ├── templates/            # HTML 模板页面
│   │   └── index.html        # 前端主页面
│   └── utils/                # 工具函数
│       └── logger.py         # 日志工具
├── data/                     # 数据存储目录
│   ├── avatars/              # 头像文件存储
│   ├── hf_models/            # Hugging Face 模型缓存
│   ├── roles/                # 角色数据目录
│   │   └── {role_name}/      # 单个角色的数据
│   │       ├── role_meta.json         # 角色元数据 (设定、配置等)
│   │       ├── conversations.json     # 对话记录索引
│   │       ├── conversations/         # 对话上下文文件
│   │       ├── raw_records/           # 原始对话记录 (按半年分区)
│   │       ├── summary_L1_daily/      # 日级总结向量库 (ChromaDB)
│   │       ├── summary_L2_weekly/     # 周级总结向量库 (ChromaDB)
│   │       ├── summary_L3_monthly/    # 月级总结向量库 (ChromaDB)
│   │       └── summary_L4_yearly/     # 年级总结向量库 (ChromaDB)
│   ├── stats/                # 统计数据
│   │   ├── model_stats.json  # 模型使用统计
│   │   └── role_stats.json   # 角色使用统计
│   ├── credentials.json      # API 凭证配置
│   ├── roles_registry.json   # 角色注册表
│   └── user_meta.json        # 用户元数据
├── llm_adapters/             # LLM 适配器层
│   ├── __init__.py
│   ├── base_adapter.py       # 适配器基类 (定义统一接口)
│   └── qwen_native_adapter.py    # 通义千问原生 API 适配器
├── vdb_tools/                # 向量数据库与记忆管理工具
│   └── hierarchical_memory_db.py  # 分层记忆管理器 (核心模块)
├── logs/                     # 日志文件目录
│   └── app.log               # 应用日志
├── scripts/                  # 脚本工具
│   └── migrate_role_meta.py  # 角色元数据迁移脚本
├── tools/                    # 辅助工具
│   ├── API_test_tool.py      # API 测试工具
│   ├── image_token_counter.py # 图片 Token 计数工具
│   └── test.jpg              # 测试图片
├── main.py                   # 全局启动入口 (Web 模式)
├── roleplay_core.py          # 角色扮演核心会话逻辑 (RoleplaySession)
├── requirements.txt          # Python 依赖列表
├── pyproject.toml            # 项目配置文件 (Poetry/pip)
├── structure                 # 本文件 - 项目结构说明
├── test_api_latency.py       # API 延迟测试脚本
└── test_qwen_adapter.py      # 通义千问适配器测试脚本
```

## 分层记忆系统

项目采用创新的四级分层记忆架构:

- L0 (工作记忆): 实时对话滑动窗口
- L1 (日级总结): 每日对话压缩总结
- L2 (周级总结): 每周对话总结
- L3 (月级总结): 每月对话总结
- L4 (年级总结): 每年对话总结

记忆系统会自动检测时间和容量边界，在空闲时进行压缩总结，确保持久化长期记忆的同时保持对话的流畅性。

## 深度回忆功能详解

深度回忆是系统的核心特色，让角色能够真正"记住"过去的重要对话：

### 工作原理
1. 语义向量检索: 使用 BAAI/bge-large-zh-v1.5 模型将您的输入向量化
2. 智能检索策略: 
   - 正常模式: 从日/周/月/年四个总结库中进行语义检索，根据时间范围智能选择检索范围
   - 增强模式: 直接从最近半年的原始对话记录库检索，提供最精确的回忆效果
3. 记忆注入: 将检索到的相关记忆片段插入到系统提示词中，让 AI 参考

### 使用场景
- 日常闲聊？选择"关闭"模式
- 想让角色回忆起过去的重要事件？选择"正常"模式
- 想让角色精确回忆起对话细节？选择"增强"模式（仅限最近半年）

## LLM 总结算法详解

### 核心优势
- 高质量总结: 使用 Qwen3.5-Flash 思考模式，生成更准确、更连贯的总结
- 分层优化: 为日/周/月/年四个层级分别设计专用 Prompt，针对性更强
- 稳定可靠: 完善的回退机制，LLM 失败时自动使用非 LLM 算法

### Prompt 设计理念
每个层级的 Prompt 都经过精心设计，针对该层级的特点：
- 日级: 关注细节和情感
- 周级: 关注主题和发展
- 月级: 关注成长和关系
- 年级: 关注历史和反思

## 可选模型

系统支持以下大语言模型，各有其特点和适用场景：

### 旗舰级高性能模型

| 模型名称 | 描述 | 上下文窗口 | 支持功能 | 适用场景 |
|---------|------|-----------|---------|---------|
| **deepseek-v4-pro** | DeepSeek 旗舰级 MoE 大模型，总参 1.6T、激活 49B | 1,000,000 tokens | 深度思考、联网搜索、工具集成、代码解释器 | 高阶科研、复杂办公、深度智能代理 |
| **qwen3.7-max** | Qwen3.7 系列最大模型，面向智能体时代的旗舰模型 | 1,000,000 tokens | 深度思考、联网搜索、工具集成、代码解释器 | 复杂任务、智能体应用、长文档处理 |
| **qwen3-max** | 通义千问3旗舰版，支持外部工具 | 262,144 tokens | 深度思考、联网搜索、工具集成、代码解释器 | 复杂推理、代码生成、专业任务 |

### 高效平衡级模型

| 模型名称 | 描述 | 上下文窗口 | 支持功能 | 适用场景 |
|---------|------|-----------|---------|---------|
| **deepseek-v4-flash** | DeepSeek 高效轻量化 MoE 模型，总参 284B、激活 13B | 1,000,000 tokens | 深度思考、联网搜索、工具集成、代码解释器 | 日常对话、内容创作、基础 RAG、批量文案 |
| **qwen3.6-plus** | 通义千问 Plus 系列最新模型，性能与效果平衡 | 1,000,000 tokens | 深度思考、多模态、联网搜索 | 日常对话、内容创作、图片理解 |
| **qwen3.5-plus** | 通义千问 3.5 增强版 | 1,000,000 tokens | 深度思考、多模态、联网搜索 | 日常对话、内容创作、图片理解 |
| **qwen3.5-flash** | 通义千问 3.5 轻量版，快速响应 | 1,000,000 tokens | 深度思考、多模态、联网搜索 | 实时对话、简单任务、高频交互 |
| **qwen3.5-397b-a17b** | 超大规模模型，397B 总参数、A17B 激活参数 | 262,144 tokens | 深度思考、多模态 | 最复杂任务、专业领域 |

### 视觉语言模型

| 模型名称 | 描述 | 上下文窗口 | 支持功能 | 适用场景 |
|---------|------|-----------|---------|---------|
| **qwen3-vl-plus** | Qwen3 视觉语言增强版 | 262,144 tokens | 深度思考、图像理解、图像生成、OCR | 图片问答、视觉分析、文档理解 |
| **qwen3-vl-flash** | Qwen3 视觉语言轻量版 | 262,144 tokens | 图像理解 | 实时视觉任务、快速图像处理 |

## 版本更新

详细开发时间线可参考[项目开发文档](./docs/项目开发文档.md)

### v3.3.2.5 (2026-05-31) — 模型扩展更新

**新特性**:
- 🚀 **新增 DeepSeek 系列模型**: deepseek-v4-pro、deepseek-v4-flash
- 🚀 **新增 Qwen3.7 系列模型**: qwen3.7-max
- 🚀 **新增 Qwen3.6 系列模型**: qwen3.6-plus
- 🚀 **新增超大规模模型**: qwen3.5-397b-a17b
- 📝 **完善模型配置**: [app/config/models.py](file:///d:/Data/PYC/LLMRolePlay/project_version_3.3.2.5_workspace/app/config/models.py) 统一管理所有模型详细信息
- 📝 **更新 README**: 新增详细的可选模型列表与功能对比

**变更**:
- [app/config/models.py](file:///d:/Data/PYC/LLMRolePlay/project_version_3.3.2.5_workspace/app/config/models.py) 重构，新增 MODEL_DETAILS 字典，包含所有模型的完整描述、特性支持、上下文窗口大小等信息

### v3.3.2 (2026-04-19) — 前端模块化重构 + 拆分遗留修复

**新特性：**
- 🏗️ **chat.js 模块化拆分**: 将 3222 行单文件（100+ 函数）拆分为 12 个独立模块 + 1 个入口文件 (`app/static/js/modules/`)
  - `state.js` — 全局状态树 + DOM 缓存（基石模块）
  - `utils.js` — 纯工具函数
  - `websocket.js` — WebSocket 连接管理
  - `roles.js` — 角色管理 + 对话管理 + 侧边栏抽屉
  - `chat-messages.js` — 消息气泡渲染 + 历史加载
  - `input-ui.js` — 输入气泡 + 发送 + 功能开关(T/S/D)
  - `image-cropper.js` — 图片上传 + Canvas 裁剪器
  - `model.js` — 模型选择器 + 启动序列
  - `stats.js` — 统计页面 (Chart.js)
  - `knowledge-base.js` — 知识库 CRUD + 文档管理
  - `doc-selector.js` — 文档选择器 + 引用 Tag + 字符偏移计算
  - `app.js` — 入口文件：事件绑定初始化
- 设计原则：零构建工具、全局作用域兼容 HTML 内联事件、纯重构零行为变更

**Bug 修复：**
- 🔧 修复 image-cropper.js：补全 `uploadTrigger/uploadMenu/uploadDropdown` 三个 DOM 变量声明，消除 `cropper` TDZ 风险（const → let）
- 🔧 修复 stats.js：补全 `statsCharts/statsDom/statsCurrentModelId` 声明及 `#open-stats-drawer`/`#close-stats-drawer` 事件绑定

### v3.3.1 (2026-04-19)

新特性:
1. 知识库引用显示优化
   - 引用标签（Citation Tag）系统：将引用内容从内联全文显示改为紧凑的标签形式
   - 输入阶段：标签显示在输入框上方，展示文件名+字符范围（如 `报告.docx :123~246`），支持 × 删除和点击预览
   - 发送后阶段：只读标签显示在用户消息气泡下方，点击可预览引用内容
   - 支持多条引用同时存在，每条独立管理
2. 后端扩展
   - 新增 `CitationMeta` 数据模型：存储知识库/文档元数据及字符范围（kb_id, doc_id, char_start, char_end）
   - 新增文档内容范围查询 API：`GET /knowledge-bases/{kb_id}/documents/{doc_id}/content?start=N&end=M`
   - WebSocket 处理层根据 metadata 动态拼接完整引用文本到 LLM 请求
   - 持久化层 context JSON 存储 `knowledge_citations` 元数据数组（不含全文），总结时自动排除引用内容
3. 全选引用功能
   - 文档预览区工具栏新增"全选引用"按钮，一键将整个文档加入引用
4. 历史恢复优化
   - 加载带引用的历史消息时，自动剥离引用包装文本，避免气泡内重复显示

Bug 修复:
1. 修复确认引用按钮 TypeError（const 变量重复赋值）
2. 修复发送带引用消息时 appendChild TypeError（renderCitationTags 返回值问题）
3. 修复跨段落选区字符偏移漂移（Section Offset Map 机制精确映射 DOM 选区到 plain_text_preview 字符位置）

### v3.3.0 (2026-04-19)

新特性:
1. 知识库功能
   - 新建/删除知识库：支持创建和删除知识库，包含名称、描述等元数据
   - 文档上传与管理：支持上传 DOCX/XLSX/PDF 格式文档，自动解析内容
   - 文档预览：支持在文档选择器中预览解析后的文档内容（标题、段落、表格）
   - 文档删除：应用内自定义确认弹窗删除文档（非浏览器原生 confirm）
   - 知识库删除确认：应用内模态框确认删除知识库及关联文档
   - 文档选择器：双栏布局（左侧 KB+文档列表，右侧预览区），支持全屏模式
   - 文本选择引用：在预览区选中文本后可引用到对话中
   - Accordion 折叠面板：侧边栏角色列表与知识库列表通过折叠面板组织

2. UI 精细修复（9 轮迭代）
   - 弹窗样式统一：新建知识库弹窗、删除确认弹窗的标题居中、间距合理、关闭按钮样式正确
   - Emoji 替换为专业文本图标：文档类型显示 DOC/XLS/PDF 而非 emoji，操作按钮使用中文文本
   - 文档选择器布局优化：左右对齐改进、预览区占比扩大至 4/5 以上、全屏模式完整覆盖
   - 侧边栏知识库列表：卡片式展示（名称左对齐、文档数右对齐），与角色列表风格一致
   - Accordion 刷新机制：展开知识库面板时自动加载已有知识库列表
   - 预览/删除按钮功能实现：预览按钮打开文档选择器并加载指定文档，删除按钮使用自定义确认弹窗
   - 路由前缀修复：API 路由从 /api/knowledge-bases 修正为 /knowledge-bases（避免重复前缀导致 404）

变更:
1. API 路由前缀修正
   - endpoints.py 中所有 knowledge-base 相关路由移除 /api 前缀（main.py 已挂载 /api 前缀）

### v3.2.7 (2026-04-18)

新特性:
1. 输入框气泡功能
   - 输入内容超出一行时自动显示展开按钮（输入框上方居中）
   - 点击按钮弹出多行气泡，完整显示输入内容
   - 气泡最大高度为屏幕 1/3，超出后内部滚动
   - 气泡模式与原输入框双向同步，切换不丢内容
   - 发送消息后自动收起气泡
   - 解决长文本输入可读性差的问题
   - 上传图片按钮与输入框纵向居中对齐

### v3.2.6 (2026-04-18)

变更:
1. CLI 模式全面下线
   - 移除 `cli/` 目录及相关代码
   - 系统仅支持 Web UI 模式
   - 更新所有文档中的 CLI 相关描述

### v3.2.3 (2026-03-28)

更新:
1. 项目结构更新
   - 更新 README.md 中的项目结构描述，与实际结构保持一致

### v3.2.2 (2026-03-28)

新特性:
1. 深度回忆功能升级
   - 新增三种回忆模式：off/normal/enhanced
   - off: 关闭，不检索任何记忆
   - normal: 从分层总结库检索
   - enhanced: 从最近半年原始记录库检索

2. LLM 总结算法
   - 使用 Qwen3.5-Flash 思考模式
   - 为日/周/月/年四个层级分别设计专用 Prompt 模板
   - 优先使用 LLM 生成，失败时回退到原非 LLM 算法
   - 使用 enable_think=True 和 thinking_budget=4096

3. 代码清理
   - 移除 qwen_web_adapter.py（不再使用）

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 联系方式

如有问题或建议，欢迎通过 Issue 联系。

---

版本: v3.3.2.5
最后更新: 2026-05-31
