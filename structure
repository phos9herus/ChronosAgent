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
│   │       └── chat.js       # 前端交互逻辑
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
