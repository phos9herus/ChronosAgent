import os
import json
import uuid
import shutil
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from app.utils.logger import get_logger
from app.services.document_parser import document_parser


class KnowledgeBaseService:
    """知识库管理服务

    负责知识库的数据持久化层，包括：
    - 目录结构创建与管理
    - JSON 文件读写（注册表、元数据）
    - 知识库的增删改查操作
    - 数据验证与错误处理
    """

    def __init__(self):
        self.base_dir = os.path.join("data", "knowledge_bases")
        self.registry_path = os.path.join(self.base_dir, "kb_registry.json")
        self.logger = get_logger("knowledge_service")
        self._ensure_base_dir()

    def _ensure_base_dir(self):
        """确保 data/knowledge_bases/ 目录存在"""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            self.logger.debug(f"确保基础目录存在: {self.base_dir}")
        except OSError as e:
            self.logger.error(f"无法创建基础目录 {self.base_dir}: {e}")
            raise

    def _create_kb_directory(self, kb_id: str) -> str:
        """创建单个知识库目录结构

        目录结构:
        {kb_id}/
        ├── kb_meta.json
        ├── documents/
        └── index/

        Args:
            kb_id: 知识库唯一标识符 (UUID格式)

        Returns:
            str: 知识库目录路径
        """
        kb_dir = os.path.join(self.base_dir, kb_id)
        documents_dir = os.path.join(kb_dir, "documents")
        index_dir = os.path.join(kb_dir, "index")

        try:
            os.makedirs(documents_dir, exist_ok=True)
            os.makedirs(index_dir, exist_ok=True)
            self.logger.debug(f"创建知识库目录结构: {kb_dir}")
            return kb_dir
        except OSError as e:
            self.logger.error(f"无法创建知识库目录 {kb_dir}: {e}")
            raise

    def _validate_kb_name(self, name: str) -> bool:
        """验证知识库名称

        Args:
            name: 知识库名称

        Returns:
            bool: 验证是否通过
        """
        if not isinstance(name, str):
            return False
        if len(name.strip()) < 1 or len(name) > 100:
            return False
        return True

    def _validate_description(self, description: Optional[str]) -> bool:
        """验证描述文本

        Args:
            description: 描述文本

        Returns:
            bool: 验证是否通过
        """
        if description is None:
            return True
        if not isinstance(description, str):
            return False
        if len(description) > 500:
            return False
        return True

    def _validate_kb_id(self, kb_id: str) -> bool:
        """验证 kb_id 是否为有效的 UUID 格式

        Args:
            kb_id: 知识库 ID

        Returns:
            bool: 验证是否通过
        """
        try:
            uuid.UUID(str(kb_id))
            return True
        except (ValueError, AttributeError):
            return False

    def _get_current_timestamp(self) -> str:
        """获取当前 ISO 格式时间戳

        Returns:
            str: ISO 8601 格式时间戳
        """
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    def _load_registry(self) -> Dict[str, Any]:
        """加载注册表 JSON 文件

        Returns:
            dict: 注册表数据，格式为 {"knowledge_bases": [...]}
        """
        default_registry = {"knowledge_bases": []}

        if not os.path.exists(self.registry_path):
            return default_registry

        try:
            with open(self.registry_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self.logger.warning(f"注册表文件格式异常，使用默认值")
                return default_registry

            if "knowledge_bases" not in data or not isinstance(data["knowledge_bases"], list):
                data["knowledge_bases"] = []

            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"注册表 JSON 解析失败: {e}")
            return default_registry
        except IOError as e:
            self.logger.error(f"注册表文件读取失败: {e}")
            return default_registry

    def _save_registry(self, data: Dict[str, Any]) -> bool:
        """保存注册表到 JSON 文件

        Args:
            data: 要保存的注册表数据

        Returns:
            bool: 保存是否成功
        """
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            self.logger.debug(f"注册表已保存: {self.registry_path}")
            return True
        except IOError as e:
            self.logger.error(f"注册表文件写入失败: {e}")
            return False

    def _load_kb_meta(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """加载单个知识库的元数据

        Args:
            kb_id: 知识库 ID

        Returns:
            Optional[dict]: 元数据字典，如果不存在则返回 None
        """
        meta_path = os.path.join(self.base_dir, kb_id, "kb_meta.json")

        if not os.path.exists(meta_path):
            return None

        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"知识库 {kb_id} 元数据读取失败: {e}")
            return None

    def _save_kb_meta(self, kb_id: str, meta_data: Dict[str, Any]) -> bool:
        """保存单个知识库的元数据

        Args:
            kb_id: 知识库 ID
            meta_data: 元数据字典

        Returns:
            bool: 保存是否成功
        """
        meta_path = os.path.join(self.base_dir, kb_id, "kb_meta.json")
        try:
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=4)
            self.logger.debug(f"知识库 {kb_id} 元数据已保存")
            return True
        except IOError as e:
            self.logger.error(f"知识库 {kb_id} 元数据写入失败: {e}")
            return False

    def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """返回所有知识库摘要列表

        Returns:
            List[dict]: 知识库摘要列表，每个元素包含:
                - kb_id: 知识库 ID
                - name: 名称
                - document_count: 文档数量
                - created_at: 创建时间
        """
        registry = self._load_registry()
        return registry.get("knowledge_bases", [])

    def create_knowledge_base(self, name: str, description: str = "") -> Dict[str, Any]:
        """创建新的知识库

        执行步骤:
        1. 验证输入参数
        2. 生成 UUID 作为 kb_id
        3. 创建目录结构
        4. 初始化 kb_meta.json
        5. 更新注册表

        Args:
            name: 知识库名称 (必填, 1-100字符)
            description: 知识库描述 (可选, 最大500字符)

        Returns:
            dict: 新创建的知识库对象，包含完整元数据

        Raises:
            ValueError: 参数验证失败时抛出
            OSError: 文件系统操作失败时抛出
        """
        if not self._validate_kb_name(name):
            raise ValueError("知识库名称必填且长度应在 1-100 字符之间")

        if not self._validate_description(description):
            raise ValueError("描述长度不能超过 500 字符")

        kb_id = str(uuid.uuid4())
        current_time = self._get_current_timestamp()

        meta_data = {
            "kb_id": kb_id,
            "name": name.strip(),
            "description": description.strip() if description else "",
            "created_at": current_time,
            "updated_at": current_time,
            "document_count": 0,
            "documents": []
        }

        try:
            self._create_kb_directory(kb_id)

            if not self._save_kb_meta(kb_id, meta_data):
                raise IOError("元数据写入失败")

            registry = self._load_registry()
            summary_entry = {
                "kb_id": kb_id,
                "name": name.strip(),
                "document_count": 0,
                "created_at": current_time
            }
            registry.setdefault("knowledge_bases", []).append(summary_entry)

            if not self._save_registry(registry):
                raise IOError("注册表更新失败")

            self.logger.info(f"成功创建知识库: {name} (ID: {kb_id})")
            return meta_data

        except Exception as e:
            self.logger.error(f"创建知识库失败: {e}")
            kb_dir = os.path.join(self.base_dir, kb_id)
            if os.path.exists(kb_dir):
                shutil.rmtree(kb_dir, ignore_errors=True)
            raise

    def get_knowledge_base(self, kb_id: str) -> Optional[Dict[str, Any]]:
        """获取知识库详情

        Args:
            kb_id: 知识库 ID (UUID格式)

        Returns:
            Optional[dict]: 知识库完整元数据，不存在则返回 None
        """
        if not self._validate_kb_id(kb_id):
            self.logger.warning(f"无效的 kb_id 格式: {kb_id}")
            return None

        meta_data = self._load_kb_meta(kb_id)
        if meta_data is None:
            self.logger.debug(f"知识库不存在: {kb_id}")

        return meta_data

    def update_knowledge_base(
        self,
        kb_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """更新知识库元数据

        Args:
            kb_id: 知识库 ID (UUID格式)
            name: 新名称 (可选)
            description: 新描述 (可选)

        Returns:
            Optional[dict]: 更新后的知识库元数据，失败则返回 None
        """
        if not self._validate_kb_id(kb_id):
            self.logger.warning(f"无效的 kb_id 格式: {kb_id}")
            return None

        meta_data = self._load_kb_meta(kb_id)
        if meta_data is None:
            self.logger.warning(f"知识库不存在，无法更新: {kb_id}")
            return None

        if name is not None:
            if not self._validate_kb_name(name):
                raise ValueError("知识库名称必填且长度应在 1-100 字符之间")
            meta_data["name"] = name.strip()

        if description is not None:
            if not self._validate_description(description):
                raise ValueError("描述长度不能超过 500 字符")
            meta_data["description"] = description.strip() if description else ""

        meta_data["updated_at"] = self._get_current_timestamp()

        if not self._save_kb_meta(kb_id, meta_data):
            self.logger.error(f"更新知识库元数据失败: {kb_id}")
            return None

        registry = self._load_registry()
        for entry in registry.get("knowledge_bases", []):
            if entry.get("kb_id") == kb_id:
                if name:
                    entry["name"] = name.strip()
                break

        self._save_registry(registry)

        self.logger.info(f"成功更新知识库: {kb_id}")
        return meta_data

    def delete_knowledge_base(self, kb_id: str) -> bool:
        """删除知识库

        执行步骤:
        1. 验证 kb_id 格式和存在性
        2. 删除整个知识库目录（包括所有文档）
        3. 从注册表中移除条目

        Args:
            kb_id: 知识库 ID (UUID格式)

        Returns:
            bool: 删除是否成功
        """
        if not self._validate_kb_id(kb_id):
            self.logger.warning(f"无效的 kb_id 格式: {kb_id}")
            return False

        meta_data = self._load_kb_meta(kb_id)
        if meta_data is None:
            self.logger.warning(f"知识库不存在，无法删除: {kb_id}")
            return False

        kb_dir = os.path.join(self.base_dir, kb_id)

        try:
            if os.path.exists(kb_dir):
                shutil.rmtree(kb_dir, ignore_errors=True)
                self.logger.debug(f"已删除知识库目录: {kb_dir}")

            registry = self._load_registry()
            original_count = len(registry.get("knowledge_bases", []))
            registry["knowledge_bases"] = [
                entry for entry in registry.get("knowledge_bases", [])
                if entry.get("kb_id") != kb_id
            ]

            if len(registry["knowledge_bases"]) < original_count:
                self._save_registry(registry)

            self.logger.info(f"成功删除知识库: {kb_id}")
            return True

        except OSError as e:
            self.logger.error(f"删除知识库目录失败 {kb_dir}: {e}")
            return False

    def upload_document(
        self,
        kb_id: str,
        original_filename: str,
        file_content: bytes,
        file_type: str
    ) -> Dict[str, Any]:
        """上传文档到知识库并自动解析

        执行步骤:
        1. 验证知识库存在
        2. 生成 doc_id (UUID)
        3. 保存原始文件到 documents/{doc_id}.{ext}
        4. 调用 document_parser.parse_file() 解析
        5. 保存解析结果到 documents/{doc_id}_parsed.json
        6. 更新 kb_meta.json 的 documents 列表
        7. 返回文档元数据

        Args:
            kb_id: 知识库 ID (UUID格式)
            original_filename: 原始文件名 (如 "report.docx")
            file_content: 文件二进制内容
            file_type: 文件类型扩展名 (如 "docx", "xlsx", "pdf")

        Returns:
            dict: 文档元数据，包含:
                - doc_id: 文档唯一标识符
                - filename: 原始文件名
                - file_type: 文件类型
                - uploaded_at: 上传时间
                - parsed: 是否解析成功
                - parse_result: 解析结果摘要

        Raises:
            ValueError: 参数验证失败或知识库不存在
            IOError: 文件操作失败
        """
        if not self._validate_kb_id(kb_id):
            raise ValueError("无效的知识库 ID 格式")

        meta_data = self._load_kb_meta(kb_id)
        if meta_data is None:
            raise ValueError(f"知识库不存在: {kb_id}")

        if not original_filename or not isinstance(original_filename, str):
            raise ValueError("文件名不能为空")

        if not file_content or not isinstance(file_content, bytes):
            raise ValueError("文件内容不能为空")

        valid_types = ['docx', 'xlsx', 'pdf']
        if file_type.lower() not in valid_types:
            raise ValueError(f"不支持的文件类型: {file_type}，支持格式: {valid_types}")

        doc_id = str(uuid.uuid4())
        current_time = self._get_current_timestamp()
        ext = f".{file_type.lower()}"

        kb_dir = os.path.join(self.base_dir, kb_id)
        documents_dir = os.path.join(kb_dir, "documents")

        original_file_path = os.path.join(documents_dir, f"{doc_id}{ext}")
        parsed_file_path = os.path.join(documents_dir, f"{doc_id}_parsed.json")

        try:
            with open(original_file_path, 'wb') as f:
                f.write(file_content)

            self.logger.info(f"原始文件已保存: {original_file_path}")

            try:
                parsed_data = document_parser.parse_file(original_file_path)
                parsed_data['doc_id'] = doc_id

                save_success = document_parser.save_parsed_data(parsed_data, parsed_file_path)
                if not save_success:
                    self.logger.warning(f"解析结果保存失败: {parsed_file_path}")

                parse_success = 'error' not in parsed_data
                if not parse_success:
                    self.logger.warning(f"文档解析遇到问题: {parsed_data.get('error', '未知错误')}")

            except Exception as e:
                self.logger.error(f"文档解析失败: {e}, 文件: {original_file_path}")
                parsed_data = {
                    'doc_id': doc_id,
                    'filename': original_filename,
                    'file_type': file_type.lower(),
                    'error': str(e),
                    'parsed_at': datetime.now().isoformat(),
                    'sections': [],
                    'plain_text_preview': ''
                }
                parse_success = False

            doc_metadata = {
                'doc_id': doc_id,
                'filename': original_filename,
                'file_type': file_type.lower(),
                'file_size': len(file_content),
                'uploaded_at': current_time,
                'parsed_status': 'success' if parse_success else 'error',
                'parsed_data_path': f'documents/{doc_id}_parsed.json'
            }

            meta_data.setdefault('documents', []).append(doc_metadata)
            meta_data['document_count'] = len(meta_data['documents'])
            meta_data['updated_at'] = current_time

            if not self._save_kb_meta(kb_id, meta_data):
                raise IOError("更新知识库元数据失败")

            registry = self._load_registry()
            for entry in registry.get('knowledge_bases', []):
                if entry.get('kb_id') == kb_id:
                    entry['document_count'] = meta_data['document_count']
                    break
            self._save_registry(registry)

            self.logger.info(
                f"文档上传成功: {original_filename} -> 知识库 {kb_id} "
                f"(解析{'成功' if parse_success else '失败'})"
            )

            return {
                'doc_id': doc_id,
                'kb_id': kb_id,
                'filename': original_filename,
                'file_type': file_type.lower(),
                'file_size': len(file_content),
                'uploaded_at': current_time,
                'parsed_status': 'success' if parse_success else 'error',
                'parsed_message': None if parse_success else parsed_data.get('error', '解析失败')
            }

        except IOError as e:
            self.logger.error(f"文件写入失败: {e}")
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
            if os.path.exists(parsed_file_path):
                os.remove(parsed_file_path)
            raise
        except Exception as e:
            self.logger.error(f"上传文档过程中发生错误: {e}")
            if os.path.exists(original_file_path):
                os.remove(original_file_path)
            if os.path.exists(parsed_file_path):
                os.remove(parsed_file_path)
            raise

    def delete_document(self, kb_id: str, doc_id: str) -> bool:
        """删除文档及其解析数据

        执行步骤:
        1. 验证 kb_id 和 doc_id 格式
        2. 检查知识库和文档是否存在
        3. 删除原始文件和解析数据文件
        4. 更新 kb_meta.json 的 documents 列表
        5. 更新注册表中的文档计数

        Args:
            kb_id: 知识库 ID (UUID格式)
            doc_id: 文档 ID (UUID格式)

        Returns:
            bool: 删除是否成功
        """
        if not self._validate_kb_id(kb_id):
            self.logger.warning(f"无效的 kb_id 格式: {kb_id}")
            return False

        if not self._validate_kb_id(doc_id):
            self.logger.warning(f"无效的 doc_id 格式: {doc_id}")
            return False

        meta_data = self._load_kb_meta(kb_id)
        if meta_data is None:
            self.logger.warning(f"知识库不存在，无法删除文档: {kb_id}")
            return False

        documents = meta_data.get('documents', [])
        doc_to_delete = next((d for d in documents if d.get('doc_id') == doc_id), None)
        if not doc_to_delete:
            self.logger.warning(f"文档不存在: {doc_id}")
            return False

        kb_dir = os.path.join(self.base_dir, kb_id)
        documents_dir = os.path.join(kb_dir, "documents")

        try:
            file_type = doc_to_delete.get('file_type', '')
            original_file = os.path.join(documents_dir, f"{doc_id}.{file_type}")
            parsed_file = os.path.join(documents_dir, f"{doc_id}_parsed.json")

            for file_path in [original_file, parsed_file]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        self.logger.debug(f"已删除文件: {file_path}")
                    except OSError as e:
                        self.logger.warning(f"删除文件失败 {file_path}: {e}")

            meta_data['documents'] = [
                d for d in documents if d.get('doc_id') != doc_id
            ]
            meta_data['document_count'] = len(meta_data['documents'])
            meta_data['updated_at'] = self._get_current_timestamp()

            if not self._save_kb_meta(kb_id, meta_data):
                self.logger.error(f"更新知识库元数据失败: {kb_id}")
                return False

            registry = self._load_registry()
            for entry in registry.get('knowledge_bases', []):
                if entry.get('kb_id') == kb_id:
                    entry['document_count'] = meta_data['document_count']
                    break
            self._save_registry(registry)

            self.logger.info(f"成功删除文档: {doc_id} (知识库: {kb_id})")
            return True

        except Exception as e:
            self.logger.error(f"删除文档过程中发生错误: {e}")
            return False

    def get_document_content_range(self, kb_id: str, doc_id: str, char_start: int, char_end: int) -> dict:
        if not self._validate_kb_id(kb_id):
            return {"error": "无效的知识库 ID 格式"}

        if not self._validate_kb_id(doc_id):
            return {"error": "无效的文档 ID 格式"}

        meta_data = self._load_kb_meta(kb_id)
        if meta_data is None:
            return {"error": f"知识库不存在: {kb_id}"}

        documents = meta_data.get('documents', [])
        doc_info = next((d for d in documents if d.get('doc_id') == doc_id), None)
        if not doc_info:
            return {"error": f"文档不存在: {doc_id}"}

        parsed_data_path = doc_info.get('parsed_data_path')
        if not parsed_data_path:
            return {"error": "文档未解析或解析数据路径缺失"}

        kb_dir = os.path.join(self.base_dir, kb_id)
        full_parsed_path = os.path.join(kb_dir, parsed_data_path)

        if not os.path.exists(full_parsed_path):
            return {"error": f"解析数据文件不存在: {full_parsed_path}"}

        try:
            with open(full_parsed_path, 'r', encoding='utf-8') as f:
                parsed_data = json.load(f)

            plain_text = parsed_data.get('plain_text_preview', '')
            total_length = len(plain_text)

            if char_start < 0:
                char_start = 0

            if char_end is None or char_end > total_length:
                char_end = total_length

            if char_start >= total_length:
                return {
                    "content": "",
                    "doc_name": doc_info.get('filename', ''),
                    "total_length": total_length,
                    "error": "起始位置超出文档长度"
                }

            content = plain_text[char_start:char_end]

            return {
                "content": content,
                "doc_name": doc_info.get('filename', ''),
                "total_length": total_length
            }

        except (json.JSONDecodeError, IOError) as e:
            self.logger.error(f"读取解析数据文件失败: {e}")
            return {"error": f"读取解析数据文件失败: {str(e)}"}
        except Exception as e:
            self.logger.error(f"获取文档内容范围时发生错误: {e}")
            return {"error": f"获取文档内容范围时发生错误: {str(e)}"}


knowledge_service = KnowledgeBaseService()
