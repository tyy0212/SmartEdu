# -*- coding: utf-8 -*-
"""
上下文依赖图谱模块
基于图结构管理对话上下文依赖关系，解决长对话中的上下文丢失问题。
借鉴代码图谱(Code Graph)思想，将对话元素构建为有向无环图(DAG)。
"""

import uuid
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import cachetools
from ..config.settings import settings
from .utils.logger import get_logger

logger = get_logger(__name__)


class NodeType(Enum):
    """对话节点类型枚举"""
    USER_INPUT = "user_input"          # 用户输入
    TOOL_CALL = "tool_call"            # 工具调用
    TOOL_RESULT = "tool_result"        # 工具结果
    PLAN = "plan"                      # 生成的计划
    REVIEW_FEEDBACK = "review_feedback"  # 审核反馈
    IMAGE_ANALYSIS = "image_analysis"  # 图像分析
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"  # 知识检索


@dataclass
class ContextNode:
    """上下文节点"""
    node_id: str
    node_type: NodeType
    content: Any
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


class ContextGraph:
    """上下文依赖图谱"""

    def __init__(self):
        """初始化上下文图谱"""
        self.nodes: Dict[str, ContextNode] = {}  # 节点ID -> 节点
        self.edges: Dict[str, Set[str]] = {}     # 节点ID -> 依赖的节点ID集合
        self.reverse_edges: Dict[str, Set[str]] = {}  # 反向边，用于快速查找依赖者

        # 语义相似度相关组件
        self.embedding_model = None  # 延迟加载
        self.embedding_cache = cachetools.LRUCache(maxsize=100)  # 节点ID -> 嵌入向量
        self.similarity_threshold = settings.initial_similarity_threshold
        self.adaptive_threshold_manager = None  # 延迟初始化

    def _get_embedding_model(self):
        """获取嵌入模型（延迟加载）"""
        if self.embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = settings.semantic_similarity_model
                logger.info(f"加载语义相似度模型: {model_name}")
                self.embedding_model = SentenceTransformer(model_name)
            except Exception as e:
                logger.error(f"加载嵌入模型失败: {e}")
                # 回退到随机嵌入（仅用于开发）
                self.embedding_model = None
        return self.embedding_model

    def _get_node_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """获取节点的嵌入向量（带缓存）"""
        if node_id in self.embedding_cache:
            return self.embedding_cache[node_id]

        node = self.nodes.get(node_id)
        if node is None:
            return None

        # 从节点内容提取文本
        content_text = str(node.content)
        if not content_text.strip():
            return None

        model = self._get_embedding_model()
        if model is None:
            return None

        try:
            embedding = model.encode(content_text)
            self.embedding_cache[node_id] = embedding
            return embedding
        except Exception as e:
            logger.error(f"生成节点嵌入失败: {e}")
            return None

    def add_node(self, node: ContextNode) -> None:
        """添加节点"""
        self.nodes[node.node_id] = node
        if node.node_id not in self.edges:
            self.edges[node.node_id] = set()
        if node.node_id not in self.reverse_edges:
            self.reverse_edges[node.node_id] = set()

        logger.debug(f"添加上下文节点: {node.node_type.value}, ID: {node.node_id}")

    def add_dependency(self, from_node_id: str, to_node_id: str) -> None:
        """
        添加依赖关系：from_node 依赖 to_node

        Args:
            from_node_id: 依赖其他节点的节点ID
            to_node_id: 被依赖的节点ID
        """
        if from_node_id not in self.nodes:
            raise ValueError(f"节点 {from_node_id} 不存在")
        if to_node_id not in self.nodes:
            raise ValueError(f"节点 {to_node_id} 不存在")

        self.edges[from_node_id].add(to_node_id)
        self.reverse_edges[to_node_id].add(from_node_id)

        logger.debug(f"添加依赖关系: {from_node_id} -> {to_node_id}")

    def get_dependencies(self, node_id: str) -> Set[str]:
        """获取节点依赖的所有节点ID"""
        return self.edges.get(node_id, set()).copy()

    def get_dependents(self, node_id: str) -> Set[str]:
        """获取依赖该节点的所有节点ID"""
        return self.reverse_edges.get(node_id, set()).copy()

    def get_relevant_context(self, node_id: str, max_depth: int = 3) -> List[ContextNode]:
        """
        获取相关上下文节点（通过依赖关系回溯）

        Args:
            node_id: 起始节点ID
            max_depth: 最大回溯深度

        Returns:
            相关上下文节点列表（按依赖顺序排序）
        """
        if node_id not in self.nodes:
            return []

        visited = set()
        result = []

        def dfs(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
            visited.add(current_id)

            # 添加当前节点（除了起始节点）
            if current_id != node_id:
                result.append(self.nodes[current_id])

            # 递归处理依赖节点
            for dep_id in self.edges.get(current_id, set()):
                dfs(dep_id, depth + 1)

        # 从起始节点的依赖开始
        for dep_id in self.edges.get(node_id, set()):
            dfs(dep_id, 1)

        return result

    def get_context_summary(self, node_id: str) -> str:
        """获取上下文摘要（用于LLM提示）"""
        relevant_nodes = self.get_relevant_context(node_id)
        if not relevant_nodes:
            return "无相关上下文"

        summary_parts = []
        for node in relevant_nodes:
            node_type = node.node_type.value
            content_preview = str(node.content)[:100] + "..." if len(str(node.content)) > 100 else str(node.content)
            summary_parts.append(f"[{node_type}] {content_preview}")

        return "相关上下文:\n" + "\n".join(summary_parts)

    def find_node_by_type(self, node_type: NodeType, limit: int = 5) -> List[ContextNode]:
        """根据节点类型查找节点"""
        result = []
        for node in self.nodes.values():
            if node.node_type == node_type:
                result.append(node)
                if len(result) >= limit:
                    break
        return result

    def to_json(self) -> str:
        """将图谱转换为JSON格式"""
        graph_data = {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [
                {"from": from_id, "to": to_id}
                for from_id, to_ids in self.edges.items()
                for to_id in to_ids
            ]
        }
        return json.dumps(graph_data, ensure_ascii=False, indent=2)

    def from_json(self, json_str: str) -> None:
        """从JSON恢复图谱"""
        graph_data = json.loads(json_str)

        # 清空当前图谱
        self.nodes.clear()
        self.edges.clear()
        self.reverse_edges.clear()

        # 恢复节点
        for node_data in graph_data["nodes"]:
            node = ContextNode(
                node_id=node_data["node_id"],
                node_type=NodeType(node_data["node_type"]),
                content=node_data["content"],
                timestamp=datetime.fromisoformat(node_data["timestamp"]),
                metadata=node_data.get("metadata", {})
            )
            self.add_node(node)

        # 恢复边
        for edge_data in graph_data["edges"]:
            self.add_dependency(edge_data["from"], edge_data["to"])

    def visualize(self) -> str:
        """生成简单的可视化表示（文本格式）"""
        lines = ["上下文依赖图谱:"]
        for node_id, node in self.nodes.items():
            dependencies = list(self.edges.get(node_id, set()))
            deps_str = ", ".join(dependencies) if dependencies else "无"
            lines.append(f"  {node.node_type.value}[{node_id[:8]}]: 依赖 -> {deps_str}")
        return "\n".join(lines)

    def calculate_similarity(self, node1_id: str, node2_id: str) -> float:
        """
        计算两个节点之间的余弦相似度

        Args:
            node1_id: 第一个节点ID
            node2_id: 第二个节点ID

        Returns:
            相似度分数 (0.0到1.0之间)
        """
        emb1 = self._get_node_embedding(node1_id)
        emb2 = self._get_node_embedding(node2_id)

        if emb1 is None or emb2 is None:
            return 0.0

        # 余弦相似度计算
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        # 将相似度限制在0-1范围
        return max(0.0, min(1.0, similarity))

    def find_similar_nodes(self, node_id: str, threshold: Optional[float] = None) -> List[str]:
        """
        查找与指定节点相似度高于阈值的历史节点

        Args:
            node_id: 查询节点ID
            threshold: 相似度阈值，如果为None则使用当前实例阈值

        Returns:
            相似节点ID列表（按相似度降序排序）
        """
        if threshold is None:
            threshold = self.similarity_threshold

        query_emb = self._get_node_embedding(node_id)
        if query_emb is None:
            return []

        similar_nodes = []
        for other_id, other_node in self.nodes.items():
            if other_id == node_id:
                continue

            # 检查是否为历史节点（时间戳早于查询节点）
            query_time = self.nodes[node_id].timestamp
            other_time = other_node.timestamp
            if other_time >= query_time:
                continue

            emb = self._get_node_embedding(other_id)
            if emb is None:
                continue

            # 计算余弦相似度
            dot_product = np.dot(query_emb, emb)
            norm1 = np.linalg.norm(query_emb)
            norm2 = np.linalg.norm(emb)

            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
                similarity = max(0.0, min(1.0, similarity))

            if similarity >= threshold:
                similar_nodes.append((other_id, similarity))

        # 按相似度降序排序
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in similar_nodes]

    def get_semantic_context(self, node_id: str, threshold: Optional[float] = None) -> List[ContextNode]:
        """
        获取与指定节点语义相关的上下文节点

        Args:
            node_id: 查询节点ID
            threshold: 相似度阈值

        Returns:
            相关的上下文节点列表（按相似度降序排序）
        """
        similar_node_ids = self.find_similar_nodes(node_id, threshold)
        return [self.nodes[node_id] for node_id in similar_node_ids]

    def update_similarity_threshold(self, feedback: float) -> None:
        """
        基于反馈更新相似度阈值（通过自适应阈值管理器）

        Args:
            feedback: 反馈值 (-1, 0, 1)
        """
        if self.adaptive_threshold_manager is None:
            # 延迟初始化自适应阈值管理器
            from .adaptive_threshold import AdaptiveThresholdManager
            self.adaptive_threshold_manager = AdaptiveThresholdManager(
                initial_threshold=self.similarity_threshold
            )

        if settings.adaptive_threshold_enabled:
            new_threshold = self.adaptive_threshold_manager.update_threshold(feedback)
            self.similarity_threshold = new_threshold
            logger.info(f"更新相似度阈值: {self.similarity_threshold:.3f} (基于反馈: {feedback})")

    def get_similarity_threshold(self) -> float:
        """获取当前相似度阈值"""
        return self.similarity_threshold


def create_user_input_node(user_input: str, metadata: Optional[Dict] = None) -> ContextNode:
    """创建用户输入节点"""
    return ContextNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.USER_INPUT,
        content=user_input,
        timestamp=datetime.now(),
        metadata=metadata or {}
    )


def create_tool_call_node(tool_name: str, tool_args: Dict, metadata: Optional[Dict] = None) -> ContextNode:
    """创建工具调用节点"""
    return ContextNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.TOOL_CALL,
        content={"tool_name": tool_name, "args": tool_args},
        timestamp=datetime.now(),
        metadata=metadata or {}
    )


def create_tool_result_node(tool_name: str, result: Any, metadata: Optional[Dict] = None) -> ContextNode:
    """创建工具结果节点"""
    return ContextNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.TOOL_RESULT,
        content={"tool_name": tool_name, "result": result},
        timestamp=datetime.now(),
        metadata=metadata or {}
    )


def create_plan_node(plan_content: str, metadata: Optional[Dict] = None) -> ContextNode:
    """创建计划节点"""
    return ContextNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.PLAN,
        content=plan_content,
        timestamp=datetime.now(),
        metadata=metadata or {}
    )


def create_review_feedback_node(feedback: str, metadata: Optional[Dict] = None) -> ContextNode:
    """创建审核反馈节点"""
    return ContextNode(
        node_id=str(uuid.uuid4()),
        node_type=NodeType.REVIEW_FEEDBACK,
        content=feedback,
        timestamp=datetime.now(),
        metadata=metadata or {}
    )


