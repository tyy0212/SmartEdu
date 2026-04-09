# -*- coding: utf-8 -*-
"""
向量数据库模块，用于教育知识检索。
使用ChromaDB存储和检索教学大纲知识点。
"""

import json
from utils.logger import get_logger
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

class VectorKnowledgeBase:
    """向量知识库"""

    def __init__(self, collection_name: str = "edu_knowledge"):
        """
        初始化向量知识库

        Args:
            collection_name: ChromaDB集合名称
        """
        self.collection_name = collection_name
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None

        self._initialize()

    def _initialize(self):
        """初始化ChromaDB和嵌入模型"""
        try:
            # 初始化ChromaDB客户端（使用持久化存储）
            self.chroma_client = chromadb.PersistentClient(
                path="./data/chroma_db",
                settings=Settings(anonymized_telemetry=False)
            )

            # 加载嵌入模型（使用轻量级的中文模型）
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

            # 获取或创建集合
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "教育知识向量数据库"}
            )

            logger.info(f"向量知识库初始化完成，集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"初始化向量知识库失败: {e}")
            raise

    def add_knowledge(self, knowledge_items: List[Dict[str, Any]]):
        """
        添加知识条目到向量数据库

        Args:
            knowledge_items: 知识条目列表，每个条目包含:
                - id: 唯一标识符
                - text: 知识文本（用于嵌入）
                - metadata: 额外元数据（如定义、年级、关键点等）
        """
        if not knowledge_items:
            return

        ids = []
        texts = []
        metadatas = []

        for item in knowledge_items:
            ids.append(item["id"])
            texts.append(item["text"])
            metadatas.append(item.get("metadata", {}))

        # 生成嵌入向量
        embeddings = self.embedding_model.encode(texts).tolist()

        # 添加到集合
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )

        logger.info(f"添加了 {len(knowledge_items)} 条知识到向量数据库")

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        语义搜索知识库

        Args:
            query: 查询文本
            n_results: 返回结果数量

        Returns:
            搜索结果列表，每个结果包含:
                - id: 条目ID
                - text: 知识文本
                - metadata: 元数据
                - score: 相似度分数
        """
        if not query.strip():
            return []

        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query]).tolist()[0]

        # 执行搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # 格式化结果
        formatted_results = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i]  # 将距离转换为相似度分数
                })

        return formatted_results

    def initialize_with_default_data(self):
        """使用默认教学大纲数据初始化知识库"""
        default_knowledge = [
            {
                "id": "photosynthesis_1",
                "text": "光合作用是绿色植物通过叶绿体，利用光能，把二氧化碳和水转化成储存能量的有机物，并释放出氧气的过程。",
                "metadata": {
                    "topic": "光合作用",
                    "definition": "绿色植物通过叶绿体，利用光能，把二氧化碳和水转化成储存能量的有机物，并释放出氧气的过程。",
                    "grade": "七年级生物",
                    "key_points": ["光能 -> 化学能", "原料：CO2+H2O", "产物：有机物+O2"],
                    "subject": "生物"
                }
            },
            {
                "id": "pythagorean_theorem_1",
                "text": "勾股定理：如果直角三角形的两条直角边长分别为a，b，斜边长为c，那么 a^2 + b^2 = c^2。",
                "metadata": {
                    "topic": "勾股定理",
                    "definition": "如果直角三角形的两条直角边长分别为a，b，斜边长为c，那么 a^2 + b^2 = c^2。",
                    "grade": "八年级数学",
                    "key_points": ["直角三角形", "商高定理", "勾三股四弦五"],
                    "subject": "数学"
                }
            },
            {
                "id": "lu_xun_1",
                "text": "鲁迅是中国现代文学的奠基人，原名周树人。代表作品有《狂人日记》、《呐喊》、《彷徨》、《朝花夕拾》等。",
                "metadata": {
                    "topic": "鲁迅",
                    "definition": "中国现代文学的奠基人，原名周树人。",
                    "works": ["《狂人日记》", "《呐喊》", "《彷徨》", "《朝花夕拾》"],
                    "common_errors": ["《骆驼祥子》（这是老舍的）", "《围城》（这是钱钟书的）"],
                    "subject": "语文"
                }
            },
            {
                "id": "cell_structure_1",
                "text": "细胞是生物体的基本结构和功能单位，分为原核细胞和真核细胞两大类。",
                "metadata": {
                    "topic": "细胞结构",
                    "definition": "细胞是生物体的基本结构和功能单位",
                    "grade": "七年级生物",
                    "key_points": ["原核细胞", "真核细胞", "细胞膜", "细胞核", "细胞质"],
                    "subject": "生物"
                }
            },
            {
                "id": "quadratic_equation_1",
                "text": "一元二次方程是形如 ax^2 + bx + c = 0 的方程，其中a、b、c为常数，且a≠0。",
                "metadata": {
                    "topic": "一元二次方程",
                    "definition": "形如 ax^2 + bx + c = 0 的方程，其中a、b、c为常数，且a≠0。",
                    "grade": "九年级数学",
                    "key_points": ["求根公式", "判别式", "因式分解法", "配方法"],
                    "subject": "数学"
                }
            }
        ]

        # 检查集合是否已有数据
        existing_count = self.collection.count()
        if existing_count == 0:
            self.add_knowledge(default_knowledge)
            logger.info("已使用默认教学大纲数据初始化向量数据库")
        else:
            logger.info(f"向量数据库已有 {existing_count} 条数据，跳过默认数据初始化")


# 全局向量知识库实例
_vector_kb = None

def get_vector_knowledge_base() -> Optional[VectorKnowledgeBase]:
    """获取全局向量知识库实例（单例模式），失败时返回None"""
    global _vector_kb
    if _vector_kb is None:
        try:
            _vector_kb = VectorKnowledgeBase()
            # 初始化默认数据
            _vector_kb.initialize_with_default_data()
            logger.info("向量知识库初始化成功")
        except Exception as e:
            logger.error(f"向量知识库初始化失败，将回退到模拟数据: {e}")
            _vector_kb = None
    return _vector_kb