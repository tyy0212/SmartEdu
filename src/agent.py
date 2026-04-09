# -*- coding: utf-8 -*-
# 加reflection architecture

# 真实 LangGraph 和 LangChain 导入
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
# 尝试导入SqliteSaver，如果失败则使用MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:
    from langgraph.checkpoint.memory import MemorySaver as SqliteSaver

# 配置和LLM导入
import json
import os
from datetime import datetime
from typing import Optional, Dict, Any
from ..config.settings import settings
from .utils.logger import get_logger

# LLM客户端
class LLMClient:
    """统一的LLM客户端，支持OpenAI和Claude，使用最新SDK"""

    def __init__(self):
        self.openai_api_key = settings.openai_api_key
        self.anthropic_api_key = settings.anthropic_api_key
        self.default_model = settings.default_llm_model
        self.fallback_model = settings.fallback_llm_model
        self.max_tokens = settings.llm_max_tokens
        self.temperature = settings.llm_temperature
        self.enable_streaming = settings.enable_llm_streaming
        self.logger = get_logger(__name__)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """生成文本响应，支持模型切换和重试"""
        # 优先使用OpenAI
        if self.openai_api_key:
            self.logger.info("使用OpenAI生成响应", model=self.default_model)
            return self._generate_openai(prompt, system_prompt, **kwargs)
        # 其次使用Claude
        elif self.anthropic_api_key:
            self.logger.info("使用Claude生成响应", model=self.fallback_model)
            return self._generate_anthropic(prompt, system_prompt, **kwargs)
        else:
            # 如果没有配置API密钥，返回模拟响应
            self.logger.warning("未配置API密钥，返回模拟响应")
            return "【LLM生成】由于未配置API密钥，使用模拟响应。请配置OPENAI_API_KEY或ANTHROPIC_API_KEY环境变量。"

    def _generate_openai(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """使用OpenAI API生成（新版SDK）"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.openai_api_key)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            model = kwargs.get('model', self.default_model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)

            self.logger.debug("调用OpenAI API", model=model, max_tokens=max_tokens)

            if self.enable_streaming:
                # 流式响应（暂不支持，返回完整响应）
                self.logger.warning("流式响应暂未实现，使用非流式调用")

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except ImportError:
            return "【错误】未安装openai包，请运行 'pip install openai'"
        except Exception as e:
            self.logger.error("OpenAI API调用失败", error=str(e))
            return f"【OpenAI API错误】{str(e)}"

    def _generate_anthropic(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """使用Anthropic API生成"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)

            system_msg = system_prompt if system_prompt else ""
            model = kwargs.get('model', self.fallback_model)
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)

            self.logger.debug("调用Claude API", model=model, max_tokens=max_tokens)

            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_msg,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except ImportError:
            return "【错误】未安装anthropic包，请运行 'pip install anthropic'"
        except Exception as e:
            self.logger.error("Claude API调用失败", error=str(e))
            return f"【Anthropic API错误】{str(e)}"

    def stream(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        """流式生成响应（生成器）"""
        # 预留流式接口
        raise NotImplementedError("流式响应暂未实现")

# 创建全局LLM客户端实例
llm_client = LLMClient()

# 1. 定义状态：这是在节点间传递的数据结构
class AgentState(object):
    def __init__(self):
        self.messages = []            # 对话历史
        self.plan = ""                # 当前生成的教学计划/答案
        self.review_feedback = ""     # Reviewer 给出的修改意见
        self.retry_count = 0          # 重试次数，防止无限死循环
        self.image_analysis = ""      # 图像分析结果
        self.similarity_feedback = None  # 语义相似度反馈 (-1, 0, 1)

# 2. 定义 Planner 节点
def planner_node(state):
    # 这里会让 LLM 调用你的 MCP Tools (check_typos_logic, search_knowledge_base 等)
    # 逻辑：根据之前的 feedback（如果有）进行修正生成
    feedback = state.get("review_feedback", "")

    # 获取logger
    logger = get_logger(__name__)

    # 获取用户输入
    user_input = state["messages"][-1].content if state["messages"] else ""

    # 导入必要的模块
    import json
    import server

    # ContextGraph 集成
    # 获取thread_id（从会话信息或使用默认值）
    thread_id = "default"
    if "_session_info" in state and "graph_session_id" in state["_session_info"]:
        thread_id = state["_session_info"]["graph_session_id"]

    # 获取当前用户的ContextGraph实例
    context_graph = session_manager.get_context_graph(thread_id)

    # 1. 添加上一轮的相似度反馈到ContextGraph（如果有）
    similarity_feedback = state.get("similarity_feedback")
    if similarity_feedback is not None:
        # 创建反馈节点
        from .context_graph import create_review_feedback_node
        feedback_node = create_review_feedback_node(
            f"相似度反馈: {similarity_feedback}",
            metadata={"feedback_type": "similarity", "value": similarity_feedback}
        )
        context_graph.add_node(feedback_node)

        # 更新阈值管理器
        context_graph.update_similarity_threshold(similarity_feedback)

        # 清除已处理的反馈（避免重复处理）
        # 注意：我们将在返回值中设置similarity_feedback为None

    # 2. 添加当前用户输入节点到ContextGraph
    from .context_graph import create_user_input_node, create_plan_node
    user_input_node = create_user_input_node(user_input)
    context_graph.add_node(user_input_node)

    # 3. 检索相似历史节点作为上下文
    similar_nodes = context_graph.find_similar_nodes(user_input_node.node_id)
    semantic_context_summary = ""
    if similar_nodes:
        # 获取相似节点的内容摘要
        similar_content = []
        for node_id in similar_nodes[:3]:  # 取前3个最相似的节点
            node = context_graph.nodes.get(node_id)
            if node:
                content_preview = str(node.content)[:50] + "..." if len(str(node.content)) > 50 else str(node.content)
                similar_content.append(f"- {node.node_type.value}: {content_preview}")

        if similar_content:
            semantic_context_summary = "语义相关的历史对话:\n" + "\n".join(similar_content)
            logger.info(f"找到 {len(similar_nodes)} 个相似历史节点，使用前 {len(similar_content)} 个作为上下文")

    # 初始化工具结果
    typo_result = None
    image_analysis = ""

    # 1. 检查用户输入中是否包含图像相关内容
    if "图片" in user_input or "图像" in user_input or "photo" in user_input:
        # 根据用户输入判断图像类型
        image_type = ""
        # 使用指定的图片地址
        image_path = "images/test_anat_1.png"
        if "试卷" in user_input or "考试" in user_input:
            image_type = "exam"
        elif "黑板" in user_input or "白板" in user_input:
            image_type = "whiteboard"
        elif "实验" in user_input:
            image_type = "experiment"
        elif "课堂" in user_input:
            image_type = "classroom"
        else:
            image_type = "teaching"

        # 调用真实的analyze_image工具
        image_analysis = server.analyze_image(image_path, image_type=image_type)

    # 2. 调用错别字检查工具（如果用户输入包含中文文本）
    if user_input and any('\u4e00' <= char <= '\u9fff' for char in user_input):
        typo_result = server.check_typos_logic(user_input)

    # 3. 调用知识检索工具（提取关键词）
    # 简单的关键词提取：从用户输入中提取可能的知识点
    keywords = []
    for word in ["光合作用", "勾股定理", "鲁迅"]:
        if word in user_input:
            keywords.append(word)

    # 如果没有明确的关键词，尝试从输入中提取其他词
    if not keywords and len(user_input) > 0:
        # 简单分词：按空格和标点分割
        import re
        words = re.findall(r'[\u4e00-\u9fff]{2,4}', user_input)
        if words:
            # 取前两个词作为关键词
            keywords = words[:2]

    # 调用知识检索工具
    knowledge_results = []
    for keyword in keywords:
        result = server.search_knowledge_base(keyword)
        knowledge_results.append((keyword, result))

    # 4. 生成教学方案（使用LLM）
    # 基于工具结果构建教学方案

    # 解析工具结果，用于提示
    typo_summary = ""
    if typo_result:
        try:
            typo_data = json.loads(typo_result)
            if typo_data.get("status") == "errors_found":
                typo_summary = f"发现 {len(typo_data.get('correction_report', []))} 处错别字，已纠正。建议文本：{typo_data.get('suggested_text', '')}"
            else:
                typo_summary = "未发现明显错别字"
        except Exception:
            typo_summary = f"错别字检查结果：{typo_result}"

    knowledge_summary = ""
    if knowledge_results:
        knowledge_items = []
        for keyword, result in knowledge_results:
            try:
                knowledge_data = json.loads(result)
                if isinstance(knowledge_data, dict) and "content" in knowledge_data:
                    content = knowledge_data["content"]
                    item = f"{keyword}: {content.get('definition', '无')} (年级: {content.get('grade', '无')})"
                    if "key_points" in content:
                        item += f"，要点: {'，'.join(content['key_points'])}"
                    knowledge_items.append(item)
                else:
                    knowledge_items.append(f"{keyword}: {result}")
            except Exception:
                knowledge_items.append(f"{keyword}: {result}")
        knowledge_summary = "知识点检索结果：\n" + "\n".join(f"- {item}" for item in knowledge_items)
    else:
        knowledge_summary = "未检索到相关知识内容"

    image_summary = ""
    if image_analysis:
        try:
            image_data = json.loads(image_analysis)
            result_info = image_data.get("result", {})
            image_summary = f"图像类型：{result_info.get('image_type', '未知')}\n内容分析：{result_info.get('content_analysis', '无')}\n教学建议：{result_info.get('suggestions', '无')}"
        except Exception:
            image_summary = f"图像分析结果：{image_analysis}"

    # 构建LLM提示
    system_prompt = """你是一个智能教育助手，负责根据学生的需求和提供的工具分析结果，生成高质量的教学方案。
    教学方案应包括以下部分：
    1. 【教学方案】整体概述
    2. 【错别字检查】（如果有）
    3. 【知识点检索】（如果有）
    4. 【图像分析】（如果有）
    5. 【教学建议】具体的教学步骤和活动设计

    请根据提供的工具结果，生成结构清晰、内容具体、可执行的教学方案。"""

    # 构建完整的上下文提示
    context_parts = []
    if semantic_context_summary:
        context_parts.append(semantic_context_summary)

    full_context = "\n\n".join(context_parts) if context_parts else "无相关历史上下文"

    user_prompt = f"""用户输入：{user_input}

历史上下文：
{full_context}

工具分析结果：
1. 错别字检查：{typo_summary}
2. 知识检索：{knowledge_summary}
3. 图像分析：{image_summary}

审核反馈：{feedback if feedback else "无"}

请基于以上信息，生成一个完整的教学方案。确保方案包含所有必要的部分，并且针对工具结果进行具体回应。注意考虑历史上下文中的相关对话内容。"""

    # 调用LLM生成方案
    llm_response = llm_client.generate(user_prompt, system_prompt)

    # 确保响应包含必要部分
    response = llm_response
    if not response.startswith("【教学方案】"):
        response = "【教学方案】\n" + response

    # 添加计划节点到ContextGraph并建立依赖关系
    plan_node = create_plan_node(response)
    context_graph.add_node(plan_node)
    context_graph.add_dependency(plan_node.node_id, user_input_node.node_id)

    # 如果有关联的相似历史节点，也建立依赖
    for similar_node_id in similar_nodes[:3]:  # 仅关联最相似的前3个节点
        if similar_node_id in context_graph.nodes:
            context_graph.add_dependency(plan_node.node_id, similar_node_id)
            logger.debug(f"计划节点 {plan_node.node_id[:8]} 依赖于相似历史节点 {similar_node_id[:8]}")

    logger.info(f"添加计划节点到ContextGraph，依赖 {len(similar_nodes[:3])} 个相似历史节点")

    # 清除已处理的相似度反馈
    return {"plan": response,
            "messages": state["messages"] + [AIMessage(content=response)],
            "image_analysis": image_analysis,
            "similarity_feedback": None,  # 清除反馈，避免重复处理
            "similar_nodes_count": len(similar_nodes)}  # 传递相似节点数量用于反馈计算

# 3. 定义 Reviewer 节点
def reviewer_node(state):
    plan = state["plan"]
    messages = state.get("messages", [])
    image_analysis = state.get("image_analysis", "")

    # 获取logger
    logger = get_logger(__name__)

    # ContextGraph 集成
    # 获取thread_id（从会话信息或使用默认值）
    thread_id = "default"
    if "_session_info" in state and "graph_session_id" in state["_session_info"]:
        thread_id = state["_session_info"]["graph_session_id"]

    # 获取当前用户的ContextGraph实例
    context_graph = session_manager.get_context_graph(thread_id)

    # 提取完整的对话上下文（所有用户和AI消息）
    conversation_context = []
    for msg in messages:
        if hasattr(msg, 'content'):
            if isinstance(msg, AIMessage):
                conversation_context.append(f"AI: {msg.content}")
            else:
                conversation_context.append(f"用户: {msg.content}")

    # 提取最新用户输入
    user_input = ""
    for msg in reversed(messages):
        if hasattr(msg, 'content') and not isinstance(msg, AIMessage):
            user_input = msg.content
            break

    # 审核逻辑：检查教学方案的质量，充分利用所有可用信息
    feedback_items = []

    # 1. 检查方案是否包含必要部分
    required_sections = ["【教学方案】", "【教学建议】"]
    missing_sections = []
    for section in required_sections:
        if section not in plan:
            missing_sections.append(section)

    if missing_sections:
        feedback_items.append(f"方案缺少必要部分：{', '.join(missing_sections)}。请参考对话历史和图像分析结果进行补充。")

    # 2. 检查方案长度和详细程度（基于对话复杂度）
    lines = plan.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]

    # 根据对话长度和复杂度调整期望
    conversation_length = len(conversation_context)
    if conversation_length > 3 and len(non_empty_lines) < 8:
        feedback_items.append("对话历史较长，但方案内容过于简略，请提供更详细的教学步骤和内容。")
    elif len(non_empty_lines) < 5:
        feedback_items.append("方案内容过于简略，请提供更详细的教学步骤和内容。")

    # 3. 检查是否有具体的知识点内容（基于用户查询）
    if user_input:
        # 检查用户是否询问了具体知识点
        knowledge_keywords = ["解释", "讲解", "什么是", "含义", "定义", "知识点", "概念"]
        if any(keyword in user_input for keyword in knowledge_keywords):
            if "知识点" not in plan and "定义" not in plan and "解释" not in plan:
                feedback_items.append(f"用户询问了知识点（'{user_input[:30]}...'），但方案未包含具体的知识点内容。请结合教学大纲添加相关知识点。")

    # 4. 检查是否有具体的教学建议（基于用户需求）
    if "教学建议" in plan:
        # 检查教学建议是否具体
        suggestion_keywords = ["设计互动练习", "结合实例", "讲解", "演示", "活动", "步骤", "练习", "评估"]
        found_keywords = [kw for kw in suggestion_keywords if kw in plan]
        if len(found_keywords) < 2:
            feedback_items.append("教学建议不够具体，请参考对话历史和图像分析结果，提供更明确的教学活动设计。")
    else:
        feedback_items.append("方案缺少【教学建议】部分，请根据用户需求添加具体的教学建议。")

    # 5. 检查错别字（简单检查）
    common_typos = ["其怪", "平率", "安照", "震憾", "副盖"]
    found_typos = []
    for typo in common_typos:
        if typo in plan:
            found_typos.append(typo)

    if found_typos:
        feedback_items.append(f"方案中存在可能的错别字：{', '.join(found_typos)}，请检查并修正。")

    # 6. 上下文感知审核：检查方案是否充分回应了用户具体需求
    if user_input:
        # 检查用户是否提到了图像相关需求
        image_keywords = ["图片", "图像", "photo", "上传", "看图", "照片", "截图"]
        if any(keyword in user_input for keyword in image_keywords):
            # 检查是否有图像分析内容
            has_image_analysis = image_analysis and image_analysis.strip()
            has_image_content = any(keyword in plan for keyword in ["图像分析", "图片分析", "视觉内容", "图像说明", "图片内容", "图像描述"])

            if has_image_analysis:
                # 尝试提取图像分析的关键信息
                try:
                    analysis_data = json.loads(image_analysis)
                    result_info = analysis_data.get("result", {})
                    image_type = result_info.get("image_type", "")
                    content_analysis = result_info.get("content_analysis", "")

                    # 检查方案是否引用了具体的图像分析内容
                    if content_analysis and content_analysis not in plan:
                        feedback_items.append(f"图像分析发现了具体内容（'{content_analysis[:50]}...'），但方案未引用这些具体发现。请将图像分析结果整合到教学方案中。")
                except Exception:
                    pass

            if has_image_analysis and not has_image_content:
                feedback_items.append("用户提到了图像，方案中包含了图像分析结果但未在计划中明确展示图像相关内容。请将图像分析结果整合到教学方案中。")
            elif not has_image_analysis and not has_image_content:
                feedback_items.append("用户提到了图像，但方案未包含图像分析内容。请确保处理了用户的图像相关需求。")

        # 检查用户是否要求纠错
        typo_keywords = ["纠错", "错别字", "修改", "改正", "作文", "检查", "修正"]
        if any(keyword in user_input for keyword in typo_keywords):
            if "错别字" not in plan and "纠错" not in plan and "修改建议" not in plan and "检查结果" not in plan:
                feedback_items.append(f"用户要求纠错（'{user_input[:30]}...'），但方案未包含错别字检查或修改建议。请确保回应用户的纠错需求。")

    # 7. 深度检查图像分析内容是否被合理利用
    if image_analysis and image_analysis.strip():
        # 尝试解析图像分析结果
        try:
            analysis_data = json.loads(image_analysis)
            result_info = analysis_data.get("result", {})
            image_type = result_info.get("image_type", "")
            content_analysis = result_info.get("content_analysis", "")
            suggestions = result_info.get("suggestions", "")

            # 检查方案中是否提到了相关图像类型
            image_type_keywords = {
                "exam": ["试卷", "考试", "题目", "答题", "试题", "得分", "批改"],
                "whiteboard": ["黑板", "白板", "板书", "公式", "推导", "演算", "图解"],
                "teaching": ["教学", "教育", "讲解", "演示", "教案", "教材", "课件"],
                "experiment": ["实验", "设备", "操作", "现象", "结果", "数据", "观察"],
                "classroom": ["课堂", "学生", "老师", "互动", "讨论", "提问", "回答"]
            }

            if image_type in image_type_keywords:
                expected_keywords = image_type_keywords[image_type]
                found_image_keywords = [kw for kw in expected_keywords if kw in plan]

                if len(found_image_keywords) < 2:
                    feedback_items.append(f"图像分析识别为{image_type}类型，但方案未针对此类型提供足够针对性的教学建议。建议使用关键词如：{', '.join(expected_keywords[:3])}。")

                # 检查是否引用了具体的图像分析内容
                if content_analysis and content_analysis not in plan:
                    feedback_items.append("方案未引用图像分析的具体内容，请将分析结果整合到教学中。")

                if suggestions and suggestions not in plan:
                    feedback_items.append("图像分析提供了教学建议，但方案未采纳这些建议，请考虑整合。")

        except Exception:
            # JSON解析失败，但仍有图像分析文本
            if image_analysis not in plan:
                feedback_items.append("方案未充分整合图像分析结果，请参考图像分析内容改进教学方案。")

    # 8. 检查方案是否考虑了对话历史（多轮对话）
    if len(conversation_context) > 2:
        # 检查方案是否回应了之前的对话内容
        recent_context = "\n".join(conversation_context[-4:])  # 最近4条消息
        # 简单检查：方案是否包含对话中出现的关键词
        context_words = set(recent_context.replace("用户:", "").replace("AI:", "").split())
        plan_words = set(plan.split())
        common_words = context_words.intersection(plan_words)

        if len(common_words) < 5 and len(context_words) > 10:
            feedback_items.append("方案似乎未充分回应对话历史，请确保考虑了之前的讨论内容。")

    # 计算相似度反馈
    similar_nodes_count = state.get("similar_nodes_count", 0)
    similarity_feedback = 0.0  # 默认无反馈

    if similar_nodes_count > 0:
        # 有相似历史节点被引用
        if not feedback_items:
            # 审核通过：引用正确
            similarity_feedback = 1.0
            logger.info(f"相似度反馈: 1.0 (正确检测，引用了 {similar_nodes_count} 个相似历史节点)")
        else:
            # 审核未通过：可能引用不当
            # 检查反馈项是否与上下文引用相关
            context_related_feedback = any(
                "对话历史" in item or "上下文" in item or "历史" in item
                for item in feedback_items
            )
            if context_related_feedback:
                similarity_feedback = -1.0
                logger.info(f"相似度反馈: -1.0 (误判，审核发现上下文相关问题)")
            else:
                # 反馈与上下文无关，保持中性
                similarity_feedback = 0.0
                logger.info(f"相似度反馈: 0.0 (引用历史节点，但问题与上下文无关)")
    else:
        # 无相似历史节点被引用
        similarity_feedback = 0.0
        logger.info("相似度反馈: 0.0 (无相似历史节点被引用)")

    # 添加审核反馈节点到ContextGraph
    from .context_graph import create_review_feedback_node
    feedback_text = "Approve" if not feedback_items else "审核意见：" + "; ".join(feedback_items[:3])
    review_node = create_review_feedback_node(
        feedback_text,
        metadata={
            "similarity_feedback": similarity_feedback,
            "similar_nodes_count": similar_nodes_count,
            "feedback_items_count": len(feedback_items)
        }
    )
    context_graph.add_node(review_node)

    # 如果没有发现问题，则批准
    if not feedback_items:
        return {"review_feedback": "Approve", "similarity_feedback": similarity_feedback}
    else:
        # 将反馈项目连接成连贯的反馈
        feedback = "审核意见：\n" + "\n".join(f"- " + item for item in feedback_items)
        feedback += "\n\n请根据以上反馈修改教学方案，确保充分考虑对话历史、图像分析结果和用户具体需求。"
        return {"review_feedback": feedback,
                "retry_count": state.get("retry_count", 0) + 1,
                "similarity_feedback": similarity_feedback}

# 4. 定义逻辑判断（Conditional Edge）
def should_continue(state):
    if state["review_feedback"] == "Approve" or state.get("retry_count", 0) > 3:
        return "end"
    return "replan"


# 1. 图定义工厂函数
def create_workflow():
    """创建状态图工作流"""
    workflow = StateGraph(dict)
    workflow.add_node("planner", planner_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "reviewer")
    workflow.add_conditional_edges(
        "reviewer",
        should_continue,
        {
            "end": END,
            "replan": "planner"
        }
    )
    return workflow

# 2. 图会话管理器（替代全局单例）
class GraphSessionManager:
    """管理每个用户的独立图会话"""

    _instance = None
    _sessions = {}  # thread_id -> GraphSession

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化数据库连接"""
        db_url = settings.database_url
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
        else:
            db_path = "./smartedu.db"

        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.checkpointer = SqliteSaver(db_path)

        # 初始化 ContextGraph 池
        self._context_graphs = {}  # thread_id -> ContextGraph
        self._context_graphs_created = {}  # thread_id -> datetime
        self._context_graphs_accessed = {}  # thread_id -> datetime

    def get_session(self, thread_id: str):
        """获取或创建用户的图会话"""
        if thread_id not in self._sessions:
            # 创建新会话
            workflow = create_workflow()
            app = workflow.compile(checkpointer=self.checkpointer)

            self._sessions[thread_id] = {
                "app": app,
                "checkpointer": self.checkpointer,
                "created_at": datetime.now(),
                "last_accessed": datetime.now()
            }
        else:
            # 更新访问时间
            self._sessions[thread_id]["last_accessed"] = datetime.now()

        return self._sessions[thread_id]

    def cleanup_expired(self, max_age_seconds=3600):
        """清理过期的会话（内存缓存）"""
        now = datetime.now()
        expired = []
        for thread_id, session in self._sessions.items():
            age = (now - session["last_accessed"]).total_seconds()
            if age > max_age_seconds:
                expired.append(thread_id)

        for thread_id in expired:
            del self._sessions[thread_id]
    def _evict_expired(self):
        """清理过期的 ContextGraph 实例（惰性清理）"""
        from ..config.settings import settings

        now = datetime.now()
        expired = []
        for thread_id, created_at in self._context_graphs_created.items():
            age = (now - created_at).total_seconds()
            if age > settings.context_graph_ttl_seconds:
                expired.append(thread_id)

        for thread_id in expired:
            del self._context_graphs[thread_id]
            del self._context_graphs_created[thread_id]
            del self._context_graphs_accessed[thread_id]

    def get_context_graph(self, thread_id: str):
        """获取或创建用户的 ContextGraph 实例"""
        from ..config.settings import settings

        # 惰性清理过期实例
        self._evict_expired()

        # LRU淘汰：如果超过最大活跃实例数，淘汰最久未使用的
        max_active = settings.max_active_context_graphs
        if len(self._context_graphs) >= max_active and thread_id not in self._context_graphs:
            # 找到最久未访问的实例
            oldest_thread_id = min(
                self._context_graphs_accessed.items(),
                key=lambda x: x[1]
            )[0]
            # 淘汰最久未使用的实例（除非它是当前请求的线程）
            if oldest_thread_id != thread_id:
                del self._context_graphs[oldest_thread_id]
                del self._context_graphs_created[oldest_thread_id]
                del self._context_graphs_accessed[oldest_thread_id]

        if thread_id not in self._context_graphs:
            from .context_graph import ContextGraph
            self._context_graphs[thread_id] = ContextGraph()
            self._context_graphs_created[thread_id] = datetime.now()
            self._context_graphs_accessed[thread_id] = datetime.now()
        else:
            # 更新访问时间
            self._context_graphs_accessed[thread_id] = datetime.now()

        return self._context_graphs[thread_id]

# 全局会话管理器
session_manager = GraphSessionManager()

# 3. 兼容性：保留全局app引用（使用默认会话）
workflow = create_workflow()
app = workflow.compile(checkpointer=session_manager.checkpointer)

# 3. 支持持久化的运行方式
def run_agent_with_persistence(user_id, user_input):
    """
    运行代理并返回最终状态

    Args:
        user_id: 用户ID，用于会话隔离
        user_input: 用户输入文本

    Returns:
        最终状态字典，包含plan、messages等
    """
    # 获取用户特定的图会话
    session = session_manager.get_session(user_id)
    user_app = session["app"]

    # 为当前用户/会话指定唯一的线程ID
    config = {"configurable": {"thread_id": user_id}}

    # 尝试获取现有状态
    try:
        current_state = user_app.get_state(config)
        state_values = current_state.values

        # 确保状态中包含会话信息
        if "_session_info" not in state_values:
            state_values["_session_info"] = {
                "graph_session_id": user_id,
                "graph_version": "1.0",
                "session_created": session["created_at"].isoformat()
            }

        # 更新最后访问时间
        state_values["_session_info"]["last_accessed"] = datetime.now().isoformat()

        # 准备输入（保留所有状态字段）
        inputs = state_values.copy()
        # 将新消息附加到现有消息列表
        inputs["messages"] = inputs.get("messages", []) + [HumanMessage(content=user_input)]

    except Exception:
        # 如果没有现有状态，使用初始状态
        inputs = {
            "messages": [HumanMessage(content=user_input)],
            "plan": "",
            "review_feedback": "",
            "retry_count": 0,
            "image_analysis": "",
            "_session_info": {
                "graph_session_id": user_id,
                "graph_version": "1.0",
                "session_created": session["created_at"].isoformat(),
                "last_accessed": datetime.now().isoformat()
            }
        }

    final_state = None
    for event in user_app.stream(inputs, config=config):
        for node_name, output in event.items():
            print("--- 节点 " + node_name + " 正在执行 ---")
            # 这里可以实时打印输出
            if "image_analysis" in output and output["image_analysis"]:
                print("图像分析结果: " + output["image_analysis"])
            final_state = output

    return final_state

# 导出关键组件
__all__ = ["app", "run_agent_with_persistence", "llm_client", "AgentState"]