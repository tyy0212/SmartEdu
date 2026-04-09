# -*- coding: utf-8 -*-
import json
import time
from utils.logger import get_logger
from pathlib import Path
import pylanguagetool

# 导入新的MCP服务器实现
from mcp_server import mcp

# 导入向量数据库
from vector_db import get_vector_knowledge_base

# 配置日志
logger = get_logger(__name__)

# ==========================================
# 模拟数据层 (对应八股中的 RAG 向量库)
# ==========================================

# 1. 模拟教学大纲知识库 (Key-Value Store)
SYLLABUS_DB = {
    "光合作用": {
        "definition": "绿色植物通过叶绿体，利用光能，把二氧化碳和水转化成储存能量的有机物，并释放出氧气的过程。",
        "grade": "七年级生物",
        "key_points": ["光能 -> 化学能", "原料：CO2+H2O", "产物：有机物+O2"]
    },
    "勾股定理": {
        "definition": "如果直角三角形的两条直角边长分别为a，b，斜边长为c，那么 a^2 + b^2 = c^2。",
        "grade": "八年级数学",
        "key_points": ["直角三角形", "商高定理", "勾三股四弦五"]
    },
    "鲁迅": {
        "definition": "中国现代文学的奠基人，原名周树人。",
        "works": ["《狂人日记》", "《呐喊》", "《彷徨》", "《朝花夕拾》"],
        "common_errors": ["《骆驼祥子》（这是老舍的）", "《围城》（这是钱钟书的）"]
    }
}

# 2. 模拟错别字混淆集 (对应错别字识别模型)
TYPO_MAPPING = {
    "其怪": "奇怪",
    "以为": "已为 (需根据上下文判断)",
    "平率": "频率",
    "即使": "及时 (需根据上下文判断)",
    "安照": "按照",
    "震憾": "震撼",
    "副盖": "覆盖"
}

# ==========================================
# 工具定义层 (Agent 可调用的 Tools)
# ==========================================

@mcp.tool()
def check_typos_logic(text):
    """
    [NLP工具] 专门用于识别和修正文本中的错别字。
    在处理学生作文、作业文本时，必须优先调用此工具。
    输入：学生写的原始文本。
    输出：纠错报告和修正后的建议。
    """
    try:
        # 使用LanguageTool进行错别字和语法检查
        tool = pylanguagetool.LanguageTool('zh-CN')  # 中文检查

        # 检查文本
        matches = tool.check(text)

        if not matches:
            return json.dumps({
                "status": "clean",
                "message": "未发现明显错别字或语法错误"
            }, ensure_ascii=False)

        # 构建纠错报告
        found_errors = []
        corrections = []

        for match in matches:
            error_msg = f"位置 {match.offset}-{match.offset+match.errorLength}: '{match.context}'"
            if match.replacements:
                suggestion = match.replacements[0]
                error_msg += f" -> 建议修改为: '{suggestion}'"
                corrections.append({
                    "offset": match.offset,
                    "length": match.errorLength,
                    "original": match.context,
                    "suggestion": suggestion,
                    "rule": match.ruleId,
                    "message": match.message
                })
            else:
                error_msg += f" - {match.message}"
                corrections.append({
                    "offset": match.offset,
                    "length": match.errorLength,
                    "original": match.context,
                    "suggestion": "",
                    "rule": match.ruleId,
                    "message": match.message
                })

            found_errors.append(error_msg)

        # 应用纠正（简单的替换，实际应用可能需要更复杂的逻辑）
        corrected_text = text
        # 注意：按偏移量从后往前替换，避免偏移量变化
        for correction in sorted(corrections, key=lambda x: x["offset"], reverse=True):
            if correction["suggestion"]:
                start = correction["offset"]
                end = start + correction["length"]
                corrected_text = corrected_text[:start] + correction["suggestion"] + corrected_text[end:]

        return json.dumps({
            "status": "errors_found",
            "original": text,
            "correction_report": found_errors,
            "suggested_text": corrected_text,
            "corrections": corrections,
            "error_count": len(matches)
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"错别字检查失败: {e}")
        # 失败时回退到模拟检查
        found_errors = []
        corrected_text = text

        for wrong, right in TYPO_MAPPING.items():
            if wrong in text:
                found_errors.append("发现错误：'" + wrong + "' -> 建议修改为：'" + right + "'")
                corrected_text = corrected_text.replace(wrong, right)

        if not found_errors:
            return json.dumps({"status": "clean", "message": "未发现明显错别字"}, ensure_ascii=False)

        return json.dumps({
            "status": "errors_found",
            "original": text,
            "correction_report": found_errors,
            "suggested_text": corrected_text
        }, ensure_ascii=False, indent=2)

@mcp.tool()
def search_knowledge_base(query):
    """
    [RAG工具] 检索智慧教育教学大纲和知识点。
    当需要解释名词、验证事实准确性、或查询考点时调用。
    输入：关键词（如 '光合作用', '鲁迅'）。
    """
    try:
        # 获取向量知识库实例
        vector_kb = get_vector_knowledge_base()

        # 检查向量知识库是否可用
        if vector_kb is None:
            raise RuntimeError("向量知识库不可用，将使用模拟数据")

        # 执行语义搜索
        results = vector_kb.search(query, n_results=3)

        if not results:
            return json.dumps({
                "source": "Edu-Vector-DB",
                "status": "not_found",
                "message": "知识库中未检索到相关内容，请尝试更换关键词。"
            }, ensure_ascii=False, indent=2)

        # 格式化结果，保持与旧格式兼容
        formatted_results = []
        for result in results:
            formatted_result = {
                "topic": result["metadata"].get("topic", "未知"),
                "definition": result["metadata"].get("definition", result["text"]),
                "grade": result["metadata"].get("grade", "未知年级"),
                "key_points": result["metadata"].get("key_points", []),
                "similarity_score": round(result["score"], 3)
            }

            # 添加额外字段
            if "works" in result["metadata"]:
                formatted_result["works"] = result["metadata"]["works"]
            if "common_errors" in result["metadata"]:
                formatted_result["common_errors"] = result["metadata"]["common_errors"]
            if "subject" in result["metadata"]:
                formatted_result["subject"] = result["metadata"]["subject"]

            formatted_results.append(formatted_result)

        # 返回最佳匹配结果（保持与旧API兼容，只返回第一个）
        best_result = formatted_results[0]

        return json.dumps({
            "source": "Edu-Vector-DB",
            "content": best_result,
            "additional_results": formatted_results[1:] if len(formatted_results) > 1 else [],
            "search_query": query,
            "total_results": len(results)
        }, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"向量数据库检索失败: {e}")
        # 失败时回退到模拟检索
        result = SYLLABUS_DB.get(query)
        if result:
            return json.dumps({
                "source": "Edu-Vector-DB (模拟回退)",
                "content": result
            }, ensure_ascii=False, indent=2)
        else:
            return json.dumps({
                "source": "Edu-Vector-DB (错误回退)",
                "error": str(e),
                "message": "知识检索服务暂时不可用，请稍后重试。"
            }, ensure_ascii=False, indent=2)

@mcp.tool()
def analyze_classroom_video(video_id):
    """
    [多模态工具] 模拟视频理解模型，分析课堂视频中的学生状态。
    输入：视频ID。
    输出：学生专注度分析报告。
    """
    # 模拟视频理解模型的推理耗时
    # time.sleep(1) 
    
    # 这里的逻辑对应你简历中的“视频理解/多模态”
    return json.dumps({
        "video_id": video_id,
        "analysis_model": "Video-MAE-V2-Finetuned",
        "result": {
            "attendance": "45/45",
            "focus_level": "High",
            "abnormal_events": ["04:20 后排学生睡觉", "15:30 举手互动活跃"],
            "summary": "课堂互动良好，但在讲解公式推导时部分学生注意力下降。"
        }
    }, ensure_ascii=False, indent=2)

@mcp.tool()
def analyze_image(image_path, prompt="", image_type=""):
    """
    [VLM工具] 利用视觉-语言模型分析图像内容，提取语义信息。
    输入：
    - image_path: 图像文件路径
    - prompt: 可选的提示词，用于引导模型关注特定内容
    - image_type: 可选的图像类型，用于选择合适的提示词模板
    输出：图像分析结果，包括物体识别、场景描述、关键信息提取等。
    """
    # 选择合适的提示词
    prompt_templates = {
        "exam": "请详细分析这张考试试卷，识别题型、知识点、难度等级，以及可能的错误模式。重点关注答题区域和得分点。",
        "whiteboard": "请详细分析这张黑板/白板内容，提取教学要点、公式、图表和关键概念。重点关注教学逻辑和知识结构。",
        "teaching": "请详细分析这张教学图片，识别教学内容、关键元素和教育意义。重点关注如何将图像内容融入教学过程。",
        "experiment": "请详细分析这张实验图像，识别实验设备、操作步骤、现象和原理。重点关注实验的教育价值和安全注意事项。",
        "classroom": "请详细分析这张课堂场景图像，识别师生互动、教学活动和课堂氛围。重点关注教学效果和学生参与度。"
    }
    
    if prompt:
        final_prompt = prompt
    elif image_type and image_type in prompt_templates:
        final_prompt = prompt_templates[image_type]
    else:
        final_prompt = "详细描述图像中的内容，包括场景、物体、人物、动作等关键信息。重点关注与教育相关的元素和潜在的教学价值。"
    
    try:
        # 尝试使用LLaVA模型分析图像
        # 先检查依赖是否可用
        import sys
        if sys.version_info[0] < 3:
            raise ImportError("Python 3 required for LLaVA")

        # 尝试导入必要的模块
        try:
            from llava.model import load_pretrained_model
            from llava.mm_utils import process_images
            from PIL import Image
            import torch
        except ImportError as e:
            logger.warning(f"LLaVA依赖缺失，使用模拟分析: {e}")
            raise

        # 加载模型
        model_path = "liuhaotian/LLaVA-Lightning-7B-delta-v1-1"
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path=model_path,
            model_base="liuhaotian/LLaVA-Lightning-7B-v1-1",
            model_name="llava_lightning_7b_v1_1"
        )

        # 处理图像
        image = process_images([image_path], image_processor, model.config)
        image = image.to(model.device, dtype=torch.float16)

        # 构建输入
        inputs = tokenizer(["USER: <image>\n" + final_prompt + "\nASSISTANT:"], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        inputs["images"] = image

        # 生成回答
        with torch.no_grad():
            generate_kwargs = inputs.copy()
            generate_kwargs.update({
                "max_new_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.95
            })
            output = model.generate(**generate_kwargs)

        # 解析回答
        response = tokenizer.decode(output[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

        # 构建返回结果
        return json.dumps({
            "image_path": image_path,
            "analysis_model": "LLaVA-Lightning-7B",
            "prompt": final_prompt,
            "result": {
                "image_type": image_type if image_type else "通用图像",
                "content_analysis": response,
                "key_elements": ["根据分析结果提取的关键元素"],
                "suggestions": "根据分析结果提供的教学建议"
            }
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        # 如果LLaVA模型调用失败，使用模拟结果
        return json.dumps({
            "image_path": image_path,
            "analysis_model": "VLM-Edu-1.0 (模拟)",
            "prompt": final_prompt,
            "result": {
                "image_type": image_type if image_type else "通用图像",
                "content_analysis": "这是一张解剖学相关的图片，包含人体器官或组织结构的示意图。",
                "key_elements": ["解剖结构", "器官名称", "生理功能"],
                "suggestions": "可以结合图像讲解解剖学知识，帮助学生理解人体结构。"
            }
        }, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    mcp.run()