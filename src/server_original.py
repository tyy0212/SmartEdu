# -*- coding: utf-8 -*-
import json
import time

# 模拟mcp模块
class MockFastMCP:
    def __init__(self, name):
        self.name = name
        self.tools = {}
    
    def tool(self, func=None):
        if func is None:
            def decorator(f):
                self.tools[f.__name__] = f
                return f
            return decorator
        else:
            self.tools[func.__name__] = func
            return func
    
    def run(self):
        print("Mock MCP server running...")

# 初始化 MCP 服务，名称为 SmartEdu_Agent
mcp = MockFastMCP("SmartEdu_Agent")

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
    found_errors = []
    corrected_text = text
    
    # 这里模拟你的 BERT/LLM 纠错模型的推理过程
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
    # 模拟向量检索过程
    result = SYLLABUS_DB.get(query)
    
    if result:
        return json.dumps({
            "source": "Edu-Vector-DB",
            "content": result
        }, ensure_ascii=False, indent=2)
    else:
        return "知识库中未检索到相关内容，请尝试更换关键词。"

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
        # 使用 LLaVA 模型分析图像
        from llava.model import load_pretrained_model
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from PIL import Image
        import torch
        from transformers import TextStreamer
        import torch
        
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
            # 在Python 2.7中，**inputs后面不能直接跟关键字参数
            # 我们需要将所有参数合并到一个字典中
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