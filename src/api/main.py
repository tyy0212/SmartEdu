# -*- coding: utf-8 -*-
"""
FastAPI RESTful接口 for SmartEdu-MCP
提供外部调用智能教育代理的能力。
"""

import json
from ..utils.logger import get_logger
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# 导入代理
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import run_agent_with_persistence

logger = get_logger(__name__)

# 定义请求和响应模型
class AgentRequest(BaseModel):
    """代理请求模型"""
    user_id: str = Query(..., description="用户ID，用于会话隔离")
    query: str = Query(..., description="用户查询文本")
    image_path: Optional[str] = Query(None, description="可选图像路径，如果查询涉及图像分析")

class AgentResponse(BaseModel):
    """代理响应模型"""
    status: str
    message: Optional[str] = None
    plan: Optional[str] = None
    error: Optional[str] = None
    session_id: Optional[str] = None

# 创建FastAPI应用
app = FastAPI(
    title="SmartEdu-MCP API",
    description="智能教育MCP代理RESTful接口",
    version="0.1.0"
)

@app.get("/")
async def root():
    """API根端点，返回服务信息"""
    return {
        "service": "SmartEdu-MCP API",
        "version": "0.1.0",
        "endpoints": {
            "/": "API信息",
            "/health": "健康检查",
            "/api/agent/query": "代理查询端点"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "SmartEdu-MCP"}

@app.post("/api/agent/query", response_model=AgentResponse)
async def query_agent(request: AgentRequest):
    """
    提交查询到智能教育代理

    参数:
    - user_id: 用户ID，用于会话隔离和状态恢复
    - query: 用户查询文本
    - image_path: 可选图像路径，如果查询涉及图像分析

    返回:
    - 代理生成的教学方案和状态
    """
    try:
        logger.info(f"收到代理查询请求: user_id={request.user_id}, query={request.query}")

        # 如果提供了图像路径，将其整合到查询中
        user_input = request.query
        if request.image_path:
            # 在实际应用中，可能需要将图像路径传递给代理
            # 这里简单地将图像信息添加到查询中
            user_input = f"{request.query} [图像路径: {request.image_path}]"

        # 调用代理（这里简化调用，实际需要处理代理的流式响应）
        # 注意：run_agent_with_persistence 会打印输出，但不返回结果
        # 我们需要修改代理函数以返回结果，或者在这里捕获输出
        # 暂时使用简化方式

        # 导入agent模块并直接调用内部函数
        from src.agent import app as agent_app

        # 准备输入
        from langchain_core.messages import HumanMessage
        inputs = {"messages": [HumanMessage(content=user_input)], "retry_count": 0, "image_analysis": ""}
        config = {"configurable": {"thread_id": request.user_id}}

        # 运行代理并收集结果
        final_state = None
        for event in agent_app.stream(inputs, config=config):
            for node_name, output in event.items():
                logger.debug(f"节点 {node_name} 输出")
                final_state = output

        if final_state and "plan" in final_state:
            plan = final_state["plan"]
            status = "success"
            message = "代理生成教学方案成功"
        else:
            plan = "代理未能生成方案"
            status = "partial"
            message = "代理运行完成但未生成明确方案"

        return AgentResponse(
            status=status,
            message=message,
            plan=plan,
            session_id=request.user_id
        )

    except Exception as e:
        logger.error(f"代理查询失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"代理处理失败: {str(e)}"
        )

@app.get("/api/agent/sessions/{user_id}")
async def get_session_info(user_id: str):
    """获取指定用户的会话信息"""
    # 这里可以查询数据库获取会话状态
    # 暂时返回简单信息
    return {
        "user_id": user_id,
        "session_exists": True,
        "message": "会话信息检索功能待实现"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)