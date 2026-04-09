import json
import inspect
from utils.logger import get_logger
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, asdict
from enum import Enum

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import settings

# 配置日志
logger = get_logger(__name__)


class ToolType(str, Enum):
    """工具类型枚举"""
    FUNCTION = "function"
    # 未来可以支持其他类型


class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: str
    description: str
    required: bool = True


class ToolDefinition(BaseModel):
    """工具定义"""
    name: str
    description: str
    type: ToolType = ToolType.FUNCTION
    parameters: List[ToolParameter] = Field(default_factory=list)
    returns: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class Tool:
    """工具实例"""
    name: str
    func: Callable
    definition: ToolDefinition


class MCPServer:
    """MCP服务器实现"""

    def __init__(self, name: str = "SmartEdu_Agent"):
        self.name = name
        self.tools: Dict[str, Tool] = {}
        self.app = FastAPI(title=name, version=settings.app_version)
        self._setup_routes()

    def tool(self, func: Optional[Callable] = None, *, name: Optional[str] = None):
        """工具装饰器"""
        def decorator(f: Callable) -> Callable:
            tool_name = name or f.__name__
            definition = self._create_tool_definition(f, tool_name)
            self.tools[tool_name] = Tool(name=tool_name, func=f, definition=definition)

            # 添加工具端点
            self._add_tool_endpoint(tool_name, f)
            return f

        if func is None:
            return decorator
        else:
            return decorator(func)

    def _create_tool_definition(self, func: Callable, name: str) -> ToolDefinition:
        """从函数创建工具定义"""
        # 解析函数签名
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""

        # 提取参数信息
        parameters = []
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_type = str(param.annotation) if param.annotation != inspect.Parameter.empty else "any"
            param_desc = f"参数 {param_name}"

            # 从docstring中提取参数描述（简化实现）
            parameters.append(ToolParameter(
                name=param_name,
                type=param_type,
                description=param_desc,
                required=param.default == inspect.Parameter.empty
            ))

        # 提取返回类型
        return_annotation = sig.return_annotation
        returns = {"type": str(return_annotation) if return_annotation != inspect.Parameter.empty else "any"}

        # 提取描述（第一行）
        description = doc.strip().split('\n')[0] if doc else f"工具 {name}"

        return ToolDefinition(
            name=name,
            description=description,
            type=ToolType.FUNCTION,
            parameters=parameters,
            returns=returns
        )

    def _add_tool_endpoint(self, tool_name: str, func: Callable):
        """为工具添加FastAPI端点"""

        async def tool_endpoint(request: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # 提取参数
                args = request.get("arguments", {})

                # 调用工具函数
                result = func(**args)

                # 返回标准化响应
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, ensure_ascii=False) if isinstance(result, (dict, list)) else str(result)
                        }
                    ]
                }
            except Exception as e:
                logger.error(f"工具 {tool_name} 执行失败: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"工具执行失败: {str(e)}")

        # 注册端点
        self.app.post(f"/tools/{tool_name}")(tool_endpoint)

    def _setup_routes(self):
        """设置MCP标准路由"""

        @self.app.get("/")
        async def root():
            return {"name": self.name, "version": settings.app_version}

        @self.app.get("/tools")
        async def list_tools() -> Dict[str, Any]:
            """列出所有可用工具"""
            tools_list = []
            for tool in self.tools.values():
                tool_dict = asdict(tool.definition)
                tools_list.append(tool_dict)

            return {"tools": tools_list}

        @self.app.get("/health")
        async def health_check():
            """健康检查端点"""
            return {"status": "healthy", "tools_count": len(self.tools)}

        @self.app.get("/openapi.json")
        async def openapi():
            """返回OpenAPI规范"""
            return self.app.openapi()

    def run(self, host: str = None, port: int = None):
        """启动MCP服务器"""
        host = host or settings.host
        port = port or settings.mcp_server_port

        logger.info(f"启动MCP服务器: {self.name}")
        logger.info(f"服务器地址: http://{host}:{port}")
        logger.info(f"可用工具: {list(self.tools.keys())}")
        logger.info(f"API文档: http://{host}:{port}/docs")

        uvicorn.run(self.app, host=host, port=port)


# 创建全局MCP服务器实例
mcp = MCPServer("SmartEdu_Agent")


if __name__ == "__main__":
    # 直接运行此文件启动MCP服务器
    from utils.logger import get_logger
    logging.basicConfig(level=logging.INFO)
    mcp.run()