# -*- coding: utf-8 -*-
# main.py
import multiprocessing
import time
from .agent import run_agent_with_persistence

def start_server():
    """启动 MCP 工具服务器"""
    print("🚀 [Server] 正在启动 SmartEdu_Agent 工具服务...")
    # 模拟服务器启动
    print("Mock MCP server running...")

def start_agent():
    """模拟 Agent 运行及断点续传演示"""
    time.sleep(2) # 等待服务器完全启动
    print("🧠 [Agent] 决策大脑已就绪。")
    
    user_id = "tong_yao_001" # 模拟用户ID
    
    print("\n--- 第一轮对话：学生提交作文 ---")
    run_agent_with_persistence(user_id, "老师，我写了一篇关于‘光合作用’的作文，请纠错。")
    
    print("\n--- 模拟：用户掉线/程序重启 ---")
    time.sleep(1)
    
    print("\n--- 第二轮对话：恢复状态并继续 ---")
    # 只要 thread_id 不变，Agent 依然记得之前的 plan 和 review_feedback
    run_agent_with_persistence(user_id, "刚才没看清，请再结合教学大纲详细解释一下。")
    
    print("\n--- 第三轮对话：测试VLM功能 ---")
    # 测试VLM功能
    run_agent_with_persistence(user_id, "老师，我上传了一张光合作用的图片，请结合图片给我讲解一下。")

if __name__ == "__main__":
    # 使用多进程同时运行
    server_proc = multiprocessing.Process(target=start_server)
    agent_proc = multiprocessing.Process(target=start_agent)
    
    server_proc.start()
    agent_proc.start()
    
    agent_proc.join()
    server_proc.terminate() # Agent 运行完后关闭 Server