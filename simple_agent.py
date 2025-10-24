#!/usr/bin/env python3
"""
简洁的Strands Agent - 支持自定义问题和系统提示词
集成MCP工具，单一代理模式
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from strands import Agent, tool
from strands_tools import (
    calculator, current_time, image_reader
)
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient
from strands.hooks import BeforeInvocationEvent
from tools.code_interpreter import AgentCoreCodeInterpreter
from tools.browser import AgentCoreBrowser
from dotenv import load_dotenv
from botocore.config import Config


load_dotenv(dotenv_path=".env")


# 配置日志
logging.getLogger("strands").setLevel(logging.INFO)
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s", 
    handlers=[logging.StreamHandler()]
)

USE_BEDROCK=os.getenv("USE_BEDROCK")=="True"
SF_API_KEY=os.getenv("SF_API_KEY")
AWS_REGION=os.getenv("AWS_REGION")

class SimpleAgent:
    """简洁的AI代理"""
    
    def __init__(self, verbose: bool = False, use_bedrock = True):
        """
        初始化代理
        
        Args:
            model: 使用的模型名称
            verbose: 是否显示详细执行过程
        """
        if use_bedrock:
            self.model = BedrockModel(
                model_id="us.anthropic.claude-haiku-4-5-20251001-v1:0", 
                boto_client_config = Config(read_timeout=3600),
                region_name=AWS_REGION, 
                temperature=1,          
                max_tokens=30000,
                additional_request_fields={
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 4096
                        }
                    }
                )
        else:
            self.model = OpenAIModel(
                client_args={
                    "api_key": "",
                    "base_url": ""
                },
                model_id='moonshotai/Kimi-K2-Instruct-0905',
                params={"max_tokens": 30000, "temperature": 0.7}
                )

        self.verbose = verbose
        self.mcp_clients = []
        self.mcp_tools = []
        
        # built-in工具
        # agentcore_code_interpreter = AgentCoreCodeInterpreter(region="us-east-1")
        # agentcore_browser = AgentCoreBrowser(region="us-east-1")
        self.basic_tools = [
            #calculator,
            #image_reader,
            current_time,
            #agentcore_code_interpreter.code_interpreter,
            #agentcore_browser.browser
        ]
        
        # 尝试连接MCP服务器
        self._setup_mcp()
        
        print(f"Agent初始化完成")
        print(f"Model: {self.model.config['model_id']}")
        print(f"Basic Tools: {len(self.basic_tools)} 个")
        print(f"MCP Tools: {len(self.mcp_tools)} 个")
    

    def _setup_mcp(self):
        """设置MCP连接"""
        try:
            # 读取MCP配置
            if os.path.exists("mcp_config.json"):
                with open("mcp_config.json", 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 连接所有启用的服务器
                for name, server_config in config.get("mcpServers", {}).items():
                    if not server_config.get("disabled", False):
                        try:
                            print(f"🔌 连接MCP服务器: {name}")
                            
                            mcp_client = MCPClient(lambda sc=server_config: stdio_client(
                                StdioServerParameters(
                                    command=sc["command"],
                                    args=sc["args"],
                                    env=sc.get("env", {})
                                )
                            ))
                            
                            mcp_client.start()
                            tools = mcp_client.list_tools_sync()
                            
                            self.mcp_clients.append((name, mcp_client))
                            self.mcp_tools.extend(tools)
                            
                            print(f"✅ {name} 连接成功，获得 {len(tools)} 个工具")
                            
                        except Exception as e:
                            print(f"⚠️  MCP服务器 {name} 连接失败: {e}")
                            continue
                
                if self.mcp_tools:
                    print(f"🎯 总计MCP工具: {len(self.mcp_tools)} 个")
                else:
                    print("⚠️  没有成功连接任何MCP服务器")
            else:
                print("⚠️  未找到mcp_config.json，跳过MCP集成")
                
        except Exception as e:
            print(f"⚠️  MCP设置失败: {e}")
    

    def create_agent(self, system_prompt: str) -> Agent:
        """
        创建代理实例
        
        Args:
            system_prompt: 系统提示词
            
        Returns:
            配置好的Agent实例
        """
        all_tools = self.basic_tools + self.mcp_tools
        
        # 根据verbose设置选择回调处理器
        if self.verbose:
            from strands.handlers.callback_handler import PrintingCallbackHandler
            callback_handler = PrintingCallbackHandler()
        else:
            callback_handler = None
 
        agent = Agent(
            model=self.model,
            tools=all_tools,
            system_prompt=system_prompt,
            callback_handler=callback_handler
            )
        
        return agent
    

    def ask(self, question: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        向代理提问
        
        Args:
            question: 用户问题
            system_prompt: 系统提示词，如果为None则使用默认
            
        Returns:
            包含回答和元数据的字典
        """
        if system_prompt is None:
            system_prompt = '''You are a deep research assistant. When you are asked to perform a deep research task, please follow these principles:
                **Thinking Approach**:
                - Pursue fundamental insights over surface-level enumeration
                - Apply critical analysis and independent reasoning
                - Break through conventional frameworks with innovative perspectives
                **Execution Flow**:
                1. **Create a research plan first** - Define objectives, key questions, and search strategies before starting
                2. Get the current time
                3. Execute the plan using web search tools to find latest and necessary information
                4. If you have gathered sufficient information, please start compiling the research report
                5. When outputting Markdown or HTML, first get the accessible local directory, then save the report and return the file path
                **Output Standards**:
                - Prioritize quality over quantity
                - Provide original insights
                - Avoid repetitive narratives
                '''
        try:
            start_time = datetime.now()
            
            # 创建代理
            agent = self.create_agent(system_prompt)
            
            # 执行查询
            response = agent(question)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 提取响应文本 - 增强错误处理
            # 增加提取thinking文本
            thinking = ""
            try:
                if hasattr(response, 'message') and response.message:
                    if isinstance(response.message, dict) and 'content' in response.message:
                        content = response.message['content']
                        if isinstance(content, list):
                            if len(content)==1:
                                answer = content[0].get('text', '')
                            elif len(content)==2:
                                thinking = content[0]['reasoningContent']['reasoningText']["text"]
                                answer = content[1].get('text', '')
                            else:
                                answer = str(content) if content else "模型返回了空响应"
                        else:
                            answer = str(content) if content else "模型返回了空响应"
                    else:
                        answer = str(response.message)
                else:
                    answer = str(response)
                
                # 确保answer不为空
                if not answer or answer.strip() == '':
                    answer = "模型未返回有效内容"
                    
            except Exception as parse_error:
                logging.error(f"解析响应时出错: {parse_error}")
                answer = f"响应解析失败: {str(response)[:200]}"
            
            # 获取使用统计
            usage = {}
            if hasattr(response, 'metrics') and response.metrics:
                try:
                    usage = response.metrics.accumulated_usage
                except:
                    usage = {}
            
            return {
                "success": True,
                "thinking_content": thinking,
                "answer": answer,
                "duration": duration,
                "usage": usage,
                "timestamp": end_time.isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def cleanup(self):
        """清理资源"""
        for name, client in self.mcp_clients:
            try:
                client.stop(None, None, None)
                print(f"🧹 {name} MCP连接已关闭")
            except:
                pass


def interactive_mode():
    """交互模式"""
    print("\n🎯 交互模式启动")
    print("输入 'quit' 退出")
    print("输入 'prompt' 修改系统提示词")
    print("输入 'verbose' 切换详细模式")
    print("输入 'help' 查看帮助")
    print("-" * 50)
    
    # 询问是否显示详细过程
    verbose_choice = input("是否显示详细执行过程？(y/n，默认n): ").strip().lower()
    verbose = verbose_choice in ['y', 'yes', '是']
    
    agent = SimpleAgent(verbose=verbose)
    current_prompt = None  # 使用默认提示词
    
    if verbose:
        print("✅ 详细模式已启用 - 将显示工具调用和思考过程")
    else:
        print("ℹ️  简洁模式 - 只显示最终结果")
    
    try:
        while True:
            user_input = input("\n💬 你的问题: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                break
            
            if user_input.lower() == 'prompt':
                print("\n当前系统提示词:")
                if current_prompt:
                    print(current_prompt[:200] + "..." if len(current_prompt) > 200 else current_prompt)
                else:
                    print("(使用默认提示词)")
                
                new_prompt = input("\n输入新的系统提示词 (回车保持不变): ").strip()
                if new_prompt:
                    current_prompt = new_prompt
                    print("✅ 系统提示词已更新")
                continue
            
            if user_input.lower() == 'verbose':
                agent.cleanup()
                agent.verbose = not agent.verbose
                agent = SimpleAgent(verbose=agent.verbose)
                status = "启用" if agent.verbose else "禁用"
                print(f"✅ 详细模式已{status}")
                continue
            
            if user_input.lower() == 'help':
                show_help()
                continue
            
            if not user_input:
                continue
            
            print("🤖 思考中...")
            result = agent.ask(user_input, current_prompt)
            
            if result["success"]:
                print(f"\n🤖 思考:\n{result['thinking_content']}")
                print(f"\n🤖 回答:\n{result['answer']}")
                print(f"\n⏱️  耗时: {result['duration']:.2f}秒")
                if result['usage']:
                    print(f"📊 Token使用: {result['usage']}")
            else:
                print(f"\n❌ 错误: {result['error']}")
    
    except KeyboardInterrupt:
        pass
    finally:
        agent.cleanup()
        print("\n👋 再见！")


def batch_mode():
    """批处理模式"""
    print("\n📝 批处理模式")
    print("请输入你的问题和系统提示词")
    print("-" * 50)
    
    # 获取用户输入
    question = input("💬 你的问题: ").strip()
    if not question:
        print("❌ 问题不能为空")
        return
    
    print("\n📋 系统提示词 (回车使用默认):")
    system_prompt = input().strip()
    if not system_prompt:
        system_prompt = None
    
    # 询问是否显示详细过程
    verbose_choice = input("\n是否显示详细执行过程？(y/n，默认n): ").strip().lower()
    verbose = verbose_choice in ['y', 'yes', '是']
    
    # 执行查询
    agent = SimpleAgent(verbose=verbose)
    
    try:
        print("\n🤖 处理中...")
        result = agent.ask(question, system_prompt)
        
        if result["success"]:
            print(f"🤖 思考:\n{result['thinking_content']}")
            print(f"\n🤖 回答:\n{result['answer']}")
            print(f"\n📊 统计信息:")
            print(f"   耗时: {result['duration']:.2f}秒")
            if result['usage']:
                print(f"   Token使用: {result['usage']}")
        else:
            print(f"\n❌ 错误: {result['error']}")
    
    finally:
        agent.cleanup()


def main():
    """主函数"""
    print("🚀 Strands Agent")
    print("=" * 30)
    
    print("\n选择模式:")
    print("1. 交互模式 (推荐)")
    print("2. 单次问答")
    
    try:
        choice = input("\n请选择: ").strip()
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            batch_mode()
        else:
            print("无效选择，启动交互模式...")
            interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 再见！")


if __name__ == "__main__":
    main()