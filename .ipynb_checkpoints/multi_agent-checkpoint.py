#!/usr/bin/env python3
"""
基于Strands GraphBuilder的多Agent任务解决系统
使用GraphBuilder构建真正的图形化工作流协作架构
对应单Agent提示词的工作流程：任务分析 -> 信息收集 -> 工具执行 -> 结果分析 -> 答案格式化
"""

import os
import json
import logging
import time
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import threading
import signal

from strands import Agent
from strands.multiagent import GraphBuilder
from strands.models import BedrockModel
from strands.models.openai import OpenAIModel
from strands_tools import calculator, current_time, image_reader, file_read
from mcp import stdio_client, StdioServerParameters
from strands.tools.mcp import MCPClient

from tools.code_interpreter import AgentCoreCodeInterpreter
from tools.browser import AgentCoreBrowser
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

# 配置增强日志系统
class WorkflowFormatter(logging.Formatter):
    """自定义日志格式化器，支持工作流追踪"""
    
    def format(self, record):
        # 添加时间戳和线程信息
        record.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        record.thread_id = threading.current_thread().ident
        
        # 基础格式
        base_format = "%(timestamp)s | %(levelname)s | %(name)s | %(message)s"
        
        # 如果有工作流上下文，添加额外信息
        if hasattr(record, 'workflow_step'):
            base_format = "%(timestamp)s | %(levelname)s | [%(workflow_step)s] | %(name)s | %(message)s"
        
        formatter = logging.Formatter(base_format)
        return formatter.format(record)

# 配置日志
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(WorkflowFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 配置根日志
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", 
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)

USE_BEDROCK = os.getenv("USE_BEDROCK") == "True"
SF_API_KEY = os.getenv("SF_API_KEY")
AWS_REGION = os.getenv("AWS_REGION")

# 性能和错误处理配置
DEFAULT_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", "300"))  # 5分钟默认超时
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))


class WorkflowError(Exception):
    """工作流执行错误基类"""
    pass


class WorkflowTimeoutError(WorkflowError):
    """工作流执行超时错误"""
    pass


class AgentExecutionError(WorkflowError):
    """Agent执行错误"""
    def __init__(self, agent_name: str, message: str, original_error: Exception = None):
        self.agent_name = agent_name
        self.original_error = original_error
        super().__init__(f"Agent '{agent_name}' execution failed: {message}")


class WorkflowBuildError(WorkflowError):
    """工作流构建错误"""
    pass


class DataPassingError(WorkflowError):
    """数据传递错误"""
    pass


# 移除限流控制 - Claude 3.5 没有限速问题


@contextmanager
def timeout_context(seconds: int):
    """超时上下文管理器"""
    def timeout_handler(signum, frame):
        raise WorkflowTimeoutError(f"Operation timed out after {seconds} seconds")
    
    # 设置信号处理器
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        # 恢复原始处理器
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """开始计时"""
        self.start_times[operation] = time.time()
        logger.info(f"🕐 开始执行: {operation}")
    
    def end_timer(self, operation: str):
        """结束计时"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation] = duration
            logger.info(f"⏱️ 完成执行: {operation} (耗时: {duration:.2f}秒)")
            del self.start_times[operation]
            return duration
        return None
    
    def get_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.metrics.copy()


class DataValidator:
    """数据验证器，优化Agent间数据传递"""
    
    @staticmethod
    def validate_task_input(task: str) -> str:
        """验证任务输入"""
        if not task or not isinstance(task, str):
            raise DataPassingError("Task input must be a non-empty string")
        
        # 清理和标准化输入
        cleaned_task = task.strip()
        if len(cleaned_task) > 10000:  # 限制输入长度
            logger.warning("⚠️ 任务输入过长，将被截断")
            cleaned_task = cleaned_task[:10000] + "..."
        
        return cleaned_task
    
    @staticmethod
    def validate_agent_output(output: Any, agent_name: str) -> str:
        """验证Agent输出"""
        if output is None:
            raise DataPassingError(f"Agent '{agent_name}' returned None output")
        
        # 转换为字符串
        if isinstance(output, str):
            result = output
        else:
            result = str(output)
        
        # 对于workflow结果，优先保留<answer>标签内容
        if agent_name == "workflow" and "<answer>" in result:
            # 提取<answer>标签内容
            import re
            answer_match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                logger.info(f"✅ 提取到最终答案: {answer_content}")
                return f"<answer>{answer_content}</answer>"
        
        # 验证输出长度
        if len(result) > 50000:  # 限制输出长度
            logger.warning(f"⚠️ Agent '{agent_name}' 输出过长，将被截断")
            
            # 如果包含<answer>标签，优先保留答案部分
            if "<answer>" in result:
                import re
                answer_match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 保留答案和部分上下文
                    context_start = result[:10000]
                    context_end = result[-5000:]
                    result = f"{context_start}\n\n[中间内容已截断]\n\n{context_end}\n\n<answer>{answer_content}</answer>"
                    logger.info(f"✅ 保留最终答案: {answer_content}")
                else:
                    result = result[:50000] + "...[truncated]"
            else:
                result = result[:50000] + "...[truncated]"
        
        return result
    
    @staticmethod
    def optimize_data_transfer(data: str) -> str:
        """优化数据传递"""
        # 移除多余的空白字符
        optimized = ' '.join(data.split())
        
        # 如果数据过大，进行压缩摘要
        if len(optimized) > 20000:
            # 保留开头和结尾，中间用摘要替代
            start = optimized[:8000]
            end = optimized[-8000:]
            middle_summary = f"\n[中间内容摘要: {len(optimized) - 16000} 字符]\n"
            optimized = start + middle_summary + end
            logger.info(f"📦 数据传递优化: 原长度 {len(data)} -> 优化后长度 {len(optimized)}")
        
        return optimized





def extract_final_answer(result: str) -> str:
    """
    从结果中提取最终答案
    
    Args:
        result: 工作流执行结果
        
    Returns:
        提取的最终答案，如果没有找到则返回空字符串
    """
    if not result:
        return ""
    
    # 使用正则表达式提取<answer>标签内容
    import re
    answer_match = re.search(r'<answer>(.*?)</answer>', result, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1).strip()
        logger.info(f"🎯 成功提取最终答案: {answer_content}")
        return answer_content
    
    # 如果没有找到<answer>标签，返回空字符串
    logger.warning("⚠️ 未找到<answer>标签，无法提取最终答案")
    return ""


def setup_tools() -> List:
    """设置共享工具集"""
    tools = []
    
    # 基础工具
    try:
        # Strands内置工具
        tools.extend([calculator, current_time, image_reader,file_read])
        logging.info("✅ 基础工具加载成功: calculator, current_time, image_reader, file_read")
        
        # AgentCore工具
        try:
            agentcore_code_interpreter = AgentCoreCodeInterpreter(region="us-east-1")
            tools.append(agentcore_code_interpreter.code_interpreter)
            logging.info("✅ AgentCore代码解释器加载成功")
        except Exception as e:
            logging.warning(f"⚠️ AgentCore代码解释器加载失败: {e}")
        
        try:
            agentcore_browser = AgentCoreBrowser(region="us-east-1")
            tools.append(agentcore_browser.browser)
            logging.info("✅ AgentCore浏览器加载成功")
        except Exception as e:
            logging.warning(f"⚠️ AgentCore浏览器加载失败: {e}")
            
    except Exception as e:
        logging.error(f"❌ 基础工具加载失败: {e}")
    
    # MCP工具
    mcp_tools = setup_mcp_tools()
    tools.extend(mcp_tools)
    
    logging.info(f"🔧 工具集设置完成，共加载 {len(tools)} 个工具")
    return tools


def setup_mcp_tools() -> List:
    """
    设置MCP工具
    
    要启用MCP工具，请在mcp_config.json中设置disabled为false：
    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
          "disabled": false
        }
      }
    }
    """
    mcp_tools = []
    
    if not os.path.exists("mcp_config.json"):
        logging.info("📝 未找到mcp_config.json文件，跳过MCP工具加载")
        return mcp_tools
    
    try:
        with open("mcp_config.json", 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        mcp_servers = config.get("mcpServers", {})
        if not mcp_servers:
            logging.info("📝 MCP配置中没有服务器定义")
            return mcp_tools
        
        for name, server_config in mcp_servers.items():
            # 检查是否被禁用 (支持disabled: 1, true, "true"等格式)
            disabled = server_config.get("disabled", False)
            if disabled in [1, True, "true", "True", "1"]:
                logging.info(f"⏭️ MCP服务器 {name} 已禁用，跳过")
                continue
            
            try:
                # 验证必要的配置项
                if "command" not in server_config or "args" not in server_config:
                    logging.warning(f"⚠️ MCP服务器 {name} 配置不完整，缺少command或args")
                    continue
                
                # 创建MCP客户端
                mcp_client = MCPClient(lambda sc=server_config: stdio_client(
                    StdioServerParameters(
                        command=sc["command"],
                        args=sc["args"],
                        env=sc.get("env", {})
                    )
                ))
                
                # 启动客户端并获取工具
                mcp_client.start()
                tools = mcp_client.list_tools_sync()
                
                if tools:
                    mcp_tools.extend(tools)
                    logging.info(f"✅ MCP服务器 {name} 连接成功，获得 {len(tools)} 个工具")
                    
                    # 记录工具名称
                    tool_names = []
                    for tool in tools:
                        if hasattr(tool, 'name'):
                            tool_names.append(tool.name)
                        elif hasattr(tool, 'tool_name'):
                            tool_names.append(tool.tool_name)
                        elif hasattr(tool, '__name__'):
                            tool_names.append(tool.__name__)
                        else:
                            tool_names.append(str(type(tool).__name__))
                    logging.info(f"   工具列表: {', '.join(tool_names)}")
                else:
                    logging.info(f"📝 MCP服务器 {name} 连接成功但没有可用工具")
                    
            except Exception as e:
                logging.warning(f"⚠️ MCP服务器 {name} 连接失败: {e}")
                continue
                
    except json.JSONDecodeError as e:
        logging.error(f"❌ MCP配置文件格式错误: {e}")
    except Exception as e:
        logging.error(f"❌ MCP设置失败: {e}")
    
    if mcp_tools:
        logging.info(f"🔧 MCP工具加载完成，共 {len(mcp_tools)} 个工具")
    else:
        logging.info("📝 没有加载任何MCP工具")
    
    return mcp_tools


def test_mcp_compatibility() -> bool:
    """测试MCP工具兼容性"""
    try:
        # 检查MCP相关依赖
        import mcp
        from mcp import stdio_client, StdioServerParameters
        from strands.tools.mcp import MCPClient
        
        logging.info("✅ MCP依赖检查通过")
        return True
        
    except ImportError as e:
        logging.error(f"❌ MCP依赖缺失: {e}")
        logging.info("💡 请安装MCP依赖: pip install mcp[cli]")
        return False
    except Exception as e:
        logging.error(f"❌ MCP兼容性测试失败: {e}")
        return False


def verify_agent_tools(agent: Agent, agent_name: str) -> bool:
    """验证Agent是否能访问必要的工具"""
    try:
        # 检查tool_names属性
        if not hasattr(agent, 'tool_names') or not agent.tool_names:
            logging.warning(f"⚠️ {agent_name} 没有配置任何工具")
            return False
        
        tool_count = len(agent.tool_names)
        logging.info(f"✅ {agent_name} 配置了 {tool_count} 个工具")
        
        # 记录工具名称
        logging.info(f"   {agent_name} 工具列表: {', '.join(agent.tool_names)}")
        
        # 验证必要的基础工具
        required_tools = ['calculator', 'current_time']
        missing_tools = [tool for tool in required_tools if tool not in agent.tool_names]
        
        if missing_tools:
            logging.warning(f"⚠️ {agent_name} 缺少必要工具: {', '.join(missing_tools)}")
        
        # 检查工具注册表
        if hasattr(agent, 'tool_registry') and hasattr(agent.tool_registry, 'tools'):
            registry_count = len(agent.tool_registry.tools)
            logging.info(f"   {agent_name} 工具注册表中有 {registry_count} 个工具")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ 验证 {agent_name} 工具时出错: {e}")
        return False


def create_model(use_bedrock: bool = USE_BEDROCK):
    """创建模型实例"""
    if use_bedrock:
        return BedrockModel(
            model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            region_name=AWS_REGION,
            temperature=0.7,
            max_tokens=20000,
        )
    else:
        return OpenAIModel(
            client_args={
                "api_key": SF_API_KEY,
                "base_url": "https://api.siliconflow.cn/v1"
            },
            model_id="zai-org/GLM-4.5V",
            params={"max_tokens": 4096, "temperature": 0.7}
        )


def create_task_analyzer_agent(tools: List = None) -> Agent:
    """创建任务分析Agent - 对应单agent的任务分析阶段"""
    system_prompt = """你是一个任务分析专家，负责分析用户提出的任务并制定执行计划。

你的职责：
1. **任务分析**: 深入理解用户的任务需求，识别任务的复杂度和类型
2. **步骤拆解**: 将复杂任务拆解为多个可执行的子任务
3. **计划制定**: 提出一个由多步骤元组（子任务、目标、操作）组成的完整计划
4. **工具识别**: 确定每个子任务需要使用的工具类型

输出格式：
```
任务类型: [信息查询/计算分析/文件处理/综合任务]
复杂度: [简单/中等/复杂]
执行计划:
1. 子任务1 - 目标: xxx - 操作: xxx - 工具: xxx
2. 子任务2 - 目标: xxx - 操作: xxx - 工具: xxx
...
预期结果格式: [数字/字符串/列表/特定格式]
```

注意：不要试图一次性完成所有工作，专注于分析和规划。
"""
    
    if tools is None:
        tools = setup_tools()
    
    return Agent(
        name="task_analyzer",
        model=create_model(),
        system_prompt=system_prompt,
        tools=tools
    )


def create_information_collector_agent(tools: List = None) -> Agent:
    """创建信息收集Agent - 对应单agent的信息收集阶段"""
    system_prompt = """你是一个信息收集专家，负责根据任务计划收集必要的信息。

你的能力：
1. **文件信息收集**: 从提供的文件中提取相关信息
2. **网络搜索**: 使用浏览器工具搜索广泛的信息
3. **多源验证**: 从不同来源收集信息进行交叉验证
4. **结构化整理**: 将收集的信息按照任务需求进行分类整理

工作原则：
- 根据任务分析结果，有针对性地收集信息
- 优先使用可靠和权威的信息源
- 记录信息来源和获取时间
- 对信息进行初步的相关性筛选

输出格式：
```
信息收集结果:
1. 来源: [文件/网络/工具] - 内容: xxx - 相关性: [高/中/低]
2. 来源: [文件/网络/工具] - 内容: xxx - 相关性: [高/中/低]
...
收集状态: [完成/部分完成/需要更多信息]
```

注意：专注于信息收集，不要进行深度分析。
"""
    
    if tools is None:
        tools = setup_tools()
    
    return Agent(
        name="information_collector",
        model=create_model(),
        system_prompt=system_prompt,
        tools=tools
    )


def create_tool_executor_agent(tools: List = None) -> Agent:
    """创建工具执行Agent - 对应单agent的工具选择和执行阶段"""
    system_prompt = """你是一个工具执行专家，负责根据任务需求选择合适的工具并执行操作。

你的专长：
1. **工具选择**: 根据子任务的目标和操作选择最合适的工具
2. **精确执行**: 使用工具执行具体的操作（计算、搜索、代码运行等）
3. **结果处理**: 处理工具执行的结果，提取有用信息
4. **错误处理**: 当工具执行失败时，尝试其他方法或工具

可用工具类型：
- 计算器：数学计算和公式求解
- 代码解释器：数据处理、分析、可视化
- 浏览器：网络搜索、信息查询
- 时间工具：获取当前时间和日期
- 图像工具：图像分析和处理
- MCP工具：各种专业工具

工作原则：
- 每次只使用一个工具
- 仔细分析工具执行结果
- 如果一种方法找不到答案，尝试另一种方法
- 记录执行过程和结果

输出格式：
```
工具执行结果:
使用工具: [工具名称]
执行操作: [具体操作描述]
执行结果: [工具返回的结果]
执行结果: [工具返回的结果]
结果分析: [对结果的解释和分析]
状态: [成功/失败/需要重试]
```
"""
    
    if tools is None:
        tools = setup_tools()
    
    return Agent(
        name="tool_executor",
        model=create_model(),
        system_prompt=system_prompt,
        tools=tools
    )


def create_result_analyzer_agent(tools: List = None) -> Agent:
    """创建结果分析Agent - 对应单agent的结果分析阶段"""
    system_prompt = """你是一个结果分析专家，负责分析各个子任务的执行结果并判断任务完成状态。

你的职责：
1. **结果整合**: 收集和整理所有子任务的执行结果
2. **完成度评估**: 判断原始任务是否已经解决
3. **质量检查**: 验证结果的正确性和完整性
4. **缺口识别**: 识别还需要补充的信息或步骤

分析维度：
- 结果完整性：是否回答了用户的原始问题
- 结果准确性：信息是否正确可靠
- 格式符合性：是否符合要求的输出格式
- 逻辑一致性：各部分结果是否逻辑一致

输出格式：
```
结果分析:
任务完成状态: [已完成/部分完成/未完成]
完成度评估: [百分比]
质量评分: [1-10分]
主要发现: [关键结果摘要]
存在问题: [发现的问题或缺口]
下一步建议: [如果未完成，建议下一步操作]
```

注意：如果任务已解决，准备最终答案；如果未解决，提供推理和下一步建议。
"""
    
    if tools is None:
        tools = setup_tools()
    
    return Agent(
        name="result_analyzer",
        model=create_model(),
        system_prompt=system_prompt,
        tools=tools
    )


def create_answer_formatter_agent(tools: List = None) -> Agent:
    """创建答案格式化Agent - 对应单agent的最终答案阶段"""
    system_prompt = """你是一个答案格式化专家，负责将分析结果格式化为符合要求的最终答案。

你的职责：
1. **答案提取**: 从分析结果中提取核心答案
2. **格式规范**: 严格按照要求的格式输出答案
3. **质量保证**: 确保答案准确、简洁、符合规范

格式要求（严格遵守）：
- **数字**: 不使用逗号，不使用$或%符号（除非特别说明）
- **字符串**: 不使用冠词，不使用缩写，数字写为阿拉伯数字
- **列表**: 逗号分隔，应用上述规则
- **特定格式**: 严格符合要求（如日期格式、数字格式等）

格式化规则示例：
- `四舍五入到千位` 表示 `93784` → `93`
- `年月中的月份` 表示 `2020-04-30` → `April in 2020`

输出要求：
- 必须使用 `<answer></answer>` 标签包裹最终答案
- 答案应该是数字、最少的词、或逗号分隔的列表
- 绝不能在没有 `<answer></answer>` 标签的情况下输出最终答案

示例输出：
1. <answer>apple tree</answer>
2. <answer>3, 4, 5</answer>
3. <answer>42</answer>
"""
    
    if tools is None:
        tools = setup_tools()
    
    return Agent(
        name="answer_formatter",
        model=create_model(),
        system_prompt=system_prompt,
        tools=tools
    )


class StrandsMultiAgentTaskSolver:
    """基于Strands GraphBuilder的多Agent任务解决系统 - 增强版本"""
    
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        """
        初始化多Agent系统
        
        Args:
            timeout: 工作流执行超时时间（秒）
        """
        self.timeout = timeout
        self.performance_monitor = PerformanceMonitor()
        self.data_validator = DataValidator()
        self.workflow = None
        self.agents = {}
        
        logger.info("🚀 开始初始化Strands GraphBuilder多Agent任务解决系统")
        
        try:
            self._initialize_system()
            logger.info("✅ Strands GraphBuilder多Agent任务解决系统初始化完成")
        except Exception as e:
            logger.error(f"❌ 系统初始化失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise WorkflowBuildError(f"System initialization failed: {e}")
    
    def _initialize_system(self):
        """初始化系统组件"""
        # 测试MCP兼容性
        self.performance_monitor.start_timer("mcp_compatibility_test")
        logger.info("🔍 测试MCP兼容性...")
        try:
            test_mcp_compatibility()
            self.performance_monitor.end_timer("mcp_compatibility_test")
        except Exception as e:
            logger.warning(f"⚠️ MCP兼容性测试失败: {e}")
            self.performance_monitor.end_timer("mcp_compatibility_test")
        
        # 设置工具集
        self.performance_monitor.start_timer("tools_setup")
        logger.info("🔧 设置工具集...")
        try:
            tools = setup_tools()
            logger.info(f"🔧 工具验证完成，共 {len(tools)} 个可用工具")
            self.performance_monitor.end_timer("tools_setup")
        except Exception as e:
            logger.error(f"❌ 工具设置失败: {e}")
            self.performance_monitor.end_timer("tools_setup")
            raise WorkflowBuildError(f"Tools setup failed: {e}")
        
        # 创建所有Agent
        self.performance_monitor.start_timer("agents_creation")
        logger.info("👥 创建Agent...")
        try:
            self._create_agents(tools)
            self.performance_monitor.end_timer("agents_creation")
        except Exception as e:
            logger.error(f"❌ Agent创建失败: {e}")
            self.performance_monitor.end_timer("agents_creation")
            raise WorkflowBuildError(f"Agents creation failed: {e}")
        
        # 验证Agent工具访问
        self.performance_monitor.start_timer("agents_validation")
        logger.info("🔍 验证Agent工具访问...")
        try:
            self._validate_agents()
            self.performance_monitor.end_timer("agents_validation")
        except Exception as e:
            logger.warning(f"⚠️ Agent验证失败: {e}")
            self.performance_monitor.end_timer("agents_validation")
        
        # 构建GraphBuilder工作流
        self.performance_monitor.start_timer("workflow_build")
        logger.info("🔗 构建GraphBuilder工作流...")
        try:
            self.workflow = self._build_workflow()
            self.performance_monitor.end_timer("workflow_build")
        except Exception as e:
            logger.error(f"❌ 工作流构建失败: {e}")
            self.performance_monitor.end_timer("workflow_build")
            raise WorkflowBuildError(f"Workflow build failed: {e}")
    
    def _create_agents(self, tools: List):
        """创建所有Agent"""
        agent_creators = [
            ("task_analyzer", create_task_analyzer_agent),
            ("information_collector", create_information_collector_agent),
            ("tool_executor", create_tool_executor_agent),
            ("result_analyzer", create_result_analyzer_agent),
            ("answer_formatter", create_answer_formatter_agent)
        ]
        
        for agent_name, creator_func in agent_creators:
            try:
                agent = creator_func(tools)
                self.agents[agent_name] = agent
                logger.info(f"✅ 创建Agent: {agent_name}")
            except Exception as e:
                logger.error(f"❌ 创建Agent失败 {agent_name}: {e}")
                raise AgentExecutionError(agent_name, f"Creation failed: {e}", e)
    
    def _validate_agents(self):
        """验证所有Agent"""
        agent_names = [
            ("task_analyzer", "TaskAnalyzer"),
            ("information_collector", "InformationCollector"),
            ("tool_executor", "ToolExecutor"),
            ("result_analyzer", "ResultAnalyzer"),
            ("answer_formatter", "AnswerFormatter")
        ]
        
        all_agents_valid = True
        for agent_key, display_name in agent_names:
            if agent_key in self.agents:
                if not verify_agent_tools(self.agents[agent_key], display_name):
                    all_agents_valid = False
            else:
                logger.error(f"❌ Agent {agent_key} 未找到")
                all_agents_valid = False
        
        if not all_agents_valid:
            logger.warning("⚠️ 部分Agent工具验证失败，但系统将继续运行")
    
    def _build_workflow(self):
        """构建GraphBuilder工作流 - 增强错误处理版本"""
        try:
            # 验证所有必需的Agent都已创建
            required_agents = ["task_analyzer", "information_collector", "tool_executor", 
                             "result_analyzer", "answer_formatter"]
            
            for agent_name in required_agents:
                if agent_name not in self.agents:
                    raise WorkflowBuildError(f"Required agent '{agent_name}' not found")
            
            # 创建GraphBuilder实例
            builder = GraphBuilder()
            logger.info("📊 创建GraphBuilder实例")
            
            # 使用builder.add_node(agent, "node_name")添加所有Agent节点
            for agent_name in required_agents:
                try:
                    builder.add_node(self.agents[agent_name], agent_name)
                    logger.info(f"➕ 添加节点: {agent_name}")
                except Exception as e:
                    raise WorkflowBuildError(f"Failed to add node '{agent_name}': {e}")
            
            # 使用builder.add_edge()连接Agent节点
            # 实现线性工作流：task_analyzer → information_collector → tool_executor → result_analyzer → answer_formatter
            edges = [
                ("task_analyzer", "information_collector"),
                ("information_collector", "tool_executor"),
                ("tool_executor", "result_analyzer"),
                ("result_analyzer", "answer_formatter")
            ]
            
            for from_node, to_node in edges:
                try:
                    builder.add_edge(from_node, to_node)
                    logger.info(f"🔗 添加边: {from_node} → {to_node}")
                except Exception as e:
                    raise WorkflowBuildError(f"Failed to add edge '{from_node}' -> '{to_node}': {e}")
            
            # 使用builder.set_entry_point("task_analyzer")设置入口
            try:
                builder.set_entry_point("task_analyzer")
                logger.info("🚪 设置入口点: task_analyzer")
            except Exception as e:
                raise WorkflowBuildError(f"Failed to set entry point: {e}")
            
            # 调用builder.build()构建最终工作流
            try:
                workflow = builder.build()
                logger.info("🏗️ GraphBuilder工作流构建成功")
                return workflow
            except Exception as e:
                raise WorkflowBuildError(f"Failed to build workflow: {e}")
                
        except Exception as e:
            logger.error(f"❌ 工作流构建失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            raise
    
    def solve_task(self, task: str, requirements: str = "") -> str:
        """
        执行任务解决 - 增强版本，包含超时和错误处理
        
        Args:
            task: 用户任务
            requirements: 用户需求描述
            
        Returns:
            工作流的最终结果
            
        Raises:
            WorkflowTimeoutError: 工作流执行超时
            AgentExecutionError: Agent执行错误
            DataPassingError: 数据传递错误
        """
        if not self.workflow:
            raise WorkflowError("Workflow not initialized")
        
        # 验证和优化输入数据
        try:
            validated_task = self.data_validator.validate_task_input(task)
            logger.info(f"🚀 开始GraphBuilder工作流执行: {validated_task[:100]}...")
        except Exception as e:
            raise DataPassingError(f"Task input validation failed: {e}")
        
        # 构建完整的任务输入
        task_input = f"""用户任务: {validated_task}
用户需求: {requirements or "解决用户提出的任务"}

请按照工作流程完成这个任务。"""
        
        # 优化数据传递
        optimized_input = self.data_validator.optimize_data_transfer(task_input)
        
        # 开始性能监控
        self.performance_monitor.start_timer("workflow_execution")
        
        # 执行工作流，带超时和重试机制
        result = self._execute_workflow_with_retry(optimized_input)
        
        # 结束性能监控
        execution_time = self.performance_monitor.end_timer("workflow_execution")
        
        # 记录性能指标
        metrics = self.performance_monitor.get_metrics()
        logger.info(f"📊 工作流执行完成，总耗时: {execution_time:.2f}秒")
        logger.info(f"📈 性能指标: {metrics}")
        
        return result
    
    def _execute_workflow_with_retry(self, task_input: str, max_retries: int = MAX_RETRIES) -> str:
        """
        带重试机制的工作流执行
        
        Args:
            task_input: 任务输入
            max_retries: 最大重试次数
            
        Returns:
            工作流执行结果
        """
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"🔄 重试执行工作流 (第 {attempt}/{max_retries} 次)")
                    time.sleep(RETRY_DELAY * attempt)  # 指数退避
                
                # 使用超时上下文执行工作流
                with timeout_context(self.timeout):
                    result = self.workflow(task_input)
                
                # 验证输出
                validated_result = self.data_validator.validate_agent_output(result, "workflow")
                logger.info("✅ 工作流执行成功")
                return validated_result
                
            except WorkflowTimeoutError as e:
                last_error = e
                logger.error(f"⏰ 工作流执行超时 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                if attempt == max_retries:
                    break
                    
            except Exception as e:
                last_error = e
                logger.error(f"❌ 工作流执行失败 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                
                # 某些错误不需要重试
                if isinstance(e, (DataPassingError, WorkflowBuildError)):
                    break
                    
                if attempt == max_retries:
                    break
        
        # 所有重试都失败了
        error_msg = f"Workflow execution failed after {max_retries + 1} attempts"
        if last_error:
            error_msg += f". Last error: {last_error}"
        
        logger.error(f"💥 {error_msg}")
        raise WorkflowError(error_msg)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        return self.performance_monitor.get_metrics()
    
    def reset_performance_metrics(self):
        """重置性能指标"""
        self.performance_monitor = PerformanceMonitor()
        logger.info("📊 性能指标已重置")
    



def interactive_mode():
    """交互模式 - 增强错误处理版本"""
    print("\n🎯 Strands GraphBuilder多Agent任务解决系统")
    print("=" * 50)
    print("基于GraphBuilder的图形化工作流协作架构")
    print("输入 'quit' 退出")
    print("输入 'help' 查看帮助")
    print("输入 'metrics' 查看性能指标")
    print("-" * 50)
    
    # 初始化系统
    try:
        system = StrandsMultiAgentTaskSolver()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查配置和依赖是否正确安装")
        return
    
    try:
        while True:
            task = input("\n🎯 请输入任务: ").strip()
            
            if task.lower() in ['quit', 'exit', '退出']:
                break
            
            if task.lower() == 'help':
                print("\n📖 使用帮助:")
                print("- 输入任何任务进行智能解决")
                print("- 系统使用GraphBuilder构建图形化工作流")
                print("- 5个Agent通过节点和边连接协作工作")
                print("- 包含任务分析、信息收集、工具执行、结果分析、答案格式化五个阶段")
                print("- 支持迭代执行和条件分支，直到任务完成")
                print("- 最终提供格式化的答案")
                print("- 内置超时机制和错误重试")
                print("- 输入 'metrics' 查看性能指标")
                continue
            
            if task.lower() == 'metrics':
                metrics = system.get_performance_metrics()
                if metrics:
                    print("\n📊 性能指标:")
                    for operation, duration in metrics.items():
                        print(f"  {operation}: {duration:.2f}秒")
                else:
                    print("\n📊 暂无性能数据")
                continue
            
            if not task:
                continue
            
            # 获取用户需求
            requirements = input("📋 特殊需求 (可选，回车跳过): ").strip()
            
            print("\n🤖 GraphBuilder工作流执行中...")
            print("📍 这可能需要几分钟时间，请耐心等待...")
            print("⏰ 超时时间: {}秒".format(system.timeout))
            
            # 执行任务解决
            try:
                result = system.solve_task(task, requirements)
                
                print(f"\n✅ 任务解决完成!")
                
                # 提取并显示最终答案
                final_answer = extract_final_answer(result)
                if final_answer:
                    print(f"\n📝 最终答案:")
                    print("=" * 60)
                    print(final_answer)
                    print("=" * 60)
                else:
                    print(f"\n📝 完整结果:")
                    print("-" * 50)
                    print(result)
                    print("-" * 50)
                
                # 显示性能指标
                metrics = system.get_performance_metrics()
                if metrics:
                    print(f"\n📊 本次执行性能:")
                    total_time = metrics.get('workflow_execution', 0)
                    print(f"  总执行时间: {total_time:.2f}秒")
                
            except WorkflowTimeoutError as e:
                print(f"\n⏰ 任务执行超时: {e}")
                print("💡 建议: 尝试简化任务或增加超时时间")
                
            except AgentExecutionError as e:
                print(f"\n🤖 Agent执行错误: {e}")
                print(f"💡 问题Agent: {e.agent_name}")
                
            except DataPassingError as e:
                print(f"\n📦 数据传递错误: {e}")
                print("💡 建议: 检查输入格式或简化任务描述")
                
            except WorkflowError as e:
                print(f"\n🔧 工作流错误: {e}")
                print("💡 建议: 重启系统或检查配置")
                
            except Exception as e:
                print(f"\n❌ 未知错误: {e}")
                logger.error(f"Unexpected error in interactive mode: {traceback.format_exc()}")
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
    finally:
        print("\n👋 再见！")


def batch_mode():
    """批处理模式 - 增强错误处理版本"""
    print("\n📝 GraphBuilder批处理模式")
    print("-" * 50)
    
    task = input("🎯 请输入任务: ").strip()
    if not task:
        print("❌ 任务不能为空")
        return
    
    requirements = input("📋 特殊需求 (可选): ").strip()
    
    # 初始化并执行
    try:
        system = StrandsMultiAgentTaskSolver()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    print("\n🤖 执行GraphBuilder工作流...")
    print(f"⏰ 超时时间: {system.timeout}秒")
    
    try:
        result = system.solve_task(task, requirements)
        
        print(f"\n✅ 任务解决完成!")
        
        # 提取并显示最终答案
        final_answer = extract_final_answer(result)
        if final_answer:
            print(f"\n📝 最终答案:")
            print("=" * 60)
            print(final_answer)
            print("=" * 60)
        else:
            print(f"\n📝 完整结果:")
            print("=" * 60)
            print(result)
        print("=" * 60)
        
        # 显示性能指标
        metrics = system.get_performance_metrics()
        if metrics:
            print(f"\n📊 执行性能:")
            for operation, duration in metrics.items():
                print(f"  {operation}: {duration:.2f}秒")
                
    except WorkflowTimeoutError as e:
        print(f"\n⏰ 任务执行超时: {e}")
        print("💡 建议: 尝试简化任务或设置更长的超时时间")
        
    except AgentExecutionError as e:
        print(f"\n🤖 Agent执行错误: {e}")
        print(f"💡 问题Agent: {e.agent_name}")
        
    except DataPassingError as e:
        print(f"\n📦 数据传递错误: {e}")
        print("💡 建议: 检查输入格式")
        
    except WorkflowError as e:
        print(f"\n🔧 工作流错误: {e}")
        
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        logger.error(f"Unexpected error in batch mode: {traceback.format_exc()}")


def print_system_info():
    """打印系统信息和配置"""
    print("🚀 Strands GraphBuilder多Agent任务解决系统")
    print("=" * 40)
    print("基于Strands GraphBuilder的图形化工作流多Agent协作架构")
    print("包含任务分析、信息收集、工具执行、结果分析、答案格式化五个Agent节点")
    print("\n🔧 系统配置:")
    print(f"  超时时间: {DEFAULT_TIMEOUT}秒")
    print(f"  最大重试次数: {MAX_RETRIES}")
    print(f"  重试延迟: {RETRY_DELAY}秒")
    print(f"  使用Bedrock: {USE_BEDROCK}")
    print("\n💡 环境变量配置:")
    print("  WORKFLOW_TIMEOUT - 工作流超时时间(秒)")
    print("  MAX_RETRIES - 最大重试次数")
    print("  RETRY_DELAY - 重试延迟(秒)")


def main():
    """主函数 - 增强版本"""
    print_system_info()
    
    print("\n选择模式:")
    print("1. 交互模式 (推荐)")
    print("2. 单次任务")
    print("3. 系统测试")
    
    try:
        choice = input("\n请选择 (1-3): ").strip()
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            batch_mode()
        elif choice == "3":
            test_system()
        else:
            print("无效选择，启动交互模式...")
            interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 再见！")


def test_system():
    """系统测试模式"""
    print("\n🧪 系统测试模式")
    print("-" * 50)
    
    try:
        print("🔍 初始化系统...")
        system = StrandsMultiAgentTaskSolver()
        
        print("✅ 系统初始化成功")
        
        # 简单测试
        print("\n🧮 执行简单测试: 1+1")
        result = system.solve_task("计算1+1等于多少")
        print(f"测试结果: {result}")
        
        # 显示性能指标
        metrics = system.get_performance_metrics()
        print(f"\n📊 性能指标: {metrics}")
        
        print("\n✅ 系统测试完成")
        
    except Exception as e:
        print(f"\n❌ 系统测试失败: {e}")
        logger.error(f"System test failed: {traceback.format_exc()}")


if __name__ == "__main__":
    main()