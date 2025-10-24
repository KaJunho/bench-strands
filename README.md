# Strands Agent

基于 Strands Agents SDK 的智能 AI 代理，支持 AWS Bedrock、MCP 工具集成、代码解释器和浏览器自动化。

## 特性

- 支持 AWS Bedrock 和 OpenAI 兼容模型
- Claude Extended Thinking 思考模式
- MCP 工具集成（文件系统、网络搜索）
- AWS AgentCore 工具（代码解释器、浏览器）
- 自定义系统提示词

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 配置

```bash
cp .env.example .env
# 编辑 .env 填入 AWS 凭证
```

必需环境变量：
```bash
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
USE_BEDROCK=True
```

### 运行

```bash
python simple_agent.py
```

## 项目结构

```
├── simple_agent.py              # 主程序
├── mcp_config.json              # MCP 配置
├── .env                         # 环境变量
└── tools/                       # 自定义工具
    ├── browser/                 # 浏览器工具
    └── code_interpreter/        # 代码解释器
```

## 可用工具

### 内置工具
- `current_time` - 获取当前时间
- `code_interpreter` - Python/JS/TS 代码执行
- `browser` - 网页浏览和交互

### MCP 工具
- `filesystem` - 文件系统操作
- `tavily-websearch` - 网络搜索
- `exa` - 高级搜索（需配置 API Key）

在 `simple_agent.py` 中取消注释启用工具，在 `mcp_config.json` 中配置 MCP 服务器。

## 使用模式

启动后选择：
1. 交互模式 - 持续对话
2. 单次问答 - 快速查询（推荐）

### 交互命令
- `quit/exit` - 退出程序
- `prompt` - 修改系统提示词
- `verbose` - 切换详细/简洁模式
- `help` - 显示帮助

### 详细模式

启动时输入 `y` 启用详细模式，可查看工具调用和 Claude Extended Thinking 思考过程。

## MCP 配置

编辑 `mcp_config.json` 配置 MCP 服务器：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "docs/"],
      "disabled": 0
    },
    "tavily-websearch": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://mcp.tavily.com/mcp/?tavilyApiKey=YOUR_KEY"],
      "disabled": 0
    }
  }
}
```

设置 `disabled: 0` 启用，`disabled: 1` 禁用。

## 常见问题

### MCP 工具连接失败
```bash
# 检查 Node.js
node --version

# 验证 API 密钥
# 编辑 mcp_config.json 中的 API Key
```

### AWS Bedrock 认证失败
```bash
# 检查 AWS 凭证
aws configure list

# 验证环境变量
echo $AWS_ACCESS_KEY_ID
```

### 启用代码解释器或浏览器
在 `simple_agent.py` 中取消注释相应工具：
```python
self.basic_tools = [
    current_time,
    agentcore_code_interpreter.code_interpreter,  # 取消注释
    agentcore_browser.browser  # 取消注释
]
```

## 相关资源

- [Strands Agents SDK](https://github.com/strands-ai/strands-agents)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [MCP 服务器列表](https://github.com/modelcontextprotocol/servers)

## 许可证

MIT License
