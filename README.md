# Strands Agent 项目

基于Strands Agents SDK的简洁AI代理，支持自定义问题和系统提示词，集成MCP工具和多种基础功能。

## 🎯 项目特色

- ✅ **简洁易用**: 专注核心功能，界面清爽
- ✅ **自定义提示词**: 支持任意系统提示词定制
- ✅ **MCP集成**: 支持Model Context Protocol工具
- ✅ **详细模式**: 可查看完整执行过程
- ✅ **多种工具**: 数学计算、文件操作、Python执行、网络请求等

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install strands-agents strands-agents-tools mcp
```

### 2. 配置环境

复制环境配置文件：
```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的AWS凭证：
```bash
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
```

### 3. 运行Agent

```bash
python simple_agent.py
```

## 📁 项目文件

```
strands-agent/
├── simple_agent.py          # 主Agent程序
├── mcp_config.json         # MCP服务器配置
├── .env.example           # 环境配置模板
├── .env                   # 环境配置文件
├── LICENSE               # 许可证
└── README.md            # 项目说明
```

## 📋 可用工具

| 工具 | 功能 | 示例用法 |
|------|------|----------|
| 🧮 calculator | 数学计算 | "计算 123 * 456 + 789" |
| ⏰ current_time | 时间查询 | "现在几点了？今天星期几？" |
| 📖 file_read | 读取文件 | "读取README.md的内容" |
| 📝 file_write | 写入文件 | "创建一个文件保存今天的日期" |
| 🖼️ image_reader | 图片分析 | "分析这张图片的内容" |
| 📚 MCP工具 | 扩展功能 | "搜索关于AI的最新新闻" |

**注意**: python_repl, http_request, shell 工具已被注释，如需使用请在代码中取消注释。

## 💡 选择模式
![alt text](image.png)
```bash
选择模式:
1. 交互模式 
2. 单次问答 (推荐)
3. 帮助信息

请选择 (1-3): 
# 测试时推荐直接填 ‘2’ 即可
```

## 🔍 详细模式

### 简洁模式 vs 详细模式

**简洁模式** (默认):
- 只显示最终回答
- 适合日常使用
- 界面清爽

**详细模式**:
- 显示工具调用过程
- 显示Agent思考步骤
- 适合调试和学习

### 启用详细模式

**启动时选择:**
```bash
python simple_agent.py
# 选择 'y' 启用详细模式
```

**运行中切换:**
```
💬 你的问题: verbose
# 输入 'verbose' 切换显示模式
```

### 详细模式示例
```
🤖 思考中...
🔧 Tool Call: current_time()
📤 Tool Result: 2025-01-20 14:30:25 (Sunday)
🔧 Tool Call: calculator(expression="365-20")
📤 Tool Result: 345
💭 Agent思考: 现在我知道了当前时间和计算结果...
✅ 最终回答: 现在是2025年1月20日14:30，星期日。距离年底还有345天。
```

## 🎛️ 交互命令

在交互模式中可以使用以下命令：

- `quit` - 退出程序
- `prompt` - 修改系统提示词
- `verbose` - 切换详细/简洁模式
- `help` - 显示帮助信息

## 🔌 MCP配置

编辑 `mcp_config.json` 来配置MCP服务器：

```json
{
  "mcpServers": {
    "web-search": {
      "command": "npx",
      "args": ["-y", "@smithery/cli@latest", "run", "exa","--key"],
      "disabled": false
    },
    "sqlite": {
      "command": "uvx",
      "args": ["mcp-server-sqlite", "--db-path", "./data.db"],
      "disabled": true
    }
  }
}
```

## 🎯 使用技巧

### 问题设计
- **具体明确**: "用Python分析CSV文件" 比 "分析数据" 更好
- **包含上下文**: 提供必要的背景信息
- **指定格式**: "生成表格"、"画图表"、"写代码"

### 提示词优化
- **明确角色**: 定义专业身份和技能
- **行为指导**: 说明期望的工作方式
- **输出要求**: 指定回答格式和详细程度

### 工具使用
- **组合使用**: 可以要求Agent使用多个工具完成复杂任务
- **文件操作**: 先创建文件，再读取分析
- **图片分析**: 支持分析图片内容、识别文字、描述场景等
- **MCP工具**: 获取实时信息和专业文档

### 图片分析示例
```
问题: "分析这张图片，告诉我图片中有什么内容"
系统提示词: "你是图像分析专家，请详细描述图片内容，包括物体、场景、文字等信息。"

# Agent会自动使用image_reader工具分析图片
```

## 🔍 故障排除

### 常见问题

**Q: MCP工具连接失败？**
```bash
# 检查网络连接
# 验证API密钥
# 确认MCP服务器配置正确
```

**Q: Python代码执行错误？**
```bash
# 检查环境变量设置
export PYTHON_REPL_INTERACTIVE=False
```

**Q: 文件操作权限问题？**
```bash
# 检查目录权限
chmod 755 .
```

### 调试模式

启用详细日志：
```python
# 在simple_agent.py开头修改
logging.basicConfig(level=logging.DEBUG)
```

## 📊 性能优化

### 提升速度
- 精简系统提示词
- 明确具体问题
- 合理选择工具

### 控制成本
- 监控Token使用量
- 避免重复查询
- 优化提示词长度

## 🎉 开始使用

1. **安装依赖**: `pip install strands-agents strands-agents-tools mcp`
2. **配置环境**: 编辑 `.env` 文件
3. **启动程序**: `python simple_agent.py`
4. **选择模式**: 交互模式或单次问答
5. **开始对话**: 输入问题和自定义提示词

## 📄 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**享受与AI代理的智能对话吧！** 🚀