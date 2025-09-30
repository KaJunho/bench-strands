#!/usr/bin/env python3
"""
Strands Agent系统启动器
提供统一的入口来选择不同的运行模式
"""

import sys
import os


def show_banner():
    """显示系统横幅"""
    print("🚀 Strands Agent System")
    print("=" * 50)
    print("基于Strands SDK和GraphBuilder的智能Agent系统")
    print("支持单Agent和GraphBuilder多Agent协作模式")
    print("=" * 50)


def show_menu():
    """显示主菜单"""
    print("\n📋 选择运行模式:")
    print("1. 单Agent模式 - 快速问答和简单任务")
    print("2. 多Agent系统 - 深度搜索和复杂任务协作")
    print("3. 系统测试 - 验证系统组件状态")
    print("4. 查看帮助 - 使用指南和说明")
    print("0. 退出")


def run_single_agent():
    """运行单Agent模式"""
    print("\n🤖 启动单Agent模式...")
    try:
        import simple_agent
        simple_agent.main()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ 运行失败: {e}")


def run_multi_agent():
    """运行多Agent模式"""
    print("\n🤖 启动GraphBuilder多Agent任务解决系统...")
    try:
        import multi_agent
        multi_agent.main()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("请确保已安装Strands依赖和GraphBuilder")
    except Exception as e:
        print(f"❌ 运行失败: {e}")





def run_test():
    """运行测试"""
    print("\n🧪 启动系统测试...")
    try:
        import test_multi_agent
        test_multi_agent.main()
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")


def show_help():
    """显示帮助信息"""
    print("\n📖 使用帮助")
    print("-" * 30)
    
    print("\n🤖 单Agent模式:")
    print("- 适合快速问答、代码生成、简单分析")
    print("- 响应速度快，资源消耗低")
    print("- 支持MCP工具集成")
    
    print("\n🤖🤖 多Agent模式:")
    print("- 基于Strands GraphBuilder的图形化工作流多Agent架构")
    print("- 适合复杂计算、数据分析、多步骤任务")
    print("- 5个专门Agent通过GraphBuilder协作工作")
    print("- 包含任务分析、信息收集、工具执行、结果分析、答案格式化等阶段")
    print("- 支持迭代执行和条件分支，直到任务完成")
    print("- 使用节点和边构建的图形化执行流程")
    
    print("\n🔧 环境配置:")
    print("1. 复制 .env.example 为 .env")
    print("2. 配置AWS凭证或OpenAI API密钥")
    print("3. 安装依赖: pip install -r requirements.txt")
    
    print("\n📁 配置文件:")
    print("- .env: 环境变量配置")
    print("- mcp_config.json: MCP工具配置")
    print("- multi_agent_config.json: 多Agent系统配置")
    
    print("\n🎯 使用建议:")
    print("- 简单查询 → 单Agent模式")
    print("- 复杂研究 → 多Agent模式")
    print("- 首次使用 → 先运行系统测试")


def check_environment():
    """检查环境配置"""
    issues = []
    
    # 检查必要文件
    required_files = [
        "simple_agent.py",
        "multi_agent.py", 
        "requirements.txt"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            issues.append(f"缺少文件: {file}")
    
    # 检查环境配置
    if not os.path.exists(".env") and not os.path.exists(".env.example"):
        issues.append("缺少环境配置文件 (.env 或 .env.example)")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        issues.append(f"Python版本过低: {sys.version_info}, 需要3.8+")
    
    if issues:
        print("\n⚠️ 环境检查发现问题:")
        for issue in issues:
            print(f"   - {issue}")
        print("\n请解决这些问题后重新运行")
        return False
    
    return True


def main():
    """主函数"""
    show_banner()
    
    # 检查环境
    if not check_environment():
        return
    
    try:
        while True:
            show_menu()
            
            choice = input("\n请选择 (0-5): ").strip()
            
            if choice == "0":
                print("\n👋 再见！")
                break
            elif choice == "1":
                run_single_agent()
            elif choice == "2":
                run_multi_agent()
            elif choice == "3":
                run_test()
            elif choice == "4":
                show_help()
            else:
                print("❌ 无效选择，请输入0-4")
            
            # 询问是否继续
            if choice in ["1", "2", "3"]:
                continue_choice = input("\n按回车键返回主菜单，或输入'q'退出: ").strip()
                if continue_choice.lower() == 'q':
                    break
    
    except KeyboardInterrupt:
        print("\n\n👋 再见！")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")


if __name__ == "__main__":
    main()