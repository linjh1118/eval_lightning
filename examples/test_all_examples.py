#!/usr/bin/env python
"""
运行示例脚本

这个脚本提供了一个简单的命令行接口来运行eval-lightning框架的示例
"""

import argparse
import importlib
import os
import sys


def list_examples():
    """列出examples目录中的所有示例"""
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    examples = []
    
    for file in os.listdir(examples_dir):
        if file.endswith('.py') and not file.startswith('__'):
            examples.append(file[:-3])  # 去掉.py后缀
    
    return examples


def main():
    """主函数"""
    examples = list_examples()
    
    parser = argparse.ArgumentParser(description='运行eval-lightning框架示例')
    parser.add_argument('example', choices=examples + ['all'], 
                        help='要运行的示例名称，或"all"运行所有示例')
    
    args = parser.parse_args()
    
    # 将当前的父目录添加到sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    if args.example == 'all':
        for example in examples:
            print(f"\n{'='*50}")
            print(f"运行示例: {example}")
            print(f"{'='*50}\n")
            try:
                module = importlib.import_module(f"examples.{example}")
                if hasattr(module, 'main'):
                    module.main()
                else:
                    print(f"警告: {example}没有main函数")
            except Exception as e:
                print(f"运行示例{example}时出错: {e}")
    else:
        try:
            module = importlib.import_module(f"examples.{args.example}")
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"警告: {args.example}没有main函数")
        except Exception as e:
            print(f"运行示例{args.example}时出错: {e}")


if __name__ == "__main__":
    main() 