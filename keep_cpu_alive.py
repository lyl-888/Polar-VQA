#!/usr/bin/env python3
"""
CPU 保活程序
用于防止容器因 CPU 利用率低于 2% 而被认为空闲
目标：将 CPU 使用率稳定在 5% 左右

使用方法：
    python keep_cpu_alive.py                    # 前台运行
    nohup python keep_cpu_alive.py > cpu_keepalive.log 2>&1 &  # 后台运行
"""

import time
import math
import sys
import signal
import os

# 配置参数
TARGET_CPU_USAGE = 5.0  # 目标 CPU 使用率（%）
SLEEP_INTERVAL = 0.01   # 休眠间隔（秒），调整这个值可以控制 CPU 使用率
WORK_DURATION = 0.0005  # 工作时间（秒），在这段时间内进行 CPU 密集型计算

# 全局变量，用于优雅退出
running = True

def signal_handler(signum, frame):
    """处理退出信号"""
    global running
    print(f"\n✅ 收到退出信号 ({signum})，正在优雅退出...")
    running = False
    sys.exit(0)

def keep_cpu_busy():
    """持续占用 CPU，使用率稳定在目标值附近"""
    global running
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"✅ CPU 保活程序已启动")
    print(f"   目标 CPU 使用率: {TARGET_CPU_USAGE}%")
    print(f"   进程 PID: {os.getpid()}")
    print(f"   按 Ctrl+C 终止程序")
    print("-" * 50)
    
    count = 0
    iteration = 0
    
    try:
        while running:
            # 工作时间：进行 CPU 密集型计算
            work_start = time.time()
            while time.time() - work_start < WORK_DURATION:
                # 轻量级数学运算，持续占用 CPU
                count += math.sqrt(count ** 2 + 1)
                count %= 1000000  # 防止数值溢出
            
            # 休眠时间：让 CPU 休息
            time.sleep(SLEEP_INTERVAL)
            
            # 每 1000 次迭代输出一次状态（可选，减少日志输出）
            iteration += 1
            if iteration % 1000 == 0:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] CPU 保活运行中... (迭代 {iteration})")
                sys.stdout.flush()  # 确保输出被刷新到文件（如果使用 nohup）
    
    except KeyboardInterrupt:
        print("\n✅ CPU 保活程序已手动终止")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        raise

if __name__ == "__main__":
    try:
        keep_cpu_busy()
    except KeyboardInterrupt:
        print("\n✅ CPU 保活程序已手动终止")
    except Exception as e:
        print(f"\n❌ 程序异常退出: {e}")
        sys.exit(1)

