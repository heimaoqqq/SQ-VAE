#!/usr/bin/env python3
"""
微多普勒数据集的64×64 vs 256×256对比实验脚本
"""

import subprocess
import sys
import time
from datetime import datetime

def run_experiment(config_name, description):
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"🚀 开始实验: {description}")
    print(f"📋 配置文件: {config_name}")
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行训练
        cmd = [
            "python", "main.py", 
            "-c", config_name,
            "--save", "--dbg", "--gpu", "0,1"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ 实验完成: {description}")
            print(f"⏱️  训练时间: {duration/60:.1f} 分钟")
            
            # 提取关键指标
            output_lines = result.stdout.split('\n')
            best_loss = None
            best_mse = None
            best_perplexity = None
            
            for line in output_lines:
                if "Best models were loaded!!" in line:
                    # 找到测试结果
                    for i, next_line in enumerate(output_lines[output_lines.index(line):]):
                        if "Test" in next_line and "Loss:" in next_line:
                            parts = next_line.split()
                            for j, part in enumerate(parts):
                                if part == "Loss:":
                                    best_loss = parts[j+1].rstrip(',')
                                elif part == "MSE:":
                                    best_mse = parts[j+1].rstrip(',')
                                elif part == "Perplexity:":
                                    best_perplexity = parts[j+1].rstrip(',')
                            break
                    break
            
            print(f"📊 最终结果:")
            print(f"   Loss: {best_loss}")
            print(f"   MSE: {best_mse}")
            print(f"   Perplexity: {best_perplexity}")
            
            return {
                'config': config_name,
                'description': description,
                'duration': duration,
                'loss': best_loss,
                'mse': best_mse,
                'perplexity': best_perplexity,
                'success': True
            }
        else:
            print(f"❌ 实验失败: {description}")
            print(f"错误输出: {result.stderr}")
            return {
                'config': config_name,
                'description': description,
                'duration': duration,
                'success': False,
                'error': result.stderr
            }
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 实验超时: {description}")
        return {
            'config': config_name,
            'description': description,
            'success': False,
            'error': 'Timeout'
        }
    except Exception as e:
        print(f"💥 实验异常: {description}")
        print(f"异常信息: {str(e)}")
        return {
            'config': config_name,
            'description': description,
            'success': False,
            'error': str(e)
        }

def main():
    """运行对比实验"""
    print("🔬 微多普勒SQ-VAE对比实验")
    print("📋 实验计划:")
    print("   1. 64×64分辨率 (遵循原项目哲学)")
    print("   2. 256×256分辨率 (保持原始分辨率)")
    
    experiments = [
        {
            'config': 'microdoppler_gauss_1_64x64.yaml',
            'description': '64×64分辨率 - 原项目策略'
        },
        {
            'config': 'microdoppler_gauss_1_256x256.yaml', 
            'description': '256×256分辨率 - 高分辨率策略'
        }
    ]
    
    results = []
    
    for exp in experiments:
        result = run_experiment(exp['config'], exp['description'])
        results.append(result)
        
        # 实验间休息
        if exp != experiments[-1]:
            print(f"\n⏸️  实验间休息 30 秒...")
            time.sleep(30)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("📊 实验总结")
    print(f"{'='*80}")
    
    print(f"{'配置':<40} {'状态':<10} {'时间(分)':<10} {'Loss':<12} {'MSE':<12} {'Perplexity':<12}")
    print("-" * 80)
    
    for result in results:
        status = "✅ 成功" if result['success'] else "❌ 失败"
        duration = f"{result.get('duration', 0)/60:.1f}" if result['success'] else "N/A"
        loss = result.get('loss', 'N/A')
        mse = result.get('mse', 'N/A')
        perplexity = result.get('perplexity', 'N/A')
        
        print(f"{result['description']:<40} {status:<10} {duration:<10} {loss:<12} {mse:<12} {perplexity:<12}")
    
    # 性能对比
    successful_results = [r for r in results if r['success']]
    if len(successful_results) >= 2:
        print(f"\n🏆 性能对比:")
        r64 = next((r for r in successful_results if '64x64' in r['config']), None)
        r256 = next((r for r in successful_results if '256x256' in r['config']), None)
        
        if r64 and r256:
            try:
                loss_64 = float(r64['loss'])
                loss_256 = float(r256['loss'])
                mse_64 = float(r64['mse'])
                mse_256 = float(r256['mse'])
                
                print(f"   Loss: 64×64={loss_64:.4f} vs 256×256={loss_256:.4f}")
                print(f"   MSE:  64×64={mse_64:.1f} vs 256×256={mse_256:.1f}")
                
                if loss_256 < loss_64:
                    print(f"   🎯 256×256在Loss上表现更好 (改善 {(loss_64-loss_256)/loss_64*100:.1f}%)")
                else:
                    print(f"   🎯 64×64在Loss上表现更好 (改善 {(loss_256-loss_64)/loss_256*100:.1f}%)")
                    
                if mse_256 < mse_64:
                    print(f"   🎯 256×256在MSE上表现更好 (改善 {(mse_64-mse_256)/mse_64*100:.1f}%)")
                else:
                    print(f"   🎯 64×64在MSE上表现更好 (改善 {(mse_256-mse_64)/mse_256*100:.1f}%)")
                    
            except (ValueError, TypeError):
                print("   ⚠️  无法进行数值对比")
    
    print(f"\n🎉 所有实验完成!")

if __name__ == "__main__":
    main()
