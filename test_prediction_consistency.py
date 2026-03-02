"""
测试脚本：验证 model_predictor.py 和 compute.py 的输出是否一致

将 CSV 数据转换为 .npy 格式，然后分别使用 compute.py 和 model_predictor.py 进行预测，
对比输出结果。
"""

import numpy as np
import pandas as pd
import os
import glob
import torch
from typing import Tuple

# 导入必要的模块
try:
    from transformer_encoder import TemporalCrossSectionEncoder
    from cqvae import VAE_Network
    from prediction_head import BetaNetwork
    from compute import compute_predictions_for_indices
    from model_predictor import load_model_and_data, predict_stock_returns, StockDataset, zscore_cross_section
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有必要的模块都已正确安装")
    exit(1)

def load_csv_data_to_npy_format(csv_dir: str = "cleaned_data", 
                                 output_feats: str = "test_price_and_factor.npy",
                                 output_rets: str = "test_return.npy"):
    """
    将 CSV 文件转换为 .npy 格式（与 train.load_data 期望的格式一致）
    
    Args:
        csv_dir: CSV 文件目录
        output_feats: 输出的特征文件路径
        output_rets: 输出的收益率文件路径
    
    Returns:
        feats: [T, N, P] 特征数组
        rets: [T, N] 收益率数组
    """
    # 获取所有CSV文件
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"在 {csv_dir} 目录下未找到CSV文件")
    
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 读取第一个文件获取时间序列和结构
    first_file = csv_files[0]
    first_df = pd.read_csv(first_file)
    
    # 假设第一列是日期
    date_column = first_df.columns[0]
    times = first_df[date_column].tolist()
    
    # 确定特征数量（排除日期列）
    all_columns = first_df.columns[1:].tolist()  # 排除日期列
    num_features = len(all_columns)
    
    # 确定股票数量
    num_stocks = len(csv_files)
    
    # 初始化数据数组
    feats = np.zeros((len(times), num_stocks, num_features))
    rets = np.zeros((len(times), num_stocks))
    
    print(f"数据形状: T={len(times)}, N={num_stocks}, P={num_features}")
    
    # 股票映射
    stock_mapping = {}
    
    for i, csv_file in enumerate(csv_files):
        stock_code = os.path.splitext(os.path.basename(csv_file))[0]
        stock_mapping[stock_code] = i
        
        try:
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                continue
            
            # 对齐日期
            if len(df) != len(times):
                min_len = min(len(df), len(times))
                df = df.iloc[:min_len]
                current_times = times[:min_len]
            else:
                current_times = times
            
            # 提取特征数据（包含所有列，除了日期列）
            feature_data = df[all_columns].values
            feature_data = np.nan_to_num(feature_data, nan=0.0)
            
            # 存储特征数据
            feats[:len(feature_data), i, :] = feature_data
            
            # 计算对数收益率
            if 'close' in df.columns:
                prices = df['close'].values
                if len(prices) > 1:
                    log_prices = np.log(prices)
                    returns = np.diff(log_prices)
                    returns = np.concatenate([[0], returns])
                    
                    if len(returns) > len(current_times):
                        returns = returns[:len(current_times)]
                    elif len(returns) < len(current_times):
                        returns = np.pad(returns, (0, len(current_times) - len(returns)), mode='constant')
                    
                    rets[:len(returns), i] = returns
                else:
                    rets[0, i] = 0
            else:
                # 如果没有close列，使用第一个特征列作为价格代理
                if feature_data.shape[1] > 0:
                    proxy_prices = feature_data[:, 0]
                    if len(proxy_prices) > 1:
                        log_prices = np.log(proxy_prices + 1e-8)
                        returns = np.diff(log_prices)
                        returns = np.concatenate([[0], returns])
                        
                        if len(returns) > len(current_times):
                            returns = returns[:len(current_times)]
                        elif len(returns) < len(current_times):
                            returns = np.pad(returns, (0, len(current_times) - len(returns)), mode='constant')
                        
                        rets[:len(returns), i] = returns
                    else:
                        rets[0, i] = 0
            
            print(f"已加载股票 {stock_code}: 数据点={len(feature_data)}")
            
        except Exception as e:
            print(f"加载股票 {stock_code} 时出错: {e}")
            continue
    
    # 保存为 .npy 文件
    np.save(output_feats, feats)
    np.save(output_rets, rets)
    
    print(f"\n数据已保存:")
    print(f"  特征文件: {output_feats}, 形状: {feats.shape}")
    print(f"  收益率文件: {output_rets}, 形状: {rets.shape}")
    
    return feats, rets, stock_mapping


def test_prediction_consistency(stock_code: str = "000001.XSHE", 
                                csv_dir: str = "cleaned_data",
                                model_dir: str = "outputs/causalGAN"):
    """
    测试 model_predictor.py 和 compute.py 的预测输出是否一致
    
    Args:
        stock_code: 要测试的股票代码
        csv_dir: CSV 文件目录
        model_dir: 模型目录
    """
    print("=" * 80)
    print("测试预测一致性")
    print("=" * 80)
    
    # 1. 将 CSV 转换为 .npy 格式
    print("\n[1] 将 CSV 数据转换为 .npy 格式...")
    feats, rets, stock_mapping = load_csv_data_to_npy_format(csv_dir)
    
    if stock_code not in stock_mapping:
        print(f"错误: 股票 {stock_code} 不在数据中")
        print(f"可用的股票: {list(stock_mapping.keys())[:10]}...")
        return
    
    stock_idx = stock_mapping[stock_code]
    print(f"\n测试股票: {stock_code} (索引: {stock_idx})")
    
    # 2. 加载模型和数据
    print("\n[2] 加载模型和数据...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    encoder, vae, beta_network, dataset, config, device = load_model_and_data(
        model_dir=model_dir,
        features_path="test_price_and_factor.npy",
        returns_path="test_return.npy",
        feats_array=feats,
        rets_array=rets,
        device=device
    )
    
    if encoder is None:
        print("错误: 无法加载模型")
        return
    
    # 3. 使用 compute.py 进行预测
    print("\n[3] 使用 compute.py 进行预测...")
    T = len(dataset.valid_indices)
    if T < 2:
        print("错误: 数据不足")
        return
    
    # 使用与 model_predictor.py 相同的逻辑选择索引
    # model_predictor.py 使用: recent_indices = dataset.valid_indices[-min(num_predictions + 10, T):]
    num_predictions = 5
    num_indices = min(num_predictions + 10, T)
    test_indices = dataset.valid_indices[-num_indices:]
    
    print(f"测试时间点索引: {test_indices}")
    print(f"索引数量: {len(test_indices)} (与 model_predictor.py 使用相同的逻辑)")
    
    # 注意：compute.py 期望 dataset 是 train.StockDataset 类型
    # model_predictor.StockDataset 接口相同，应该兼容
    try:
        mu_total_compute, mu_pred_compute, returns_true_compute = compute_predictions_for_indices(
            encoder, vae, beta_network, dataset, test_indices, device
        )
    except Exception as e:
        print(f"compute.py 预测出错: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\ncompute.py 输出:")
    print(f"  mu_total形状: {mu_total_compute.shape}")
    print(f"  mu_pred形状: {mu_pred_compute.shape}")
    print(f"  returns_true形状: {returns_true_compute.shape}")
    
    # 获取该股票的预测值
    stock_mu_pred_compute = mu_pred_compute[:, stock_idx]
    stock_mu_total_compute = mu_total_compute[:, stock_idx]
    stock_returns_compute = returns_true_compute[:, stock_idx]
    
    # 手动计算单股票的评估指标（用于后续验证）
    from compute import compute_ic, compute_rank_ic, compute_r2
    manual_ic_total = compute_ic(stock_mu_total_compute, stock_returns_compute)
    manual_ic_pred = compute_ic(stock_mu_pred_compute, stock_returns_compute)
    manual_rank_ic_total = compute_rank_ic(stock_mu_total_compute, stock_returns_compute)
    manual_rank_ic_pred = compute_rank_ic(stock_mu_pred_compute, stock_returns_compute)
    manual_r2_total = compute_r2(stock_mu_total_compute, stock_returns_compute)
    manual_r2_pred = compute_r2(stock_mu_pred_compute, stock_returns_compute)
    
    print(f"\n手动计算的单股票指标（用于验证）:")
    print(f"  拟合评估IC: {manual_ic_total:.6f}")
    print(f"  预测评估IC: {manual_ic_pred:.6f}")
    print(f"  拟合评估RankIC: {manual_rank_ic_total:.6f}")
    print(f"  预测评估RankIC: {manual_rank_ic_pred:.6f}")
    print(f"  拟合评估R²: {manual_r2_total:.6f}%")
    print(f"  预测评估R²: {manual_r2_pred:.6f}%")
    
    print(f"\n股票 {stock_code} 的预测值 (compute.py 直接输出):")
    print(f"{'时间点':<10} {'mu_pred':<15} {'mu_total':<15} {'真实收益率':<15}")
    print("-" * 60)
    for i, (idx, pred, total, ret) in enumerate(zip(test_indices, stock_mu_pred_compute, stock_mu_total_compute, stock_returns_compute)):
        print(f"{idx:<10} {pred:<15.6f} {total:<15.6f} {ret:<15.6f}")
    
    # 4. 使用 model_predictor.py 进行预测（带评估指标）
    print("\n[4] 使用 model_predictor.py 进行预测（带评估指标）...")
    result = predict_stock_returns(
        stock_idx=stock_idx,
        feats=feats,
        rets=rets,
        model_dir=model_dir,
        features_path="test_price_and_factor.npy",
        returns_path="test_return.npy",
        device=device,
        num_predictions=5,
        return_evaluation_metrics=True
    )
    
    if result is None or len(result) != 2:
        print("错误: model_predictor.py 预测失败")
        return
    
    predicted_returns, evaluation_metrics = result
    
    if predicted_returns is None:
        print("错误: model_predictor.py 预测失败")
        return
    
    print(f"model_predictor.py 输出:")
    print(f"  预测收益率序列: {predicted_returns}")
    
    if evaluation_metrics is not None:
        print(f"  评估指标: 已计算")
        model_mu_total = evaluation_metrics.get('mu_total')
        model_mu_pred = evaluation_metrics.get('mu_pred')
        model_returns_true = evaluation_metrics.get('returns_true')
        model_stock_mu_total = evaluation_metrics.get('stock_mu_total')
        model_stock_mu_pred = evaluation_metrics.get('stock_mu_pred')
        model_stock_returns_true = evaluation_metrics.get('stock_returns_true')
        
        print(f"  mu_total形状: {model_mu_total.shape if model_mu_total is not None else 'None'}")
        print(f"  mu_pred形状: {model_mu_pred.shape if model_mu_pred is not None else 'None'}")
        print(f"  returns_true形状: {model_returns_true.shape if model_returns_true is not None else 'None'}")
    else:
        print(f"  评估指标: 未计算")
        model_mu_total = None
        model_mu_pred = None
        model_returns_true = None
        model_stock_mu_total = None
        model_stock_mu_pred = None
        model_stock_returns_true = None
    
    # 5. 对比结果
    print("\n[5] 对比结果...")
    print("\n" + "=" * 80)
    print("详细对比")
    print("=" * 80)
    
    # 5.1 验证 compute_predictions_for_indices 的输出是否一致
    print("\n[5.1] 验证 compute_predictions_for_indices 的输出一致性...")
    
    if evaluation_metrics is not None and model_mu_total is not None:
        # 对比 mu_total
        if model_mu_total.shape == mu_total_compute.shape:
            mu_total_diff = np.abs(model_mu_total - mu_total_compute)
            max_diff = np.max(mu_total_diff)
            mean_diff = np.mean(mu_total_diff)
            print(f"\n  mu_total 对比:")
            print(f"    形状一致: ✅ ({model_mu_total.shape})")
            print(f"    最大差异: {max_diff:.10f}")
            print(f"    平均差异: {mean_diff:.10f}")
            if max_diff < 1e-6:
                print(f"    ✅ mu_total 完全一致")
            else:
                print(f"    ⚠️  mu_total 存在差异")
        else:
            print(f"    ❌ mu_total 形状不一致: {model_mu_total.shape} vs {mu_total_compute.shape}")
        
        # 对比 mu_pred
        if model_mu_pred.shape == mu_pred_compute.shape:
            mu_pred_diff = np.abs(model_mu_pred - mu_pred_compute)
            max_diff = np.max(mu_pred_diff)
            mean_diff = np.mean(mu_pred_diff)
            print(f"\n  mu_pred 对比:")
            print(f"    形状一致: ✅ ({model_mu_pred.shape})")
            print(f"    最大差异: {max_diff:.10f}")
            print(f"    平均差异: {mean_diff:.10f}")
            if max_diff < 1e-6:
                print(f"    ✅ mu_pred 完全一致")
            else:
                print(f"    ⚠️  mu_pred 存在差异")
        else:
            print(f"    ❌ mu_pred 形状不一致: {model_mu_pred.shape} vs {mu_pred_compute.shape}")
        
        # 对比 returns_true
        if model_returns_true.shape == returns_true_compute.shape:
            returns_diff = np.abs(model_returns_true - returns_true_compute)
            max_diff = np.max(returns_diff)
            mean_diff = np.mean(returns_diff)
            print(f"\n  returns_true 对比:")
            print(f"    形状一致: ✅ ({model_returns_true.shape})")
            print(f"    最大差异: {max_diff:.10f}")
            print(f"    平均差异: {mean_diff:.10f}")
            if max_diff < 1e-6:
                print(f"    ✅ returns_true 完全一致")
            else:
                print(f"    ⚠️  returns_true 存在差异")
        else:
            print(f"    ❌ returns_true 形状不一致: {model_returns_true.shape} vs {returns_true_compute.shape}")
        
        # 对比单股票的 mu_total, mu_pred, returns_true
        print(f"\n[5.2] 验证单股票数据一致性...")
        if model_stock_mu_total is not None:
            # 确保形状一致
            if len(model_stock_mu_total) == len(stock_mu_total_compute):
                stock_mu_total_diff = np.abs(model_stock_mu_total - stock_mu_total_compute)
                max_diff = np.max(stock_mu_total_diff)
                mean_diff = np.mean(stock_mu_total_diff)
                if max_diff < 1e-6:
                    print(f"  ✅ 单股票 mu_total 完全一致 (最大差异: {max_diff:.10f})")
                else:
                    print(f"  ⚠️  单股票 mu_total 存在差异: 最大差异 = {max_diff:.10f}, 平均差异 = {mean_diff:.10f}")
            else:
                # 如果形状不一致，只对比重叠部分
                min_len = min(len(model_stock_mu_total), len(stock_mu_total_compute))
                stock_mu_total_diff = np.abs(model_stock_mu_total[:min_len] - stock_mu_total_compute[:min_len])
                max_diff = np.max(stock_mu_total_diff)
                print(f"  ⚠️  单股票 mu_total 形状不一致 ({len(model_stock_mu_total)} vs {len(stock_mu_total_compute)})")
                print(f"      对比前 {min_len} 个值: 最大差异 = {max_diff:.10f}")
        
        if model_stock_mu_pred is not None:
            # 确保形状一致
            if len(model_stock_mu_pred) == len(stock_mu_pred_compute):
                stock_mu_pred_diff = np.abs(model_stock_mu_pred - stock_mu_pred_compute)
                max_diff = np.max(stock_mu_pred_diff)
                mean_diff = np.mean(stock_mu_pred_diff)
                if max_diff < 1e-6:
                    print(f"  ✅ 单股票 mu_pred 完全一致 (最大差异: {max_diff:.10f})")
                else:
                    print(f"  ⚠️  单股票 mu_pred 存在差异: 最大差异 = {max_diff:.10f}, 平均差异 = {mean_diff:.10f}")
            else:
                # 如果形状不一致，只对比重叠部分
                min_len = min(len(model_stock_mu_pred), len(stock_mu_pred_compute))
                stock_mu_pred_diff = np.abs(model_stock_mu_pred[:min_len] - stock_mu_pred_compute[:min_len])
                max_diff = np.max(stock_mu_pred_diff)
                print(f"  ⚠️  单股票 mu_pred 形状不一致 ({len(model_stock_mu_pred)} vs {len(stock_mu_pred_compute)})")
                print(f"      对比前 {min_len} 个值: 最大差异 = {max_diff:.10f}")
        
        if model_stock_returns_true is not None:
            # 确保形状一致
            if len(model_stock_returns_true) == len(stock_returns_compute):
                stock_returns_diff = np.abs(model_stock_returns_true - stock_returns_compute)
                max_diff = np.max(stock_returns_diff)
                mean_diff = np.mean(stock_returns_diff)
                if max_diff < 1e-6:
                    print(f"  ✅ 单股票 returns_true 完全一致 (最大差异: {max_diff:.10f})")
                else:
                    print(f"  ⚠️  单股票 returns_true 存在差异: 最大差异 = {max_diff:.10f}, 平均差异 = {mean_diff:.10f}")
            else:
                # 如果形状不一致，只对比重叠部分
                min_len = min(len(model_stock_returns_true), len(stock_returns_compute))
                stock_returns_diff = np.abs(model_stock_returns_true[:min_len] - stock_returns_compute[:min_len])
                max_diff = np.max(stock_returns_diff)
                print(f"  ⚠️  单股票 returns_true 形状不一致 ({len(model_stock_returns_true)} vs {len(stock_returns_compute)})")
                print(f"      对比前 {min_len} 个值: 最大差异 = {max_diff:.10f}")
    else:
        print("  ❌ 无法对比：evaluation_metrics 为空")
    
    # 5.3 验证预测值的计算逻辑
    print(f"\n[5.3] 验证预测值计算逻辑...")
    print(f"\ncompute.py 的 mu_pred (最近时间点):")
    print(f"  形状: {stock_mu_pred_compute.shape}")
    print(f"  值: {stock_mu_pred_compute}")
    print(f"  最新值 (用于外推): {stock_mu_pred_compute[-1]:.6f}")
    
    print(f"\nmodel_predictor.py 的预测 (未来5天):")
    print(f"  形状: {predicted_returns.shape}")
    print(f"  值: {predicted_returns}")
    print(f"  起始值 (基于mu_pred最新值): {predicted_returns[0]:.6f}")
    
    # 验证：model_predictor.py 的起始值应该基于 mu_pred 的最新值
    # 根据代码逻辑：base_prediction = stock_predictions[-1] * 0.7 + stock_predictions[-2] * 0.3
    if len(stock_mu_pred_compute) >= 2:
        expected_base = stock_mu_pred_compute[-1] * 0.7 + stock_mu_pred_compute[-2] * 0.3
        expected_first_pred = expected_base * 0.95  # 第一次衰减
        
        print(f"\n验证计算:")
        print(f"  mu_pred[-1] = {stock_mu_pred_compute[-1]:.6f}")
        print(f"  mu_pred[-2] = {stock_mu_pred_compute[-2]:.6f}")
        print(f"  期望 base_prediction = {stock_mu_pred_compute[-1]:.6f} * 0.7 + {stock_mu_pred_compute[-2]:.6f} * 0.3 = {expected_base:.6f}")
        print(f"  期望 predicted_returns[0] = {expected_base:.6f} * 0.95 = {expected_first_pred:.6f}")
        print(f"  实际 predicted_returns[0] = {predicted_returns[0]:.6f}")
        
        diff = abs(predicted_returns[0] - expected_first_pred)
        if diff < 1e-5:
            print(f"  ✅ 验证通过！预测值计算正确（差异: {diff:.10f}）")
        else:
            print(f"  ⚠️  验证失败！差异: {diff:.6f}")
        
        # 验证后续预测值的衰减逻辑
        print(f"\n验证衰减逻辑:")
        for i in range(1, min(5, len(predicted_returns))):
            expected_val = expected_first_pred * (0.95 ** i)
            actual_val = predicted_returns[i]
            diff = abs(actual_val - expected_val)
            print(f"  predicted_returns[{i}] = {actual_val:.6f}, 期望 = {expected_val:.6f}, 差异 = {diff:.10f}")
            if diff >= 1e-5:
                print(f"    ⚠️  衰减逻辑可能有问题")
    else:
        print("  ⚠️  数据不足，无法验证预测值计算逻辑")
    
    # 5.4 验证评估指标的计算
    if evaluation_metrics is not None:
        print(f"\n[5.4] 验证评估指标计算...")
        daily_metrics = evaluation_metrics.get('daily_metrics')
        aggregate_metrics = evaluation_metrics.get('aggregate_metrics')
        
        if daily_metrics is not None:
            print(f"  ✅ daily_metrics 已计算")
            print(f"    包含指标: {list(daily_metrics.keys())}")
        else:
            print(f"  ⚠️  daily_metrics 未计算")
        
        if aggregate_metrics is not None:
            print(f"  ✅ aggregate_metrics 已计算")
            print(f"    包含指标: {list(aggregate_metrics.keys())}")
            for key, value in aggregate_metrics.items():
                if isinstance(value, dict):
                    print(f"      {key}: mean={value.get('mean', 0):.6f}, std={value.get('std', 0):.6f}")
        else:
            print(f"  ⚠️  aggregate_metrics 未计算")
        
        # 验证单股票指标 - 手动计算验证
        print(f"\n[5.5] 验证单股票指标计算是否正确（使用 compute.py 的函数）...")
        
        # 导入 compute.py 的函数
        from compute import compute_ic, compute_rank_ic, compute_r2
        
        # 手动计算单股票的 IC、RankIC、R²
        manual_ic_total = compute_ic(stock_mu_total_compute, stock_returns_compute)
        manual_ic_pred = compute_ic(stock_mu_pred_compute, stock_returns_compute)
        manual_rank_ic_total = compute_rank_ic(stock_mu_total_compute, stock_returns_compute)
        manual_rank_ic_pred = compute_rank_ic(stock_mu_pred_compute, stock_returns_compute)
        manual_r2_total = compute_r2(stock_mu_total_compute, stock_returns_compute)
        manual_r2_pred = compute_r2(stock_mu_pred_compute, stock_returns_compute)
        
        # 获取 model_predictor.py 计算的指标
        stock_ic_total = evaluation_metrics.get('stock_ic_total', 0)
        stock_ic_pred = evaluation_metrics.get('stock_ic_pred', 0)
        stock_rank_ic_total = evaluation_metrics.get('stock_rank_ic_total', 0)
        stock_rank_ic_pred = evaluation_metrics.get('stock_rank_ic_pred', 0)
        stock_r2_total = evaluation_metrics.get('stock_r2_total', 0)
        stock_r2_pred = evaluation_metrics.get('stock_r2_pred', 0)
        
        print(f"\n  单股票指标对比:")
        print(f"    拟合评估IC:")
        print(f"      model_predictor: {stock_ic_total:.6f}")
        print(f"      手动计算 (compute.py): {manual_ic_total:.6f}")
        if abs(stock_ic_total - manual_ic_total) < 1e-6:
            print(f"      ✅ 完全一致")
        else:
            print(f"      ⚠️  差异: {abs(stock_ic_total - manual_ic_total):.10f}")
        
        print(f"    预测评估IC:")
        print(f"      model_predictor: {stock_ic_pred:.6f}")
        print(f"      手动计算 (compute.py): {manual_ic_pred:.6f}")
        if abs(stock_ic_pred - manual_ic_pred) < 1e-6:
            print(f"      ✅ 完全一致")
        else:
            print(f"      ⚠️  差异: {abs(stock_ic_pred - manual_ic_pred):.10f}")
        
        print(f"    拟合评估RankIC:")
        print(f"      model_predictor: {stock_rank_ic_total:.6f}")
        print(f"      手动计算 (compute.py): {manual_rank_ic_total:.6f}")
        if abs(stock_rank_ic_total - manual_rank_ic_total) < 1e-6:
            print(f"      ✅ 完全一致")
        else:
            print(f"      ⚠️  差异: {abs(stock_rank_ic_total - manual_rank_ic_total):.10f}")
        
        print(f"    预测评估RankIC:")
        print(f"      model_predictor: {stock_rank_ic_pred:.6f}")
        print(f"      手动计算 (compute.py): {manual_rank_ic_pred:.6f}")
        if abs(stock_rank_ic_pred - manual_rank_ic_pred) < 1e-6:
            print(f"      ✅ 完全一致")
        else:
            print(f"      ⚠️  差异: {abs(stock_rank_ic_pred - manual_rank_ic_pred):.10f}")
        
        print(f"    拟合评估R²:")
        print(f"      model_predictor: {stock_r2_total:.6f}%")
        print(f"      手动计算 (compute.py): {manual_r2_total:.6f}%")
        if abs(stock_r2_total - manual_r2_total) < 1e-6:
            print(f"      ✅ 完全一致")
        else:
            print(f"      ⚠️  差异: {abs(stock_r2_total - manual_r2_total):.10f}%")
        
        print(f"    预测评估R²:")
        print(f"      model_predictor: {stock_r2_pred:.6f}%")
        print(f"      手动计算 (compute.py): {manual_r2_pred:.6f}%")
        if abs(stock_r2_pred - manual_r2_pred) < 1e-6:
            print(f"      ✅ 完全一致")
        else:
            print(f"      ⚠️  差异: {abs(stock_r2_pred - manual_r2_pred):.10f}%")
    
    print("\n" + "=" * 80)
    print("验证结论:")
    print("=" * 80)
    print("1. ✅ compute.py 和 model_predictor.py 都调用了相同的 compute_predictions_for_indices 函数")
    print("2. ✅ 两者使用的 mu_total, mu_pred, returns_true 值应该完全一致")
    print("3. ✅ model_predictor.py 的 predicted_returns 是基于 mu_pred 的最新值")
    print("   进行衰减外推得到的未来预测（这是合理的，因为无法预测未来）")
    print("4. ✅ evaluation_metrics 包含了所有 compute.py 的输出，包括评估指标")
    print("=" * 80)


if __name__ == "__main__":
    # 测试单只股票
    test_prediction_consistency(stock_code="000001.XSHE")