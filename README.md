# Gated Neural Basis Model

独立的 PyTorch GNBM 核心实现：共享基函数、完整基函数门控、训练、集成预测及贡献分解。

本目录仅包含核心代码和内存中的合成输入示例，不包含真实数据、数据集读取与预处理、特征构建、搜索、嵌套验证、实验管理、模型权重或论文结果。

## 运行

已验证环境为 Python 3.12.13、PyTorch 2.5.1、NumPy 2.4.6。请在具备这些依赖的环境中，从项目根目录直接运行，无需安装本项目：

```powershell
python -B -m examples.synthetic_demo
```

示例使用 CPU 和随机生成的输入数组，不读取或写出数据文件，不包含标准化等数据集处理实现。为便于演示，训练上限为 20 轮；可加 `--epochs 5` 快速运行。预处理后的特征矩阵及 log1p 目标由调用方提供。

## 核心接口

`GNBM(feature_names, parameters=None)` 构建模型。每个标量特征通过共享基函数网络；隐藏层使用 Linear、LayerNorm、GELU、Dropout，输出层不归一化。门控读取完整输入与展平后的基函数，对拼接输入执行 LayerNorm，再生成各特征、各基函数的 logits。

softmax 沿基函数维度计算，门控乘数为 `(1-alpha) + alpha*B*q`，每个特征的乘数均值为 1。门控输出层零初始化，温度和强度为固定超参数。网络输出为各特征贡献之和加全局偏置，网络内部不截断。

`fit_gnbm(...)` 接受特征矩阵及已转换的 log1p 目标，返回模型、最佳轮次、验证评分和训练历史。默认设置为 AdamW、学习率 `1e-3`、权重衰减 `1e-5`、批大小 `256`、最多 `240` 轮、早停耐心 `48` 轮和梯度范数上限 `0.5`。默认种子为 `1729, 2718, 31415`，优化使用 float32 MSE。

```python
from gnbm import fit_gnbm, predict_ensemble

model, best_epoch, validation_mae, history = fit_gnbm(
    feature_names, X_train, y_train_log1p,
    x_valid=X_valid, y_valid_log1p=y_valid_log1p,
    seed=1729, device="cpu",
)
predicted_log1p, predicted_response = predict_ensemble([model], X_test)
```

输入必须有限且特征顺序一致。早停使用原始输出 MAE，恢复最佳状态；历史字段 `validation_mae_log1p_raw` 记录此指标。返回的 `validation_mae` 使用保留模型的非负截断预测。省略验证集时按指定轮数训练，评分为 `None`。默认结构仅为可运行示例，并非稿件最终最优配置。

`predict_ensemble(...)` 先平均各模型的原始 log1p 输出，再截断为非负值并执行 `expm1`，返回两种尺度。

`model(x, return_aux=True)` 提供门控、特征贡献、同一拟合模型关闭门控后的 `ungated_feature_contributions`，以及两者之差 `net_gating_adjustments`。贡献求和加全局偏置可重构原始 log1p 输出；关闭门控的诊断不等于重新训练 NBM，也不表示因果效应。
