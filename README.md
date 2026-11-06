# 作业八　自制语言模型的知识蒸馏（纯 CPU）— 代码与运行结果

## 文件说明

| 文件 | 内容 |
|---|---|
| `distill_lm.py` | 主脚本：语料生成、字符级 GPT、教师训练、从零/蒸馏学生、T×α 消融、特征蒸馏、动态量化 |
| `hw8_extras.py` | 拓展 C（自蒸馏）与 E（教师规模扫描） |
| `make_figs.py` | 由 json 结果生成报告用图 |
| `runs_distill/results_*.json` | 各次运行的完整原始数据（loss 曲线、消融网格、量化指标） |
| `runs_distill/fig*.png` | 报告图 1（训练曲线）与图 2（T×α 热力图） |
| `runs_distill/*.pt` | 训练好的教师权重、蒸馏学生、fp32/int8 导出版本 |
| `logs/` | 各步骤完整控制台输出 |

## 复现命令

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib

python distill_lm.py --mode quick          # 环境自检（约 7 s）
python distill_lm.py --mode teacher --retrain   # 训练教师（约 9 s）
python distill_lm.py --mode all            # 从零 vs 蒸馏（约 7 s）
python distill_lm.py --mode ablate         # T×α 消融 16 组（约 31 s）
python distill_lm.py --mode all --feat 0.1 # 拓展 A 特征蒸馏
python distill_lm.py --mode quantize       # 拓展 D 动态量化导出
python hw8_extras.py                       # 拓展 C 自蒸馏 + E 教师规模扫描
```

## 实测环境

Windows 11 · Python 3.10.21 · PyTorch 2.14.0+cpu · AMD Ryzen 7 8845H（8 核 16 线程）· 纯 CPU

## 主要结果

- 教师 0.737 M：留出集 loss 0.8322，PPL 2.30
- 学生 0.147 M 从零：1.2722 / PPL 3.57 / 一致率 0.613
- 学生 0.147 M 蒸馏（T=4, α=0.5）：**1.2109 / PPL 3.36 / 一致率 0.656**，相对下降 4.8%，压缩 5.0×
- T×α 16 组最优：T=8、α=0.3（1.4008）；最差：T=8、α=0.9（1.6724）
