# RL-ZL：REMUS-100-like AUV Stage 0–1

本仓库严格依据 **《REMUS100_RWPVSD_SAC_实验方案_V4_硬错误修订版》** 实现 RWPVSD-SAC 实验的 **Stage 0：环境与动力学验证** 和 **Stage 1：低难度 SAC Teacher**。Stage 1 复用 Stage 0 的同一个环境，不复制或绕过动力学、感知、奖励与终止逻辑。

## 当前已实现

- 500 m × 500 m × 100 m 三维浅水环境，合法深度 5–95 m；
- REMUS-100-like 欠驱动规划层运动学；
- 纵向速度、俯仰角速度、艏摇角速度的一阶执行器响应；
- 球体、垂直圆柱体、椭球体障碍及安全膨胀；
- 背景流、局部涡旋和时变扰动组成的解析海流；
- 航向/俯仰对齐的局部 26-ray 距离感知；
- 连续归一化动作、53 维归一化观测、奖励分量与规则安全代价；
- V4 规定的早期 `q_d/r_d` 收缩范围与 `Δq_d/Δr_d` 变化率限制；
- V4 规定的稠密奖励 `[-5, 5]` 裁剪与规则安全代价裁剪/归一化；
- 成功、碰撞、边界、深度、动力学、振荡和超时分类；
- 基于粗栅格 A* 的场景可通行性检查；
- 固定随机种子复现、单元测试和三维轨迹验证图。

Stage 1 在上述环境之上新增：

- PyTorch tanh-Gaussian SAC actor，V4 教师结构 `53→256→256→128→(μ, logσ)`；
- 双 Q critic、target critic、自动熵温度和 Polyak 软更新；
- 区分 `terminated` 与时间截断 `truncated` 的经验回放，超时仍允许 Bellman bootstrap；
- Stage 1A 无流 `4–6` 障碍和 Stage 1B 弱流 `6–8` 障碍 curriculum；
- 验证筛查、curriculum 确认和最终测试使用三组互不重叠的固定种子；
- 20 episode 仅作趋势筛查，curriculum 晋级必须额外通过 100 episode 确认；
- 评估记录保留场景几何、终端姿态/角速度与动力学裁剪诊断；
- 独立 100 episode 最终测试与 V4 严格门槛：成功率 `>90%`、碰撞率 `≤10%`。

当前仓库完成了 Stage 1 代码和短程 smoke run；短程运行只证明训练链路可执行，不代表教师已经训练达标，也不能作为论文结果。

## 放到 Windows 的 `D:\RL+ZL`

打开 Anaconda Prompt 或 PowerShell：

```powershell
cd D:\
git clone https://github.com/Whsjbwy/RL-ZL.git "RL+ZL"
cd "D:\RL+ZL"
```

如果目录已经存在：

```powershell
cd "D:\RL+ZL"
git pull
```

## 环境安装

推荐继续使用你的 `auv-rl` Conda 环境：

```powershell
conda activate auv-rl
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

PyTorch 请优先保留你现有环境中的版本。若尚未安装，请根据本机 CUDA 和 RTX 5060 情况使用 PyTorch 官方安装选择器，不要在已有可用环境里重复覆盖安装。

## 一键验证

```powershell
python -m unittest discover -s tests -v
python scripts/check_device.py
python scripts/run_stage0_cases.py --config configs/stage0.yaml
python scripts/run_stage0_validation.py --config configs/stage0.yaml --episodes 20
```

## Stage 1：先诊断，再正式训练

如果使用 Codex 负责实验执行、调参和阶段验收，请先让它完整阅读
[`docs/STAGE1_CODEX_EXECUTION_GATE.md`](docs/STAGE1_CODEX_EXECUTION_GATE.md)。该文件规定了
Stage 1 的固定口径、数据隔离、调参顺序、交付物和进入 Stage 2 前的硬门槛。

先确认 PyTorch、CUDA 和 GPU：

```powershell
python scripts/check_device.py
```

运行 128 步端到端诊断。该命令会真实执行 replay 采样、SAC 反向传播、评估和 checkpoint 保存，但不会产生有效成功率：

```powershell
python scripts/run_stage1_smoke.py
```

诊断通过后，在 RTX 5060 上正式训练：

```powershell
python scripts/train_stage1_teacher.py --config configs/stage1_teacher.yaml --device cuda
```

训练中的 20 回合固定验证只负责发现候选 checkpoint；候选还必须在另一组固定种子上通过 100 回合确认（成功率严格大于 90%，碰撞率不高于 10%），才会进入下一难度。最终 checkpoint 再使用第三组独立种子测试 100 episodes：

```powershell
python scripts/evaluate_stage1_teacher.py `
  --config configs/stage1_teacher.yaml `
  --checkpoint artifacts/stage1_teacher/checkpoints/final.pt `
  --episodes 100 `
  --seed-split final_test
```

长训练中断后可从模型和优化器状态继续；经验回放不会写进 checkpoint，恢复后固定先收集 10,000 条新样本，再恢复梯度更新：

```powershell
python scripts/train_stage1_teacher.py `
  --config configs/stage1_teacher.yaml `
  --device cuda `
  --resume artifacts/stage1_teacher/checkpoints/latest.pt
```

验证结果输出到：

```text
artifacts/stage0_validation/
├── summary.json
├── trajectory.png
└── ray_geometry.png

artifacts/stage0_cases/
├── stage0_cases.json
├── single_obstacle_avoidance.png
└── single_obstacle_rays.png

artifacts/stage1_teacher/
├── summary.json
├── training_episodes.jsonl
├── training_updates.jsonl
├── evaluations/
└── checkpoints/
```

Stage 0 通过标准：

1. 全部单元测试通过；
2. V4 指定的随机动作、固定直行、固定转弯、无障碍到达、单障碍绕行和海流漂移测试通过；
3. 随机动作连续运行无 NaN/Inf；
4. 观测值始终位于声明范围；
5. 碰撞、边界、深度、振荡与超时分类正确；
6. 生成的场景通过可通行性检查；
7. 三维轨迹和奖励分量能够正常保存。

## 项目结构

```text
RL-ZL/
├── configs/stage0.yaml
├── configs/stage1_teacher.yaml
├── scripts/
│   ├── check_device.py
│   ├── run_stage0_validation.py
│   ├── run_stage1_smoke.py
│   ├── train_stage1_teacher.py
│   └── evaluate_stage1_teacher.py
├── src/rl_zl/
│   ├── config.py
│   ├── current.py
│   ├── current_field.py
│   ├── dynamics.py
│   ├── environment.py
│   ├── obstacles.py
│   ├── remus_dynamics.py
│   ├── remus_env.py
│   ├── scenario.py
│   ├── replay.py
│   ├── sac.py
│   ├── evaluation.py
│   ├── training_config.py
│   └── training.py
└── tests/
```

## 科学边界

该环境是 REMUS-100-like planning-level model，不是实艇推进器、水动力参数辨识或海试系统。角速度、执行器时间常数、奖励系数和未由 V4 固定的 SAC 超参数均作为明确记录的仿真设计参数；正式结果必须冻结配置、训练/验证/测试种子与代码版本。Stage 1 只实现普通 SAC Teacher；PER、Small-SAC 和蒸馏属于后续阶段，不在本阶段提前混入。
