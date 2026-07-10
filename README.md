# RL-ZL：REMUS-100-like AUV Stage 0

本仓库严格依据 **《REMUS100_RWPVSD_SAC_实验方案_V4_硬错误修订版》** 实现 RWPVSD-SAC 实验的 **Stage 0：环境与动力学验证**。当前版本只验证仿真基础是否正确，不包含正式 SAC/PER-SAC 训练结果，也不生成或伪造论文数据。

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
├── scripts/
│   ├── check_device.py
│   └── run_stage0_validation.py
├── src/rl_zl/
│   ├── config.py
│   ├── current.py
│   ├── current_field.py
│   ├── dynamics.py
│   ├── environment.py
│   ├── obstacles.py
│   ├── remus_dynamics.py
│   ├── remus_env.py
│   └── scenario.py
└── tests/
```

## 科学边界

该环境是 REMUS-100-like planning-level model，不是实艇推进器、水动力参数辨识或海试系统。角速度、执行器时间常数、奖励系数等均作为明确记录的仿真设计参数；后续正式实验必须冻结配置、训练/验证/测试种子与代码版本。
