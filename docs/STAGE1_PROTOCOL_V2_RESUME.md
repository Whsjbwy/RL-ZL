# Stage 1 protocol-v2 修复与本次 checkpoint 恢复说明

## 结论

本次先修复实验控制代码，不修改 SAC 超参数、奖励、动作范围、环境难度、海流、障碍物或
目标半径。原因是 seed `20260710` 的旧结果存在明显的验收口径问题：20 回合验证得到
`19/20=95%`，但同一 checkpoint 的 100 回合结果只有 `83/100=83%`。因此不能根据
20 回合结果继续调参或宣布 Stage 1B 通过。

protocol-v2 强制执行：

1. 20 回合仅作趋势筛查，不能直接晋级；
2. curriculum 晋级必须通过另一 seed block 的 100 回合确认；
3. 最终 checkpoint 再使用第三个独立 seed block 测试 100 回合；
4. 旧 checkpoint 恢复时，先补做 Stage 1A 的 100 回合正式确认；
5. checkpoint 不含 replay，恢复后先收集 10,000 条新 transition，期间不更新网络；
6. 修复恢复后的评估调度，避免在 minimum step 处连续触发 10 次“补课评估”；
7. 日志增加起终点、垂向距离、初始目标俯仰角、终端姿态/角速度、海流速度和动力学
   clipping 诊断。

## 固定参数（本轮禁止修改）

以下配置继续使用原基线：

- actor/critic：`256, 256, 128`；
- learning rate：`3e-4`；
- `gamma=0.99`，`tau=0.005`；
- batch size：`256`；
- Stage 1B 弱背景流：`0–0.15 m/s`；
- 障碍：`6–8` 个球体/圆柱体；
- 目标半径：`10 m`；
- 俯仰/艏摇命令范围、奖励系数和终止条件全部不变。

只有 protocol-v2 下继续训练到 Stage 1B 累计 300,000 steps 后仍未通过，才能根据新增
诊断决定是否进行单因素参数实验。不得现在同时修改学习率、奖励和动作范围。

## Windows 本地执行

在包含新代码的项目根目录执行：

```powershell
cd "D:\RL+ZL\RL-ZL_Stage1_V4"
conda activate auv_rl
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python scripts/check_device.py
python -m unittest discover -s tests -v
python scripts/run_stage0_cases.py --config configs/stage0.yaml
python scripts/run_stage0_validation.py --config configs/stage0.yaml --episodes 20
python scripts/run_stage1_smoke.py
```

准入要求：本地 PyTorch 已安装，因此全部测试必须通过，SAC 三项测试不能出现 `skipped`；
Stage 0 两项 gate 和 smoke 必须为 `true`。

先查找旧 checkpoint 的实际位置：

```powershell
Get-ChildItem artifacts -Recurse -Filter latest.pt |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 5 FullName, LastWriteTime
```

假设旧 checkpoint 位于：

```text
artifacts/stage1_teacher/runs/baseline_seed20260710/checkpoints/latest.pt
```

使用新目录保存 protocol-v2 结果，避免覆盖旧证据：

```powershell
python scripts/train_stage1_teacher.py `
  --config configs/stage1_teacher.yaml `
  --device cuda `
  --output-dir artifacts/stage1_teacher/runs/protocol_v2_resume_seed20260710 `
  --resume artifacts/stage1_teacher/runs/baseline_seed20260710/checkpoints/latest.pt
```

如果 checkpoint 在别处，只替换 `--resume` 后面的路径，不要移动或删除旧结果。

## 正常运行时应看到的事件

1. `resume_ready`：应显示旧 checkpoint 的 `total_steps=380000`、Stage 1B 索引和
   `gradient_updates_resume_at_step=390000`；
2. `resume_protocol_confirmation`：先运行旧 checkpoint 的 Stage 1A 100 回合确认；
3. Stage 1A 确认通过后继续 Stage 1B；若失败，程序会明确终止，不能绕过；
4. Stage 1B 在 global step 390,000 首次运行20回合趋势筛查，而不是在380,001开始连续评估；
5. 趋势候选通过时运行 `curriculum_confirmation` 100回合；
6. 最终运行 `independent_final_test` 100回合。

训练过程只在评估开始/结束时打印结构化进度，其他时间终端可能暂时没有新行。可另开一个
PowerShell 观察 GPU：

```powershell
nvidia-smi -l 5
```

## 停止与通过标准

- Stage 1A 补充确认：成功率严格 `>0.90`，碰撞率 `<=0.10`；
- Stage 1B curriculum 确认：100回合成功率严格 `>0.90`，碰撞率 `<=0.10`；
- Stage 1B 独立最终测试：100回合成功率严格 `>0.90`，碰撞率 `<=0.10`；
- `stage1_gate_passed` 必须为 `true`；
- 失败回合必须全部保留，不能删除 dynamics/depth/boundary/timeout；
- 单个 seed 通过仍只代表主链路可行，论文稳定性结论至少需要3个训练 seed。

## 跑完后打包

```powershell
$run = "artifacts/stage1_teacher/runs/protocol_v2_resume_seed20260710"
Compress-Archive -Path `
  "$run/summary.json", `
  "$run/training_episodes.jsonl", `
  "$run/training_updates.jsonl", `
  "$run/evaluations", `
  "$run/checkpoints/final.pt", `
  "$run/checkpoints/latest.pt" `
  -DestinationPath "stage1_protocol_v2_seed20260710.zip" -Force
```

把 `stage1_protocol_v2_seed20260710.zip` 发回分析。checkpoint 体积可能较大；若上传受限，
至少发送 `summary.json`、两个 JSONL 日志和完整 `evaluations` 目录，但不要只发截图。

