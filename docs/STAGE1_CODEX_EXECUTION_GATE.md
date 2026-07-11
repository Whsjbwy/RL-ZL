# REMUS100 RWPVSD-SAC Stage 1：Codex 实验执行、调参与验收规范

> 适用项目：`Whsjbwy/RL-ZL`
>
> 唯一方案依据：`REMUS100_RWPVSD_SAC_实验方案_V4_硬错误修订版.docx`
>
> 当前阶段：Stage 1——低难度 SAC/Teacher 跑通
>
> 本文件用途：直接交给 Codex，作为运行实验、诊断失败、调整参数、保存证据和判断是否进入 Stage 2 的强制执行规范。

---

## 0. 给 Codex 的最高优先级指令

你现在担任以下角色：

1. AUV 三维路径规划强化学习工程师；
2. REMUS-100-like planning-level model 实现工程师；
3. PyTorch SAC 训练工程师；
4. 实验数据审计员；
5. Stage 1 阶段验收裁判。

你的任务不是只让程序“能够启动”，而是严格依据最终 V4 实验方案，在已经通过的 Stage 0 环境基础上，完成 Stage 1 低难度 SAC Teacher 的训练、验证、诊断和调参，最终给出有数据支撑的“通过”或“不通过”结论。

必须遵守以下原则：

- 不得伪造成功率、碰撞率、训练曲线、checkpoint 或轨迹；
- 不得将 128 步 smoke run 当作正式训练结果；
- 不得删除失败回合，不得只统计成功轨迹；
- 不得为了提高成功率偷偷减少测试难度、修改测试种子或扩大目标半径；
- 不得同时修改多个关键参数后声称知道是哪一个参数起作用；
- 不得在 Stage 1 提前接入 PER、Small-SAC、蒸馏、LAC、HYCOM 或 OOD 实验；
- 不得绕过 Stage 0 的动力学、碰撞、ray、海流、奖励和终止逻辑重新写一个简化环境；
- 不得把 REMUS-100-like planning-level model 描述成真实 REMUS 100 精确水动力模型；
- 每次调参都必须保存原配置、修改原因、Git commit、随机种子、完整结果和失败类型；
- 最终测试集只能用于阶段验收，不能反复用于调参。

如果发现代码实现与 V4 文档冲突，先停止正式训练，指出冲突位置、影响和建议修正方案；完成修正并通过全部回归测试后才能继续训练。

---

## 1. 依据优先级与冲突处理

所有判断按以下优先级执行：

1. 最终 V4 实验方案；
2. 本文件列出的 Stage 1 固定口径；
3. `configs/stage1_teacher.yaml` 中已经记录的工程默认值；
4. 代码注释和 README；
5. Codex 自己提出的工程优化建议。

如果低优先级内容与高优先级内容冲突，以高优先级为准。对于 V4 没有明确固定的 SAC 超参数，可以采用工程默认值或进行受控调参，但必须明确标记为“工程选择”，不能写成“V4 规定”。

---

## 2. Stage 1 与 Stage 0 的关系

Stage 1 必须在 Stage 0 基础上增加训练代码，不能重新创建另一套实验环境。

Stage 0 已负责：

- 500 m × 500 m × 100 m 三维浅水环境；
- REMUS-100-like 欠驱动规划层运动学；
- 一阶速度、俯仰角速度和艏摇角速度响应；
- 球体、圆柱体和椭球体障碍物；
- 障碍物安全膨胀、碰撞检测和 signed distance；
- 26 条 heading-aligned body-frame 局部射线；
- 解析海流、局部涡旋和时变扰动；
- 53 维局部观测；
- 连续三维动作；
- 奖励、安全代价和失败分类；
- 场景可通行性预检查；
- 轨迹、ray 和环境单元测试。

Stage 1 只在上述基础上增加：

- 普通 SAC Teacher；
- uniform replay buffer；
- Stage 1A/1B curriculum；
- 训练日志；
- checkpoint；
- 固定验证；
- 100 episode 阶段验收；
- 失败类型统计和训练诊断。

Stage 1 不包含：

- PER-SAC 正式教师；
- Small-SAC；
- teacher rollout 蒸馏数据集；
- BC、Policy Distillation、P+Q Distillation；
- safety-cost distillation；
- RWPVSD-SAC 完整学生；
- OOD 泛化；
- 感知噪声和 ray dropout；
- HYCOM；
- LAC safety critic；
- 真实海试或推进器级控制。

这些内容属于 Stage 2 及后续阶段。普通 SAC Teacher 没有稳定通过 Stage 1 前，不允许提前加入 PER 来掩盖基础问题。

---

## 3. Stage 1 开始前的强制准入条件

在正式训练前依次执行以下检查。任何一项失败，都不得开始长训练。

### 3.1 代码版本检查

```powershell
cd "D:\RL+ZL"
git status
git log -3 --oneline
```

要求：

- 工作目录中的代码来源明确；
- 没有无法解释的未提交修改；
- 记录本次实验使用的 Git commit；
- 不允许在训练中途无记录地改代码。

### 3.2 Python、PyTorch 和 GPU 检查

```powershell
conda activate auv-rl
python scripts/check_device.py
```

正式 GPU 训练要求：

- Python 能正常启动；
- PyTorch 能正常导入；
- `CUDA available: True`；
- 至少识别到一个 CUDA 设备；
- GPU 名称应为用户实际设备，例如 NVIDIA GeForce RTX 5060；
- 记录 PyTorch 版本、CUDA runtime、GPU 型号和显存。

如果 `CUDA available: False`：

- 停止正式训练；
- 不要用修改代码的方式伪造 CUDA；
- 检查当前是否真的激活 `auv-rl`；
- 检查 PyTorch 是否为 CUDA 版本；
- 输出完整诊断信息，等待修复。

### 3.3 全部单元测试

```powershell
python -m unittest discover -s tests -v
```

要求：

- 全部测试通过；
- SAC actor、critic、checkpoint 和 replay 测试不得跳过；
- 如果因为没有 PyTorch 而跳过 SAC 测试，不得开始正式训练。

当前工程预期至少包含以下测试类别：

- configuration；
- current；
- dynamics；
- environment；
- evaluation gate；
- obstacle geometry；
- replay buffer；
- SAC actor/critic/update/checkpoint；
- scenario reproducibility；
- Stage 1 curriculum config。

### 3.4 Stage 0 回归测试

```powershell
python scripts/run_stage0_cases.py --config configs/stage0.yaml
python scripts/run_stage0_validation.py --config configs/stage0.yaml --episodes 20
```

要求：

- `all_passed: true`；
- `stage0_environment_gate_pass: true`；
- `numerical_failures: 0`；
- 固定直行、固定转弯、无障碍到达、单障碍绕行、海流漂移和 ray 几何全部正确；
- 观测保持在声明范围；
- 不允许为了 Stage 1 训练而破坏 Stage 0 已通过的行为。

### 3.5 Stage 1 smoke run

```powershell
python scripts/run_stage1_smoke.py
```

要求：

- `smoke_passed: true`；
- replay 中存在有效样本；
- SAC 的 `updates > 0`；
- checkpoint 文件存在且能够读取；
- actor/critic/alpha loss 有限；
- 无 NaN/Inf；
- smoke 的成功率不作为正式判断。

只有 3.1–3.5 全部通过，才能开始 Stage 1A。

---

## 4. 不允许修改的基础实验口径

以下内容属于环境和任务定义，不是为了提高成功率可以随意调整的普通超参数。

### 4.1 空间与深度

| 项目 | 固定设置 |
|---|---:|
| 三维水域 | 500 m × 500 m × 100 m |
| 合法深度 | 5–95 m |
| 标准最大步数 | 600 |
| 标准目标半径 | 8 m |
| Stage 1A/1B 目标半径 | 10 m |

Stage 1 的 10 m 目标半径是 curriculum 设置，不得进一步扩大来制造高成功率。进入 Stage 2 后必须恢复 8 m。

### 4.2 动力学与动作

策略输出的连续动作是：

```text
[u_d, q_d, r_d]
```

代码内部角度和角速度统一使用 rad 和 rad/s。

Stage 1 早期动作范围：

```text
q_d ∈ [-4°, 4°]/s
r_d ∈ [-6°, 6°]/s
```

动作变化率：

```text
|Δq_d| ≤ 2°/s
|Δr_d| ≤ 3°/s
```

不得在 Stage 1 未通过时扩展到完整 `±8°/s` 和 `±12°/s`，否则会增加急转、俯仰振荡和动力学失败。

### 4.3 局部感知

- 使用 26 条机体系对齐射线；
- 射线随当前航向和俯仰旋转；
- 每条射线查询最近膨胀障碍边界；
- 射线截断到传感器最大距离后归一化；
- 策略不得读取完整障碍地图；
- 粗栅格 A* 只用于剔除不可达地图，不得作为策略输入。

### 4.4 成功条件

成功必须同时满足：

- 到达当前 curriculum 的目标半径；
- 全过程无碰撞；
- 无边界越界；
- 无深度违规；
- 无动力学约束违规；
- 无振荡失败；
- 未超过最大步数。

不能把“最终距离较小”“接近目标”或“没有碰撞”单独算作成功。

### 4.5 失败类型

必须保留下列失败类型：

```text
collision
depth
boundary
dynamics
oscillation
timeout
```

失败优先级必须与环境实现一致，避免同一回合重复计入多个主要失败类别。失败回合不能删除，不能只保留成功轨迹。

---

## 5. Stage 1 curriculum 固定设置

### 5.1 Stage 1A：基础到达能力

| 项目 | 设置 |
|---|---|
| 海流 | 无海流 |
| 障碍数量 | 4–6 |
| 障碍类型 | 球体、圆柱体 |
| 障碍半径 | 8–15 m |
| 目标半径 | 10 m |
| 目的 | 证明普通 SAC 能在基础三维环境学会到达和避障 |

Stage 1A 不允许：

- 涡旋；
- 时变海流；
- 8–12 个标准障碍；
- 椭球体加密 OOD；
- 8 m 目标半径；
- 感知噪声；
- PER。

### 5.2 Stage 1B：弱流补偿能力

| 项目 | 设置 |
|---|---|
| 海流 | 0–0.15 m/s 弱背景流 |
| 涡旋 | 关闭 |
| 时变扰动 | 关闭 |
| 障碍数量 | 6–8 |
| 障碍类型 | 球体、圆柱体 |
| 障碍半径 | 8–15 m |
| 目标半径 | 10 m |
| 目的 | 验证弱流补偿和低难度避障能力 |

只有 Stage 1A 达标后才允许进入 Stage 1B。不得同时训练 Stage 1A 和 Stage 1B 后再倒推 Stage 1A 是否通过。

### 5.3 关于 V4 中的 Stage 1C

V4 curriculum 表还给出了 Stage 1C：背景流 + 简单涡旋、8–10 个障碍、8 m 目标半径，用于过渡到标准任务。但 V4 的阶段化执行计划把 Stage 1 的核心验收定义为“无流或 0–0.15 m/s 弱流、4–6 个障碍、10 m 目标半径、Teacher 成功率 >90%”。

因此当前执行顺序是：

1. 先完成并验收 Stage 1A；
2. 再完成并验收 Stage 1B；
3. Stage 1A/1B 未通过，不得启用 Stage 1C；
4. Stage 1C 作为进入 Stage 2 前的过渡实验，需单独配置、单独保存结果，不得与 Stage 1A/1B 的通过结果混在一起；
5. 未经用户确认，不自动进入 Stage 2。

---

## 6. SAC Teacher 固定结构与当前工程默认值

### 6.1 V4 固定网络结构

Teacher Actor：

```text
state(53) → 256 → 256 → 128 → μ, logσ
```

Teacher Critic Q1/Q2：

```text
state(53) + action(3) → 256 → 256 → 128 → Q
```

Actor 使用连续高斯策略并通过 tanh 映射到 `[-1, 1]^3`。

Stage 1 不得缩小 Teacher 网络。学生网络压缩属于 Stage 3，不属于 Stage 1 调参。

### 6.2 当前工程默认值

以下是当前 `configs/stage1_teacher.yaml` 中的可复现工程默认值。V4 未全部逐项固定这些数值，因此它们必须被标记为工程基线：

| 参数 | 当前值 |
|---|---:|
| actor learning rate | 3e-4 |
| critic learning rate | 3e-4 |
| alpha learning rate | 3e-4 |
| gamma | 0.99 |
| tau | 0.005 |
| initial alpha | 0.2 |
| entropy tuning | 自动 |
| target entropy | `-action_dim` |
| replay capacity | 1,000,000 |
| batch size | 256 |
| random warm-up | 10,000 transitions |
| update start | 10,000 transitions |
| updates per environment step | 1 |
| gradient clipping | 10.0 |
| validation interval | 10,000 steps |
| trend validation | 20 episodes（仅筛查，不得晋级） |
| confirmation interval | 50,000 steps |
| curriculum confirmation | 100 episodes（独立种子） |
| independent final test | 100 episodes（第三组独立种子） |
| post-resume replay warm-up | 10,000 transitions |

V4 对标准强 Teacher 建议总训练量约为 `1.0×10^6–2.0×10^6 transitions`。Stage 1A/1B 是低难度 curriculum，当前代码为每个子阶段设置了工程训练预算。若在现有预算内没有达标，不能直接宣布方案失败，应先按本文规定的诊断顺序检查环境、奖励和训练稳定性。

### 6.3 SAC 实现必须具备的正确性

- Twin Q critics；
- target critics；
- `min(Q1, Q2)` 抑制过估计；
- entropy-regularized Bellman target；
- tanh Gaussian log-probability Jacobian 修正；
- reparameterization trick；
- automatic entropy tuning；
- Polyak soft update；
- time-limit truncation 不作为真实 terminal；
- collision/depth/boundary/dynamics 等真实失败作为 terminal；
- checkpoint 包含 actor、critic、target critic、优化器、alpha 和阶段元数据；
- 恢复训练时回到正确 curriculum 阶段；
- 所有 loss 和参数保持有限。

---

## 7. 奖励函数与允许调节范围

### 7.1 V4 奖励组成

总奖励包括：

```text
progress
goal
collision
depth
clearance
current
energy
smooth
step
```

当前基线系数：

| 奖励项 | 基线值/范围 |
|---|---:|
| progress 系数 | 40，V4 建议 30–50 |
| goal | +400 |
| collision | -400 |
| depth | -250 |
| clearance | 20 |
| current | 0.2 |
| energy | 0.05 |
| smooth | 0.02 |
| step | -0.01 |
| 非终止稠密奖励裁剪 | [-5, 5] |

### 7.2 奖励调参规则

允许优先调整：

1. `progress` 在 V4 建议的 30–50 范围内调整；
2. 当动作振荡明显时，小幅提高 smooth penalty；
3. 当近障碍碰撞明显但目标到达能力正常时，检查 clearance 计算后再小幅调整 clearance penalty；
4. 保持所有方法使用相同奖励口径；
5. 每次只调整一个奖励组，并保留前后对比。

不得执行：

- 把目标奖励提高到极端数值以掩盖稠密奖励错误；
- 删除碰撞惩罚；
- 关闭深度失败；
- 取消 reward clipping 后不检查 Q 值爆炸；
- 为 Teacher 单独使用更有利奖励，而后续 Small-SAC 使用另一套奖励；
- 在 Stage 1A 无海流时用海流奖励制造非零信号；
- 通过扩大目标半径替代奖励调试。

### 7.3 海流奖励反作弊

海流奖励只能在以下两项同时成立时产生：

1. 海流方向有助于目标方向；
2. AUV 实际距离目标更近。

如果出现顺流绕圈、累计海流奖励高但目标距离不下降，应判定为奖励黑客，不能把高 return 当作策略改善。

---

## 8. 训练、验证和最终测试的严格隔离

必须区分四类数据：

### 8.1 训练场景

- 用于采集 replay 和更新网络；
- 场景种子可随机生成；
- 不用于最终成功率报告。

### 8.2 固定趋势验证场景

- 用于 checkpoint 选择和调参；
- 种子冻结；
- 每次评估使用同一组验证任务；
- 不允许看到结果后替换“难种子”。

20 episode 趋势验证只用于发现候选 checkpoint。样本量不足以作为 curriculum
晋级证据，无论成功率多高都不能直接晋级。

### 8.3 Curriculum 确认场景

- 与趋势验证使用不同的固定 seed block；
- 每次至少 100 episodes；
- 只有确认成功率严格 `>90%` 且碰撞率 `≤10%` 才允许晋级；
- 反复确认至少间隔 50,000 个训练步，避免用确认集进行高频选择。

### 8.4 最终测试场景

- 与训练种子、验证种子不重叠；
- 只有配置冻结后才运行；
- 至少 100 episodes；
- 正式结果建议 300 episodes；
- 不得用最终测试集反复调参。

在开始正式实验前，Codex 必须审计代码是否提供 `validation_seed`、
`confirmation_seed` 和 `final_test_seed` 三个互不重叠的 seed block。若复用，应先修复并
增加测试。该修改只能改变数据划分，不能改变环境难度。

---

## 9. Stage 1 分层验收门槛

### Gate A：工程链路通过

必须全部满足：

- 单元测试全部通过；
- Stage 0 回归全部通过；
- Gymnasium 环境检查通过；
- smoke run 通过；
- replay、SAC update、checkpoint 无异常；
- 无 NaN/Inf。

Gate A 只说明代码能训练，不说明 Teacher 已学会任务。

### Gate B：Stage 1A 通过

必须在固定 Stage 1A 测试配置上满足：

- 最终评估 episodes ≥100；
- 成功率严格 `>90%`；
- 不是 `≥90%`，恰好 90% 仍不通过；
- 碰撞率 `≤10%`；
- 数值失败为 0；
- 失败类型统计完整；
- 没有修改目标半径、障碍数量和测试种子来提高结果；
- 保存通过 Gate B 的 checkpoint、配置和评估明细。

### Gate C：Stage 1B 通过

必须在固定 Stage 1B 弱流配置上满足：

- 最终评估 episodes ≥100；
- 成功率严格 `>90%`；
- 碰撞率 `≤10%`；
- 数值失败为 0；
- 失败类型统计完整；
- 弱流范围确实为 0–0.15 m/s；
- 无涡旋、无时变强流；
- 保存通过 Gate C 的 checkpoint、配置和评估明细。

### Gate D：多随机种子复核

V4 要求所有算法至少使用 3 个随机种子；计算资源允许时正式结果使用 5 个随机种子。

因此：

- 单个种子通过只能说明工程主链路可行；
- 至少 3 个训练种子后才能声称 Stage 1 结果稳定；
- 每个种子分别报告成功率、碰撞率、超时率和失败分布；
- 汇总报告 mean ± std；
- 不能只选择最好的 seed；
- 某个 seed 明显失败时必须保留并分析。

只有 Gate A、B、C 和多种子复核均完成，才可以向用户建议进入 Stage 2。未经用户确认，不自动启动 Stage 2。

---

## 10. 每次实验必须保存的数据

每个 run 必须使用独立目录，禁止覆盖旧结果。建议：

```text
artifacts/stage1_teacher/runs/
├── stage1a_seed20260710_baseline/
├── stage1a_seed20260711_tune_progress45/
├── stage1b_seed20260710_baseline/
└── ...
```

每个 run 至少保存：

```text
run_manifest.json
config_frozen.yaml
git_commit.txt
training_episodes.jsonl
training_updates.jsonl
validation_history.jsonl
summary.json
evaluation_final_100.json
checkpoints/latest.pt
checkpoints/best.pt
checkpoints/final.pt
representative_success_trajectory.*
representative_collision_trajectory.*
representative_timeout_trajectory.*
```

`run_manifest.json` 至少记录：

- run 名称；
- 开始和结束时间；
- Git commit；
- 是否存在未提交修改；
- Python 版本；
- PyTorch 版本；
- CUDA runtime；
- GPU 名称和显存；
- Stage 1A 或 Stage 1B；
- 完整配置路径及配置 SHA256；
- 训练 seed；
- 验证 seed 列表；
- 最终测试 seed 列表；
- 总 transitions；
- 总 updates；
- checkpoint 选择规则；
- 是否发生中断和恢复；
- 最终 gate 结论。

训练曲线至少记录：

- episode return；
- success rate；
- collision rate；
- timeout rate；
- depth/boundary/dynamics/oscillation 失败率；
- episode length；
- goal distance；
- minimum obstacle clearance；
- actor loss；
- critic loss；
- alpha loss；
- entropy/alpha；
- Q1/Q2 mean；
- target Q mean；
- replay size；
- environment steps；
- gradient updates。

---

## 11. Codex 调参闭环

每一轮必须执行以下闭环，不得跳步。

### 第一步：读取证据

读取：

- 冻结配置；
- 最近训练日志；
- 验证结果；
- 失败类型分布；
- actor/critic/alpha 曲线；
- 目标距离曲线；
- 代表性失败轨迹。

### 第二步：确定唯一主要问题

每轮只能选择一个主要问题，例如：

- 学不会目标方向；
- 碰撞率过高；
- 深度失败过高；
- 动力学失败过高；
- timeout 过高；
- Q 值爆炸；
- entropy 长期过高；
- 策略过度保守；
- 验证波动过大。

### 第三步：提出一个可证伪假设

示例：

```text
假设：Stage 1A timeout 高是因为 progress 信号偏弱，而不是障碍物不可达。
证据：A* 可行率为 100%，碰撞率低，但多数回合目标距离持续缓慢下降后超时。
修改：只把 progress 从 40 调到 45，其他参数保持不变。
预期：timeout 下降、success 上升，collision 不明显上升。
```

### 第四步：一次只修改一个参数组

参数组定义：

- 环境 curriculum；
- 奖励尺度；
- 动作平滑；
- learning rate；
- batch/replay/warm-up；
- entropy；
- target update；
- 网络结构。

Stage 1 原则上不修改网络结构。环境 curriculum 只能按 Stage 1A→Stage 1B 顺序变化，不能用于“调出”好看的测试结果。

### 第五步：先短诊断，再完整训练

短诊断只验证：

- loss 是否有限；
- Q 值是否异常；
- replay 是否更新；
- 目标距离是否出现学习趋势。

短诊断不能用于 Gate B/C。只有完整预算和最终 100 episode 测试才能用于阶段通过判断。

### 第六步：生成对比报告

每轮调参后必须输出：

| 字段 | 内容 |
|---|---|
| run_id | 唯一实验编号 |
| parent_run | 对照实验编号 |
| 唯一修改 | 参数及前后值 |
| 修改原因 | 由什么证据触发 |
| 训练预算 | transitions/updates |
| success | 前后变化 |
| collision | 前后变化 |
| timeout | 前后变化 |
| 其他失败 | 前后变化 |
| Q/entropy | 是否稳定 |
| 结论 | 保留/回退/继续验证 |

没有对照数据，不得声称某次调参有效。

---

## 12. 按失败现象进行诊断和调整

### 12.1 `dynamics` 失败率高

优先检查：

1. 动作是否已经裁剪到 `[-1,1]`；
2. 归一化动作到 `u_d/q_d/r_d` 的映射是否正确；
3. degree 与 rad 是否混用；
4. `q_d/r_d` 是否仍为 Stage 1 收缩范围；
5. `Δq_d/Δr_d` 限制是否真正生效；
6. pitch clip 前的 hard violation 是否被大量触发；
7. actor 输出是否长期饱和在 ±1。

允许调整：

- 检查并修复单位或动作映射错误；
- 在 V4 范围内增强平滑惩罚；
- 降低 actor learning rate；
- 检查 entropy 是否过高。

禁止：

- 关闭 dynamics failure；
- 扩大物理限制；
- 把所有动作强行设为 0；
- 删除失败回合。

### 12.2 `collision` 失败率高

优先检查：

1. ray 几何和障碍可视化；
2. body-frame 到 world-frame 旋转；
3. 膨胀障碍和真实障碍距离是否混用；
4. clearance penalty 是否在安全距离内产生；
5. 动作是否振荡；
6. 碰撞是否集中于特定形状或特定方向射线；
7. 目标方向奖励是否压过避障信号。

允许调整：

- 修复几何 bug；
- 小幅增加 clearance penalty；
- 小幅增加 smooth penalty；
- 降低 exploration entropy 或 actor learning rate；
- 延长同一合法 curriculum 的训练预算。

禁止：

- 减少最终测试障碍数量；
- 缩小障碍半径；
- 取消安全膨胀；
- 只统计没有碰撞的回合。

### 12.3 `depth` 失败率高

优先检查：

- z 轴正方向定义是否一致；
- 合法深度 5–95 m 是否正确；
- 目标方向 body-frame 的垂向分量；
- pitch 符号；
- 深度边界观测；
- depth penalty；
- 垂向海流是否在 Stage 1A/B 被错误启用。

禁止通过扩大合法深度范围解决。

### 12.4 `boundary` 失败率高

优先检查：

- x/y 边界距离观测；
- 起终点 margin；
- 边界安全代价；
- 目标位置是否在合法范围；
- 弱流是否持续把 AUV 推向边界。

不得扩大 500 m × 500 m 环境边界。

### 12.5 `timeout` 失败率高

区分两种情况：

1. 目标距离持续下降但速度太慢；
2. 目标距离反复变化或基本不下降。

情况 1 优先检查：

- progress 奖励是否偏弱；
- speed action 是否过于保守；
- clearance/safety 是否过强；
- 路径预算和 A* 可行性。

情况 2 优先检查：

- observation 索引；
- goal direction；
- reward hacking；
- entropy；
- actor/critic 是否真正更新；
- Q 值是否失真。

不得直接增加最大步数来掩盖策略不会到达的问题。最大步数 600 属于固定任务口径。

### 12.6 `oscillation` 失败率高

检查：

- 动作变化率限制；
- smooth penalty；
- actor 输出饱和；
- entropy 温度；
- 目标进展奖励是否允许反复靠近—远离刷分。

如出现进展奖励黑客，启用或实现 V4 预留的 best-distance progress 防护，但必须对所有后续基线统一使用并记录版本变化。

### 12.7 critic loss/Q 值爆炸

检查：

- 终止奖励是否重复加入；
- reward clipping 是否生效；
- `terminated` 与 `truncated` 是否混淆；
- target critic 是否停止梯度；
- tau 是否正确；
- 学习率是否过高；
- replay 中是否存在 NaN/Inf；
- observation/action 是否越界。

调整顺序：

1. 修复实现错误；
2. 检查奖励尺度；
3. 降低 learning rate；
4. 检查 gradient clipping；
5. 再考虑 batch size。

不得先改网络结构逃避数值错误。

### 12.8 entropy 长期过高或过低

长期过高的表现：动作随机、急转、碰撞或 dynamics failure 多。

长期过低的表现：策略过早确定、停滞在次优行为、不同场景适应差。

优先检查自动 entropy tuning 是否正确，target entropy 是否为 `-action_dim`。只有确认实现正确后，才允许调整 initial alpha 或 alpha learning rate。

### 12.9 策略过于保守

表现：碰撞率低但 timeout 高、路径长、速度低、离障碍过远。

检查：

- clearance penalty；
- 安全代价是否错误进入 Stage 1 SAC reward；
- energy penalty；
- entropy；
- progress reward。

Stage 1 的规则安全代价主要用于记录和后续蒸馏，不得在没有方案依据的情况下作为巨大额外惩罚压制 Teacher。

---

## 13. 推荐的调参优先级

必须按以下优先级排查，前一层未确认前不要跳到后一层。

### 优先级 1：实现正确性

- 单位；
- observation；
- action mapping；
- collision/ray；
- reward；
- terminal/truncation；
- replay；
- SAC Bellman target；
- checkpoint。

### 优先级 2：curriculum 难度

- Stage 1A 无流；
- Stage 1B 弱背景流；
- 不提前加涡旋和标准障碍密度。

### 优先级 3：V4 奖励范围内微调

- progress 30–50；
- smooth；
- clearance；
- 检查 dense clipping。

### 优先级 4：SAC 稳定性参数

- learning rate；
- warm-up；
- batch size；
- alpha；
- target update。

### 优先级 5：训练预算

只有前面均正常但学习仍在持续改善时，才增加训练 transitions。

### 不属于 Stage 1 的“解决办法”

- PER；
- 更大网络；
- 蒸馏；
- LAC；
- 全局规划器直接给策略路径；
- 修改测试难度；
- 只选择有利 seed。

---

## 14. 本地运行命令

### 14.1 更新与安装

```powershell
cd "D:\RL+ZL"
git pull origin main
conda activate auv-rl
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

### 14.2 准入验证

```powershell
python scripts/check_device.py
python -m unittest discover -s tests -v
python scripts/run_stage0_cases.py --config configs/stage0.yaml
python scripts/run_stage0_validation.py --config configs/stage0.yaml --episodes 20
python scripts/run_stage1_smoke.py
```

### 14.3 正式训练

```powershell
python scripts/train_stage1_teacher.py `
  --config configs/stage1_teacher.yaml `
  --device cuda `
  --output-dir artifacts/stage1_teacher/runs/baseline_seed20260710
```

### 14.4 中断恢复

checkpoint 不包含 replay 内容。恢复后程序必须先收集 10,000 条新 transition，期间只
执行策略、不做梯度更新。旧版 checkpoint 若没有 Stage 1A 的正式确认元数据，程序必须
先补做独立 100 episode Stage 1A 确认，通过后才可续训 Stage 1B。

```powershell
python scripts/train_stage1_teacher.py `
  --config configs/stage1_teacher.yaml `
  --device cuda `
  --output-dir artifacts/stage1_teacher/runs/baseline_seed20260710 `
  --resume artifacts/stage1_teacher/runs/baseline_seed20260710/checkpoints/latest.pt
```

### 14.5 最终 100 episode 测试

在最终测试 seed 已与验证 seed 分离的前提下运行：

```powershell
python scripts/evaluate_stage1_teacher.py `
  --config configs/stage1_teacher.yaml `
  --checkpoint artifacts/stage1_teacher/runs/baseline_seed20260710/checkpoints/final.pt `
  --curriculum-index -1 `
  --episodes 100 `
  --seed-split final_test `
  --output artifacts/stage1_teacher/runs/baseline_seed20260710/evaluation_final_100.json
```

如需审计复现，可用 `--base-seed` 显式覆盖；该值会写入输出文件。正式结果不得在查看
结果后更换 seed。

---

## 15. Codex 每轮必须提交的阶段报告

每轮训练后按以下格式输出，不得只回复“训练完成”。

```markdown
# Stage 1 实验报告

## 1. 运行身份
- run_id：
- Git commit：
- config SHA256：
- Stage：1A / 1B
- seed：
- device：
- PyTorch/CUDA：

## 2. 训练规模
- transitions：
- gradient updates：
- episodes：
- wall-clock time：
- 是否发生恢复训练：

## 3. 固定验证结果
- episodes：
- success rate：
- collision rate：
- timeout rate：
- depth failure rate：
- boundary failure rate：
- dynamics failure rate：
- oscillation failure rate：
- mean return：
- mean successful path length：
- mean successful travel time：
- mean successful energy proxy：
- mean/minimum clearance：

## 4. 训练稳定性
- actor loss：
- critic loss：
- alpha/entropy：
- Q1/Q2：
- target Q：
- NaN/Inf：

## 5. 主要失败模式
- 最大失败类型：
- 占比：
- 代表性轨迹：
- 初步根因：

## 6. 本轮唯一修改
- 参数：
- 旧值：
- 新值：
- 修改依据：
- 预期影响：
- 实际影响：

## 7. Gate 判断
- Gate A：PASS/FAIL
- Gate B：PASS/FAIL/NA
- Gate C：PASS/FAIL/NA
- 是否允许进入下一阶段：是/否

## 8. 下一步
- 保留当前配置 / 回退 / 单参数继续验证
- 下一轮只修改：
```

---

## 16. Stage 1 最终通过清单

Codex 只有在所有方框都有真实证据时才能写“Stage 1 通过”。

- [ ] Git commit 和冻结配置已保存；
- [ ] Python/PyTorch/CUDA/GPU 信息已保存；
- [ ] 全部单元测试通过；
- [ ] SAC 测试没有因缺少 PyTorch 被跳过；
- [ ] Stage 0 回归通过；
- [ ] Stage 1 smoke 通过；
- [ ] Stage 1A 使用无流、4–6 障碍、10 m 目标半径；
- [ ] Stage 1A 最终测试 ≥100 episodes；
- [ ] Stage 1A 成功率严格 >90%；
- [ ] Stage 1A 碰撞率 ≤10%；
- [ ] Stage 1B 使用 0–0.15 m/s 弱背景流；
- [ ] Stage 1B 使用 6–8 障碍、10 m 目标半径；
- [ ] Stage 1B 最终测试 ≥100 episodes；
- [ ] Stage 1B 成功率严格 >90%；
- [ ] Stage 1B 碰撞率 ≤10%；
- [ ] 数值失败为 0；
- [ ] 训练、验证和最终测试 seed 分离；
- [ ] 所有失败回合均保留；
- [ ] failure counts 完整；
- [ ] checkpoint 可加载并复现确定性动作；
- [ ] 至少 3 个训练 seed 已完成；
- [ ] 报告 mean ± std；
- [ ] 没有提前加入 PER、蒸馏或 OOD；
- [ ] 没有为了通过而修改固定测试难度；
- [ ] 用户已看到 Stage 1 报告并明确同意进入 Stage 2。

任意一项缺失，结论只能写：

```text
Stage 1 尚未完全通过；当前已完成到 ______，主要阻塞是 ______。
```

---

## 17. 最终停止条件

出现以下任意情况，立即停止当前长训练并诊断：

- NaN/Inf；
- critic loss 持续爆炸；
- Q 值数量级持续无界增长；
- replay 中存在非有限值；
- GPU 未实际启用；
- Stage 0 回归失败；
- 环境配置与 V4 冲突；
- 大量场景不可达；
- collision/ray 可视化不一致；
- 训练/验证/test seed 泄漏；
- 日志或 checkpoint 无法保存；
- 代码在训练期间发生未记录变化；
- 通过删除失败数据才能得到高成功率。

停止后必须先形成根因报告，不得无目的地连续改参数。

---

## 18. 给 Codex 的最终执行命令

请严格执行以下任务：

1. 从头审计 `RL-ZL` 当前 Stage 0 和 Stage 1 代码；
2. 对照 V4 和本文件核对环境、动作、观测、奖励、终止、curriculum 和 SAC；
3. 先运行全部准入测试；
4. 检查训练、验证和最终测试 seed 是否隔离；若未隔离，先修复并增加测试；
5. 创建不可覆盖的 baseline run 目录和 run manifest；
6. 先训练 Stage 1A；
7. 使用固定验证集诊断，不使用最终测试集调参；
8. 若不达标，按照失败类型一次只修改一个参数组；
9. 每次修改都保留父实验、配置差异和指标对比；
10. Stage 1A 通过后再训练 Stage 1B；
11. Stage 1B 配置冻结后运行独立 100 episode 最终测试；
12. 至少完成 3 个训练 seed，并报告 mean ± std；
13. 按本文件模板生成 Stage 1 阶段报告；
14. 没有满足全部 Gate 时，不得进入 Stage 2；
15. 在执行任何会覆盖旧结果、删除 checkpoint、修改正式配置或推送远程仓库的操作前，先向用户说明影响并取得确认。

最终目标不是“把成功率调高”，而是在不改变任务口径、不泄漏测试集、不删除失败数据的前提下，获得可复现、可审计、符合 V4 的 SAC Teacher，并为 Stage 2 标准环境 Teacher 提供可靠起点。
