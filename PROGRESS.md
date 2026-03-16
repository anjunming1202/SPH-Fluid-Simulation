# GPU Fluid Simulation — Progress Summary
*Last updated: 2026-03-16 | Branch: `phase3/1m-route-d`*

---

## 当前状态

| 维度 | 状态 |
|------|------|
| 粒子规模 | 场景设置为 700k（可调），目标 1M |
| 排序算法 | O(N) Counting Sort（已替换 O(N log²N) Bitonic）|
| 边界处理 | **Akinci 2012 boundary particles**（最新，未完整验证）|
| FPS（700k） | ~30fps（ComputeForce 是瓶颈，排序已不再是瓶颈）|
| 稳定性 | 500k 稳定；1M 未充分测试 |
| 当前分支 | `phase3/1m-route-d` |

---

## Git 历史与里程碑

```
f635767  fix: Akinci 2012 boundary particles replace approximate ghost density  ← 最新
aa96929  feat: O(N) counting sort replaces O(N log²N) bitonic sort + stability monitor
7489ff9  feat: pressure/density clamping + Route D 500k scene (45x27)
92fea09  fix: autoRestDensity divisor 7→3.5 and scene params
059eac3  perf+stability: split-substep + Wendland C2 kernel
ea9cc01  feat: normalize near-pressure by targetDensity for scale-independent SPH
fc4a77e  feat: auto-compute targetDensity from particle count and bounds
1e29870  feat: Phase 2 — Bitonic Sort replaces atomic bucket grid (500k capable)
8dee90d  refactor: remove overengineering (Tait, LinearDrag), add CFL-adaptive dt
```

**分支结构：**
- `master` — 稳定 CPU 基线（Phase 1）
- `feature/500k-large-bounds` — Phase 2 中间实验分支
- `phase3/1m-route-d` — 当前工作分支（Phase 3，1M 粒子 GPU SPH）

---

## 关键文件

| 文件 | 作用 |
|------|------|
| `Assets/Scripts/Fluid2D/FluidSimGPU2D.cs` | 主控脚本：Init/SimStep/稳定性监控/Akinci 参数计算 |
| `Assets/Shaders/FluidCompute2D.compute` | 所有 SPH kernel：Density/Force/Integrate/RenderData |
| `Assets/Shaders/CountingSort.compute` | O(N) Counting Sort（Blelloch scan + scatter）|
| `Assets/Shaders/BitonicSort.compute` | 旧排序（保留作 fallback，useBitonicSort=false 禁用）|
| `Assets/Scenes/FluidGPU_Demo.unity` | 场景：700k 粒子，bounds≈36×20，substeps=1 |

---

## 已解决的核心问题与关键思路

### 1. cohesionPressure 无效 → autoRestDensity 除数错误
**问题：** 无论怎么调 cohesionPressure 都没有效果。
**根因：** autoRestDensity 公式除以 7（理论 Wendland 积分常数），但粒子在重力下只填满约半个 bounds，实际密度是理论的 2×。导致 targetDensity 太低，压力始终为正，cohesion 从不激活。
**修复：** 除数 7 → 3.5（经验值，与旧 q² kernel 用 6→3 同理）。

### 2. force substeps 为何无效（重要发现）
**根因：** `GravityPredict` 只在外层 substep 开始时更新 `_Predicted`。force loop 内多次 ComputeForce 读的都是相同的 `_Predicted`（未更新位置）→ 每次迭代力完全相同 → 等价于只做了一次。
**结论：** force substeps 在当前架构下对稳定性毫无帮助，应避免使用，改用外层 substeps 或 CFL-adaptive dt。

### 3. 500k 爆炸 → maxDensityRatio 密度钳制
**问题：** 500k 粒子时偶发爆炸（transient overlap → 密度暴涨 → 力暴涨 → 爆炸）。
**修复：** 在 ComputeForce 中对密度钳制：`di_c = min(di, targetDensity × maxDensityRatio)`（默认=4）。限制了瞬态重叠产生的力，消除了爆炸而不影响稳态。

### 4. Bitonic Sort → Counting Sort（性能）
**问题：** 1M 粒子时 Bitonic Sort 需要 ~210 次 GPU dispatch，成为瓶颈（30fps）。
**方案：** O(N) Counting Sort，6 次 dispatch：
1. ClearCounts → 2. CountParticles → 3. LocalScan（Blelloch，TILE_SIZE=1024）
4. GlobalScan（1 thread 顺序扫描 tile sums）→ 5. Finalize → 6. Scatter
**结果：** 排序从瓶颈消除，FPS 从 30→31-32fps（瓶颈转移到 ComputeForce/ComputeDensity，这是 SPH 不可避免的）。

### 5. 边界粒子堆积（Tensile Instability at Boundaries）
**问题：** 边界处形成紧密一排高能量粒子，持续振荡，"消除不掉的能量"。
**根因分析（3层）：**
1. **密度亏损（Boundary Deficiency）**：边界粒子有约半圆邻居缺失 → 密度低于 targetDensity → 压力为负
2. **Tensile Instability**：负压力 + cohesion → 粒子互相吸引聚集成一条线 → nearDensity 暴涨 → 弹射 → 循环
3. **Wall Ghost Force（已删）**：之前实现的 wall ghost force 相当于弹簧，与重力形成谐振子，持续泵入能量

**修复路径：**
- 删除 wall ghost force（消除能量输入源）
- 在 ComputeForce 近壁面时关闭 cohesion（`effCohesion=0`）
- 添加 wall viscous drag（`_WallFriction` 参数，能量耗散）
- **最终方案：Akinci 2012 boundary particles**（见下）

### 6. Akinci 2012 Boundary Particles（最新，未完整验证）
**方案来源：** Akinci et al. 2012, "Versatile Rigid-Fluid Coupling for Incompressible SPH"，SPlisHSPlasH/DualSPHysics 的标准方案。
**核心公式：**
```
ψ_b = targetDensity / (2 × Σ_k W(k × d_bp))
```
对 d_bp = h/2，Σ_k W = W(0) + 2×W(h/2) = 1 + 2×0.1875 = **1.375**，ψ_b = D/2.75

**验证（理论）：**
- 底部粒子(d=0)收到的 BP 密度：ψ_b × 1.375 = D/2 ✓
- 总密度 = 真实邻居(D/2) + Akinci BP(D/2) = D ✓

**实现：**
- ComputeDensity/ComputeForce：对每面墙解析迭代 boundary particles（k 循环，约 4 次/墙，无需额外 buffer）
- ComputeForce：BP 提供排斥压力（`wallP = max(0, pi_pressure)`）+ no-slip 粘度（`-vi × q × viscosityStrength × bpV`）
- C#：Init() 中计算 `_bpSpacing = h/2` 和 `_bpVolume`，通过 `SetStaticUniforms()` 每帧传递

---

## 当前已知问题 / 待验证

### ⚠️ Akinci BP 尚未在 Play Mode 充分验证
最后一次 commit（f635767）是本 session 最新成果，MCP 工具频繁掉线导致无法在 session 内充分测试。
**回来后第一件事：** 进入 Play Mode，检查：
1. 编译是否通过（看 console 有无 shader 编译错误）
2. 底部是否还有明显的密集积聚
3. 有无新的爆炸/不稳定现象
4. 稳定性监控日志（`[Stability] OK` 或 `[Stability] Warning`）

### ⚠️ _BPVolume 校准可能需要调整
理论上 ψ_b = D/2.75 对底部(d=0)精确。但对中间高度(d=h/4)稍微过补偿约 10%（分析见对话记录）。如果流体在底部形成"正压气泡"把粒子推离地面，说明 BP 过强，减小 `_bpVolume` 的系数（从 2.75 改为 3.0-3.5）。

### ⚠️ 1M 粒子性能未充分测试
FPS 目标：>60fps（1M 粒子）。当前 700k ≈ 30fps。
主要瓶颈：ComputeDensity + ComputeForce（O(N×k)，k≈40 邻居平均）。
**潜在优化路径：**
- 减小 smoothingRadius（更少邻居）
- 增大 bounds 降低密度
- ForceSubsteps=1（已是默认）
- 考虑 compute shader 的 warp divergence 优化

### ⚠️ for 循环上界可能越界
Akinci BP 循环中：
```hlsl
int k1 = (int)ceil((pi.x - _BoundsMin.x + h) / bpS);
```
如果粒子接近右边界，k1 可能很大，但 `if (bx > _BoundsMax.x) break;` 保护了实际访问。
验证：在极端情况下（粒子在角落）是否有 GPU 线程超时。

### ⚠️ Scene 文件参数不确定
scene 文件存储的是 1000000 粒子 + 64×38 bounds，但上次 Play 日志显示 700k + 36×20。
原因：用户在 Inspector 改了但没保存。回来后需确认场景参数并保存。

---

## 重要参数参考（700k baseline）

```
numParticles      = 700000
boundsSize        = (36, 20)    [Route D 缩放：100k×√7≈2.65]
smoothingRadius   = 0.05
substeps          = 1
forceSubsteps     = 1           [多次无效，见问题2]
targetDensity     = ~2.18       [autoRestDensity 自动计算]
pressureMultiplier= 450
nearPressureMultiplier = 8      [除以 targetDensity 后的 effective 值]
cohesionPressure  = 10
viscosityStrength = 0.1
collisionDamping  = 0.4
maxDensityRatio   = 4
maxVelocity       = 200
wallFriction      = 0           [Akinci no-slip 已处理]
_bpSpacing        = 0.025       [h/2，自动计算]
_bpVolume         = D/2.75      [Akinci 体积，自动计算]
```

---

## 稳定性监控使用方式

运行时每秒自动输出一条日志：
```
[Stability] OK — fps=31 pos0=(-17.96,9.97) speed=0.5 alerts=0
[Stability] Warning frame=xxx NaN=False OOB=False highSpeed=True pos=(...) speed=xxx (alert #1)
```
- `alerts` 累计警告次数
- `highSpeed` = 速度超过 maxVelocity × 0.707
- `NaN` = 粒子位置出现 NaN（已爆炸）
- 注意：fps=Infinity 出现在第一帧（smoothDeltaTime 还未稳定）

---

## 架构决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 排序 | Counting Sort（O(N)）| Bitonic 在 1M 时 210 dispatches 太慢 |
| 核函数 | Wendland C2（q⁴(1+4q')）| 消除配对不稳定性（pairing instability）|
| 密度公式 | WCSPH：P = B(ρ-ρ₀) | 简单稳定，Tait equation 被撤销（过于复杂）|
| 边界 | Akinci 2012 解析 BP | 物理正确的 kernel completeness，无额外 buffer |
| dt | CFL-adaptive | cs = √(B/ρ₀)，dtCFL = 0.4h/cs，自动防爆 |
| 自适应 ρ₀ | autoRestDensity（除数 3.5）| N/A/h 变化时自动保持平衡 |

---

## 回来继续时的 Checklist

```
[ ] git checkout phase3/1m-route-d
[ ] 打开 Unity 2022.3.34f1
[ ] 打开 Assets/Scenes/FluidGPU_Demo.unity
[ ] 在 Inspector 确认参数（见上表）
[ ] Play Mode → 检查 console 无编译错误
[ ] 观察稳定性日志 ~10 秒
[ ] 如果底部仍有明显积聚：调整 _bpVolume 系数（2.75 → 3.0-3.5）
[ ] 如果爆炸：减小 pressureMultiplier 或增大 maxDensityRatio
[ ] 记录 FPS 数据
[ ] 考虑推进到 1M 粒子测试
```

---

## 潜在的下一步

1. **验证 Akinci BP**（最高优先级）— Play Mode 测试稳定性和边界行为
2. **1M 粒子性能优化** — 当前 30fps，目标 60fps+
   - 可能需要 h 更大 + bounds 更大来降低每粒子邻居数
   - 研究 GPU occupancy 和 warp divergence
3. **障碍物（Obstacle）支持** — 目前 AABB 障碍可用，未测试
4. **可视化改进** — 速度/密度 debug 色彩模式已实现
5. **IISPH 或 DFSPH** — 如果 WCSPH 在 1M 时稳定性不满足，考虑不可压缩 SPH

---

*文档由 Claude Code (claude-sonnet-4-6) 生成于 phase3/1m-route-d 分支*
