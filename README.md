# 后坐力轨迹重建（Feature Matching + RANSAC）

脚本默认自动扫描整段视频并分析一轮完整弹匣射击：

1. 全片扫描右下角当前弹药数字，通过字形变化的周期性自动识别射击帧范围、射速、弹匣容量和射击数；再用动态规划细化分段，并把数字从 `n` 变为 `n-1` 的第一帧作为关键帧。
2. 只在瞄准镜外的墙面区域提取 SIFT（或 ORB）特征；瞄准镜、枪体、HUD 和高亮火花均被屏蔽。
3. 相邻帧用 KNN Feature Matching + Lowe ratio test，随后以 RANSAC 剔除火花等动态外点，再把内点拟合为无缩放的平移+旋转 SE(2)。
4. 在每个关键帧内独立检测准星：2× 镜使用黑色横、纵刻度线交点及其小角度旋转，93R/G18 的 1× 镜使用紧凑红点检测。强烈火花不会直接决定准星位置。
5. 把关键帧的实际刻度线交点逆变换到自动识别区间的起始帧坐标系，得到后坐力轨迹。
6. 根据关键帧和视频 FPS 生成首发为 0 的 `shot_time_ms`；根据像素轨迹、FOV 和镜倍估算最大俯仰角。

核心公式为：

```text
A_i : frame(i-1) 的镜外背景 -> frame(i)
C_i = C_(i-1) @ inverse(A_i) : frame(i) -> analysis_start_frame
p_i = C_i @ [reticle_x_i, reticle_y_i, 1]
```

因此最终点 `p_i` 同时包含镜外画面的平移/旋转和准星自身的帧内抖动；不会把关键帧间的画面变换直接当成弹着点。

## 安装与运行

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\analyze_recoil.py "D:\DF\RM277_x1_opx2.mp4"
```

本机实际找到的视频是 `D:\DF\RM277_x1_opx2.mp4`，而不是 `D:\DF\RM277\_x1\_opx2.mp4`。它已经是脚本的默认输入，所以也可以直接运行：

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py
```

例如自动分析 AR57，并写入独立目录：

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py "D:\DF\AR57.mp4" `
  --output-dir .\recoil_output_ar57
```

自动模式假设视频包含一轮从满弹匣连续打到 `0` 的射击，因此识别到的连续数字变化次数就是弹匣容量。若视频只录了部分弹匣，必须同时提供四个手动参数：

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py VIDEO.mp4 `
  --start-frame 400 --end-frame 1000 --start-ammo 45 --shot-count 45
```

## 主要输出

- `recoil_output/keyframes_recoil.csv`：自动识别的弹药变化关键帧、`shot_time_ms`、准星屏幕位置、统一坐标、每发位移、背景分量、准星抖动分量和旋转。
- `recoil_output/all_frames_motion.csv`：自动射击区间每一帧的匹配数、RANSAC 内点、内点率、重投影误差、平移和旋转。
- `recoil_output/ammo_detection.csv`：全片每帧弹药字形变化分数、自动阈值和粗定位事件。
- `recoil_output/recoil_trajectory.png`：后坐力轨迹图（右/上为正）。
- `recoil_output/reticle_keyframes_contact_sheet.jpg`：45 个关键帧的刻度线交点检测复核图。
- `recoil_output/feature_mask.png`：绿色为镜外 RANSAC 实际使用区域。
- `recoil_output/summary.json`：自动弹匣数、帧范围、关键帧、发射时间、最低质量指标以及 FOV/镜倍 pitch 估算。

CSV 中可直接使用：

- `recoil_x_right_px`, `recoil_y_up_px`：相对分析起始帧的累计后坐力点。
- `shot_time_ms`：以首发为 0，根据弹药变化关键帧和视频 FPS 计算的发射时间。
- `shot_delta_x_right_px`, `shot_delta_y_up_px`：当前发相对上一发的位移。
- `background_only_*`：若准星固定在画面中心时，仅镜外运动产生的轨迹。
- `reticle_jitter_contribution_*`：准星帧内位置变化额外贡献的分量。

## 校准参数

默认 ROI 和掩膜已按这段 2560×1440 视频标定。若弹药位置改变，用：

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py VIDEO.mp4 --ammo-roi x0,y0,x1,y1
```

如果某些帧的运动质量偏低，先查看 `all_frames_motion.csv` 的 `status`、`inliers`、`inlier_ratio` 和 `median_reprojection_error_px`，再调整：

```text
--feature-scale 0.75
--max-features 3500
--ratio-test 0.78
--ransac-threshold 2.5
```

脚本不会静默接受 RANSAC 失败的帧：失败步长会标记为 `interpolated`，数量同时写入 `summary.json`。

默认 pitch 参数与 Recoil Trainer 一致：游戏配置 FOV 为 104° 的 4:3 参考横向 FOV，瞄准镜为 2×。可覆盖：

```powershell
.\.venv\Scripts\python.exe .\analyze_recoil.py VIDEO.mp4 `
  --fov-deg 104 --fov-axis reference-horizontal --scope-magnification 2
```

换算使用针孔投影，不是简单的“像素比例乘 FOV”。`reference-horizontal` 会先按镜倍缩小 ADS FOV，再按 Recoil Trainer 的 4:3 参考模型换算纵向焦距。

## 转换为 Recoil Trainer Profile JSON

`convert_to_recoiltrainer.py` 会读取累计轨迹列 `recoil_x_right_px`、`recoil_y_up_px`，生成 Recoil Trainer 的 `WeaponProfile` JSON。它不会把准星抖动丢掉，因为这两个累计坐标已经同时包含镜外画面运动与刻度线在帧内的位置变化。

默认转换当前识别结果：

```powershell
.\.venv\Scripts\python.exe .\convert_to_recoiltrainer.py
```

输出为 `recoil_output/rm277_x1_opx2_recoiltrainer.json`。脚本会：

- 把识别结果第 1 发映射为 Profile 的 `shot_index=0`，不额外增加虚构的起始发；
- 优先直接使用识别阶段写出的 `shot_time_ms`，并验证 `t_ms` 从 0 开始且严格递增；
- 横纵坐标使用同一个缩放倍数，默认把纵向跨度归一到 240，同时保持实际左右摆动比例；
- 自动把 `summary.json` 中按 104° FOV、2× 镜估算的最大 pitch 写入 `recorded_recoil_pitch_range_deg`；
- 默认写入 `smoothing="spline"`、`smoothing_strength=0.2`；
- 调用 Recoil Trainer 自身的平滑和“Auto Segment”实现，根据平滑后轨迹的 X 方向变化生成 `segments`；
- 使用 `C:\XiaodeDocuments\Programs\RecoilTrainer` 中正式的 `WeaponProfile` 模型做加载和往返校验。

常用自定义参数：

```powershell
.\.venv\Scripts\python.exe .\convert_to_recoiltrainer.py `
  .\recoil_output\keyframes_recoil.csv `
  --output .\recoil_output\rm277.json `
  --name "RM277 x1 OPX2" `
  --target-vertical-span 240
```

其中轨迹形状由识别数据决定；`recorded_recoil_pitch_range_deg` 决定训练场内这段纵向轨迹代表的实际俯仰角。若有游戏内准确的总后坐角度，可用 `--recorded-pitch-deg VALUE` 覆盖自动估算。调试时若不希望调用训练器自动分段，可显式使用 `--segmentation single`。

## 批量处理 Delta Force 视频并导入 Steam 版

`batch_delta_force.py` 会发现 `D:\DF` 下的全部 `.mp4`，逐一运行完整分析和转换流程。只有所有项目均通过弹匣 OCR、关键帧数、RANSAC、准星置信度、pitch、双语字段、平滑和分段校验后，`--import-steam` 才会写入 Steam 版数据目录：

```powershell
.\.venv\Scripts\python.exe .\batch_delta_force.py `
  --max-workers 2 --resume --import-steam
```

批处理规则：

- 游戏名分别写入 `Delta Force` 与 `三角洲行动`；训练器列表标题使用 `Delta Force · <weapon>`（EN）和 `三角洲·<weapon>`（CN）。中点不会在卡片字体中显示成类似“丨/I”的竖线。Workshop 标题对应为 `Delta Force · <weapon> Recoil` 和 `三角洲·<weapon> 后坐力`，武器字段本身仍只保存枪名。
- 93R、G18 使用 1×；其他枪械使用 2×。
- QBZ95-1 的重建 pitch 除以 `0.89`；其他枪械的幅度不修正。
- 所有配置使用 `spline`、平滑强度 `0.2`，并调用 Recoil Trainer 的 Auto Segment。
- `--resume` 只复用仍能通过当前流水线版本和全部质量检查的输出。
- `--convert-only` 复用已验证的 CSV/summary，只重建 JSON；适用于批量修改标题或其他 Profile 元数据。
- 导入前要求 `RecoilTrainer.exe` 未运行，并自动备份数据库和整个 profiles 目录到 `%LOCALAPPDATA%\RecoilTrainer\codex_backups`。

批量结果写入 `delta_force_batch_output`。其中：

- `batch_manifest.json`：全部视频、弹匣数、帧范围、pitch、镜倍、修正系数、质量指标与最终 JSON 路径。
- `steam_import_report.json`：本次导入的 43 个 profile ID、Steam 数据目录和可恢复备份位置。
- 每把枪的子目录：完整分析中间产物、复核图、日志和双语 Recoil Trainer JSON。

### 异常跳变审计

自动弹药事件只负责粗定位射击区间。最终 `1 -> 0` 边界由“稳定空仓帧 + 全部弹药状态联合动态规划”确定，避免中途弱变化峰漏检后，把空仓动画误补成最后几发。

CSV 始终保留 Feature Matching + RANSAC 与准星合成得到的原始 `recoil_*` 和 `shot_delta_*` 列。如果最后一个点同时满足以下条件，流程会把它识别为射击结束后的空仓/枪械下落动画，而不是新的后坐力脉冲：

- 纵向突降超过 80 px；
- 总位移超过 100 px；
- 同时超过此前逐发位移中位数的 5 倍及 `median + 8 × robust_sigma`。

此时只在 `trainer_recoil_*` 列中，用最近可靠逐发增量的稳健中位数外推末发；原始观测不覆盖。`summary.json` 和最终 JSON 的 `data.reconstruction.trajectory_corrections` 会记录原始增量、替换增量、阈值和原因。中段的 burst 射击回正（例如 M16A4 三连发之间的长间隔）不会被这一规则修改。

批量重跑会从上一份 `batch_manifest.json` 与 `steam_import_report.json` 恢复已导入的 Profile ID，因此 Steam 更新为原位覆盖，不会生成重复同名配置。
