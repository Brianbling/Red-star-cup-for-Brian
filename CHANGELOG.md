# 工作日志

## 2026-07-28

### Phase 0 — 环境搭建 [完成]
- 创建目录 `checkpoints/`、`CE-CSL/CE-CSL/keypoints/{train,dev,test}/`
- 验证依赖：torch 2.7.1+cu128 (GPU)、mediapipe 0.10.35、opencv 5.0.0、numpy 2.2.6、ultralytics 8.4.105
- 生成 `requirements.txt`
- 更新 CLAUDE.md（ultralytics 版本确认、编程规范新增日志更新规则）

### Phase 1 — 关键点预处理 [进行中]
- **重要发现**：MediaPipe 0.10.35 API 完全不同——旧版 `mp.solutions.hands` 已移除，改为 `mp.tasks.vision.HandLandmarker`
  - 需单独下载 `.task` 模型文件（`models/hand_landmarker.task`，约 7.6MB）
  - `visibility` → `presence`，`detect()` → `detect_for_video(image, timestamp_ms)`
  - 时间戳单调递增约束：每个视频需创建新的 landmarker 实例
  - 遥测上报失败（被墙）不影响功能
- 创建 `preprocess_keypoints.py`（~220 行），支持断点续跑
- 小批量验证通过（10 个视频），输出质量：非零帧率 99%+，左右手关键点均有数据
- 输出格式：(T, 84) float32，存至 `CE-CSL/CE-CSL/keypoints/{split}/{video_id}.npy`
- 待全量跑
- 清理冗余：删除 `mediapipe-master/` (84MB, pip已装) 和 `hagrid-master/` (13MB, 仅参考)
- git init + .gitignore + 首次 commit（排除视频/模型权重/npy/zip，纳管代码和文档）
