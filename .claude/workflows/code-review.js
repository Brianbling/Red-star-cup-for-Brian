export const meta = {
  name: 'code-review',
  description: '多角度代码审查：并行分析管线、推理引擎、训练流程，然后对抗验证发现的问题并合成报告',
  phases: [
    { title: '审查', detail: '推理管线 + 训练流程 + 数据处理 三路并行' },
    { title: '验证', detail: '对抗性验证每个发现' },
    { title: '合成', detail: '生成综合审查报告' },
  ],
}

// ============================================================
// 审查维度：针对实时手语识别系统
// ============================================================
const DIMENSIONS = [
  {
    key: 'inference_pipeline',
    prompt: `审查实时推理管线代码，对照 ARCHITECTURE.md 检查实现完整性：
    - 统一预处理是否真的只做一次（MediaPipe 和 YOLO 共享）
    - MediaPipe Hands 置信度清洗（visibility<0.6→坐标置零）是否正确实现
    - 坐标归一化是否以手腕间距为基准
    - 滑动窗口（3秒/90帧）的 enqueue/dequeue 逻辑是否正确
    - 融合仲裁层状态机（动态/静默/缓冲）三态转换是否完整
    - YOLO 降频运行（每5帧）是否实现
    - 三重 AND 门控条件（静默+N帧确认+置信度）是否全部检查
    - CTC 贪心解码+后处理去重是否有重复字符合并和blank过滤
    - 手部丢失清缓存（连续M帧→清窗口+重置隐状态）是否实现
    列出所有发现，标注严重程度 high/medium/low`,
  },
  {
    key: 'training',
    prompt: `审查训练管线代码，重点分析：
    - 关键点预处理脚本是否正确调用 MediaPipe Hands 逐帧提取
    - .npy 保存格式和数据加载是否匹配
    - 词汇表构建是否从 CSV Gloss 列按 / 分割去重
    - 模型架构：1D Conv(42→256) + BiLSTM(256→512, 2层) + Linear(512→vocab+1) 是否和架构设计一致
    - CTC Loss 配置是否正确（blank 索引位置、input_length 计算）
    - batch_size=1 + collate_fn 按序列长度排序是否正确
    - 早停逻辑和 checkpoint 保存是否完善
    - 是否复用了 TFNet 的 BiLSTM.py、Train.py 框架
    列出所有发现，标注严重程度 high/medium/low`,
  },
  {
    key: 'data',
    prompt: `审查数据处理代码，重点分析：
    - CE-CSL 数据路径是否使用 Windows 路径（非 Linux 硬编码）
    - 标签解析是否正确处理 Gloss 格式（斜杠分隔、数字、标点）
    - train-01418 孤立行是否已处理
    - 视频帧率和 MediaPipe 处理帧率是否匹配
    - 训练/验证/测试数据划分是否正确
    - 数据增强（如有）是否合理
    - 关键点序列长度分布是否合理过滤异常短/长的视频
    列出所有发现，标注严重程度 high/medium/low`,
  },
]

const FINDINGS_SCHEMA = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          title: { type: 'string', maxLength: 80 },
          summary: { type: 'string' },
          severity: { enum: ['high', 'medium', 'low'] },
          isPositive: { type: 'boolean' },
          file: { type: 'string' },
          suggestion: { type: 'string' },
        },
        required: ['title', 'summary', 'severity', 'isPositive'],
      },
    },
  },
  required: ['findings'],
}

// ============================================================
// Phase 1: 三路并行审查
// ============================================================
phase('审查')
const reviews = await pipeline(
  DIMENSIONS,
  d => agent(d.prompt, { label: `审查:${d.key}`, phase: '审查', schema: FINDINGS_SCHEMA }),
)

const allFindings = reviews.filter(Boolean).flatMap(r => r.findings)

// ============================================================
// Phase 2: 对抗验证（只验证 high + medium）
// ============================================================
phase('验证')
const toVerify = allFindings.filter(f => f.severity !== 'low')

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    isReal: { type: 'boolean' },
    isDuplicate: { type: 'boolean' },
    adjustedSeverity: { enum: ['high', 'medium', 'low'] },
    counterEvidence: { type: 'string' },
  },
  required: ['isReal'],
}

const verified = await pipeline(
  toVerify,
  f => agent(
    `对抗验证以下发现。你是质疑者——默认立场是推翻它。只有当代码确实支持时才能 confirm。
    对照 ARCHITECTURE.md 中的设计规范判断实现是否正确。

发现: ${f.title}
严重程度: ${f.severity}
描述: ${f.summary}
文件: ${f.file || '未指定'}
建议: ${f.suggestion || '无'}

请读取相关代码后给出判断。`,
    { label: `验证:${f.title.slice(0, 30)}`, phase: '验证', schema: VERDICT_SCHEMA }
  ).then(v => v ? { ...f, verdict: v } : null),
)

const confirmed = verified
  .filter(Boolean)
  .filter(f => f.verdict.isReal && !f.verdict.isDuplicate)
  .map(f => ({ ...f, severity: f.verdict.adjustedSeverity || f.severity }))

// ============================================================
// Phase 3: 合成报告
// ============================================================
phase('合成')
const report = await agent(
  `基于以下经过验证的发现，生成一份项目代码审查报告。

## 确认的问题 (${confirmed.filter(f => !f.isPositive).length} 个)
${confirmed.filter(f => !f.isPositive).map(f => `- [${f.severity}] ${f.title} — ${f.summary} (${f.file || 'N/A'})`).join('\n') || '无'}

## 确认的优点 (${confirmed.filter(f => f.isPositive).length} 个)
${confirmed.filter(f => f.isPositive).map(f => `- ${f.title} — ${f.summary} (${f.file || 'N/A'})`).join('\n') || '无'}

## 被推翻的发现 (${toVerify.length - confirmed.length} 个)
${toVerify.filter(f => !confirmed.includes(f)).map(f => `- ${f.title}`).join('\n') || '无'}

## 未验证 (${allFindings.filter(f => f.severity === 'low').length} 个 low)
${allFindings.filter(f => f.severity === 'low').map(f => `- ${f.title}`).join('\n') || '无'}

报告结构：
1. 总体评估（一句话）
2. 高风险问题（按优先级，对照 ARCHITECTURE.md 规范）
3. 中风险问题
4. 架构亮点
5. 改进路线图（分为"必须修/建议修/v2再说"三级）`,
  { label: '合成报告', phase: '合成' }
)

return {
  report,
  stats: {
    totalFindings: allFindings.length,
    verified: verified.filter(Boolean).length,
    confirmed: confirmed.length,
    falseAlarms: toVerify.length - confirmed.length,
  },
}
