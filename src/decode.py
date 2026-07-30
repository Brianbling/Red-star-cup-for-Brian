"""
CTC 贪心解码 + 前缀束搜索 + 后处理去重。
"""
import torch
import numpy as np
from collections import defaultdict


def ctc_greedy_decode(log_probs, blank=0):
    """
    log_probs: (T, vocab_size) 或 (T, B, vocab_size)
    返回: list of (seq_len,) tensors — 已去重去 blank
    """
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(1)  # (T, 1, vocab_size)

    T, B, V = log_probs.shape
    best = log_probs.argmax(dim=-1)  # (T, B)

    results = []
    for b in range(B):
        path = best[:, b]  # (T,)
        path = torch.unique_consecutive(path)
        path = path[path != blank]
        results.append(path)

    return results


def ctc_prefix_beam_search(log_probs, beam_width=3, blank=0):
    """
    CTC 前缀束搜索。替代贪心 argmax，不修改模型，仅推理时生效。

    log_probs: (T, V) — 单样本每帧的 log 概率
    beam_width: 束宽
    返回: list of token ids（最佳路径）
    """
    T, V = log_probs.shape
    # beams: {prefix_tuple: [log_prob_blank, log_prob_non_blank]}
    beams = {(): [float("-inf"), 0.0]}

    for t in range(T):
        new_beams = defaultdict(lambda: [float("-inf"), float("-inf")])

        for prefix, (p_b, p_nb) in beams.items():
            for c in range(V):
                prob = log_probs[t, c].item()

                if c == blank:
                    new_p_b = np.logaddexp(
                        new_beams[prefix][0],
                        np.logaddexp(p_b + prob, p_nb + prob),
                    )
                    new_beams[prefix][0] = new_p_b
                else:
                    if len(prefix) > 0 and prefix[-1] == c:
                        # 重复 token → 新对齐路径
                        new_beams[prefix][1] = np.logaddexp(
                            new_beams[prefix][1], p_b + prob)
                        new_beams[prefix][1] = np.logaddexp(
                            new_beams[prefix][1], p_nb + prob)
                    else:
                        new_prefix = prefix + (c,)
                        new_beams[new_prefix][1] = np.logaddexp(
                            new_beams[new_prefix][1],
                            np.logaddexp(p_b + prob, p_nb + prob),
                        )

        scored = []
        for prefix, (p_b, p_nb) in new_beams.items():
            score = np.logaddexp(p_b, p_nb)
            scored.append((prefix, score, p_b, p_nb))
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {p: [pb, pnb] for p, _, pb, pnb in scored[:beam_width]}

    best = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))
    return list(best[0])


def ctc_decode_batch(log_probs, blank=0, beam_width=0):
    """
    批量解码。beam_width=0 用贪心，>0 用束搜索。

    log_probs: (T, B, V)
    返回: list of tensors
    """
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(1)

    _, B, _ = log_probs.shape
    results = []
    for b in range(B):
        lp = log_probs[:, b, :]  # (T, V)
        if beam_width > 0:
            tokens = ctc_prefix_beam_search(lp, beam_width, blank)
            results.append(torch.tensor(tokens, dtype=torch.long))
        else:
            tokens = ctc_greedy_decode(lp.unsqueeze(1), blank)[0]
            results.append(tokens)
    return results


def ctc_decode_text(log_probs, idx2word, blank=0):
    """解码为文本列表，用贪心解码。"""
    token_seqs = ctc_greedy_decode(log_probs, blank)
    texts = []
    for tokens in token_seqs:
        words = [idx2word[int(t)] for t in tokens if int(t) < len(idx2word)]
        texts.append(" ".join(words))
    return texts

