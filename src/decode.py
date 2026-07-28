"""
CTC 贪心解码 + 后处理去重。
"""
import torch


def ctc_greedy_decode(log_probs, blank=0):
    """
    log_probs: (T, vocab_size) 或 (T, B, vocab_size)
    返回: (B, seq_len) token id 列表（已去重去 blank）
    """
    if log_probs.dim() == 2:
        log_probs = log_probs.unsqueeze(1)  # (T, 1, vocab_size)

    T, B, V = log_probs.shape
    best = log_probs.argmax(dim=-1)  # (T, B)

    results = []
    for b in range(B):
        path = best[:, b]  # (T,)
        # 合并连续相同 token
        path = torch.unique_consecutive(path)
        # 去 blank
        path = path[path != blank]
        results.append(path)

    return results


def ctc_decode_text(log_probs, idx2word, blank=0):
    """解码为文本列表，每个元素是空格分隔的中文字符串。"""
    token_seqs = ctc_greedy_decode(log_probs, blank)
    texts = []
    for tokens in token_seqs:
        words = [idx2word[int(t)] for t in tokens if int(t) < len(idx2word)]
        texts.append(" ".join(words))
    return texts
