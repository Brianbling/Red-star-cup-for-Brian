"""
公共词汇表工具函数。
build_vocab.py 和 dataset.py 共享，确保标签清洗逻辑一致。
"""


def clean_word(w: str) -> str:
    """清洗标签词：去括号内容、去尾部数字、strip

    使用 depth 计数器支持括号嵌套。
    """
    depth = 0
    result = []
    for c in w:
        if c in "(（[{【":
            depth += 1
        elif c in ")）]}】":
            if depth > 0:
                depth -= 1
        elif depth == 0:
            result.append(c)
    w = "".join(result).strip()

    if w and w[-1].isdigit() and not w[0].isdigit():
        w = w.rstrip("0123456789")

    return w.strip()
