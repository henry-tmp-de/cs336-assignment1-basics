# train_tinystories.py —— TinyStories 全量字节级 BPE 训练（vocab_size=10000）
# 用法: python train_tinystories.py [数据文件路径]   （省略则用 2.1GB 全量）
# 输出: tinystories_vocab.json + tinystories_merges.txt + 分段计时
import json
import os
import sys
import time
from pathlib import Path

from tests.adapters import run_train_bpe
from tests.common import gpt2_bytes_to_unicode

DEFAULT_DATA = Path("data/TinyStoriesV2-GPT4-train.txt")
OUT_VOCAB = Path("tinystories_vocab.json")
OUT_MERGES = Path("tinystories_merges.txt")

_byte2char = gpt2_bytes_to_unicode()   # 字节 → 可打印字符（GPT-2 风格，无损映射）


def tok_str(b: bytes) -> str:
    """把一个 bytes token 转成可打印字符串（与参考实现同格式）"""
    return "".join(_byte2char[x] for x in b)


if __name__ == "__main__":
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    size_mb = data_path.stat().st_size / 1e6
    print(f"数据: {data_path}（{size_mb:.0f} MB）")
    print(f"num_workers = {os.cpu_count()}，vocab_size = 10000，special=<|endoftext|>\n")

    timings = {}
    t_total = time.time()
    vocab, merges = run_train_bpe(
        str(data_path),
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        num_workers=os.cpu_count(),
        timings=timings,
        verbose=1000,
    )
    timings["total"] = time.time() - t_total

    # 存盘 1: vocab → JSON（键=token字符串，值=id）
    vocab_str = {tok_str(v): k for k, v in vocab.items()}
    with open(OUT_VOCAB, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False)

    # 存盘 2: merges → txt（每行"token1 token2"，按合并顺序）
    with open(OUT_MERGES, "w", encoding="utf-8") as f:
        for t1, t2 in merges:
            f.write(f"{tok_str(t1)} {tok_str(t2)}\n")

    # 最长 token
    longest = max(vocab.values(), key=len)
    longest_txt = tok_str(longest)

    print("===== 计时 =====")
    for k, v in timings.items():
        print(f"  {k:12s}: {v:7.1f}s")
    print(f"\n词表大小: {len(vocab)}  （含 <|endoftext|> + 256 字节）")
    print(f"合并次数: {len(merges)}")
    print(f"最长 token: {longest_txt!r}  （{len(longest)} 字节）")
    print(f"\n已保存: {OUT_VOCAB}（{OUT_VOCAB.stat().st_size/1e6:.1f} MB）")
    print(f"已保存: {OUT_MERGES}（{OUT_MERGES.stat().st_size/1e6:.1f} MB）")
