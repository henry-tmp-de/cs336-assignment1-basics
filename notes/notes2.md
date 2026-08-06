# Assignment 1 进度笔记 2 — run_train_bpe 完成 & 待办清单

## 一、当前状态（2026-08-06）

- `run_train_bpe` 三个测试**全绿**：
  - `test_train_bpe` PASSED —— merges 与标准答案逐字节一致
  - `test_train_bpe_special_tokens` PASSED —— 特殊 token 切分完成（`<|endoftext|>` 永不参与合并）
  - `test_train_bpe_speed` PASSED —— <1.5s 达标
- 当前 merge 循环用**检查版**：`if best_pair in zip(word, word[1:])` 跳过无关词，每轮仍扫全部词。
- 特殊 token 切分 = 按 `<|endoftext|>` `re.split` 切段 → 每段单独预分词 → 频次跨段累加进同一个 chardict。

## 二、还没做 / 可以继续优化的清单

### A. Part 1 剩余主体：get_tokenizer（encode / decode / encode_iterable）
- `tests/test_tokenizer.py` 25 个测试当前全红，这是 Part 1 剩下最大的活。
- 路线图见本文档第四节。

### B. run_train_bpe 进一步优化（当前已达标，可选）
1. **【已记录，未采用】索引版 merge 循环（pair→words）**——完整代码见第三节。
   - 为什么更快：每轮只处理含 best_pair 的词，不扫全部词。实测 0.35s vs 检查版 1.37s vs 原版 1.81s。
   - **什么时候必须做**：TinyStories 全量训练（2.1GB）。检查版会跑 10-30 分钟，索引版才能进 2 分钟（题目要求）。
2. **multiprocessing 预分词**：2.1GB 按文档边界（`<|endoftext|>`）分块，多进程并行 `re.findall`，再合并词频字典。Ryzen AI 9 用逻辑核数（365 是 20 线程 / HX370 是 24 线程）。
3. **读文件指定 `encoding="utf-8"`**：Windows 下 `open()` 默认 GBK，TinyStories 含非 ASCII 字符会崩。

### C. TinyStories 全量训练（PDF 2 分书面题）
- `vocab_size=10000` + `<|endoftext|>`，读 `data/TinyStoriesV2-GPT4-train.txt`。
- 保存：vocab → JSON（键=token 字符串、值=id），merges → txt（每行空格分隔的两个 token）。
- 分段计时（读文件/预分词/merge/保存）→ 回答 (b) 哪部分最耗时。
- 回答 (a)：总耗时 / 最长 token 是什么 / 是否合理（看它词频是否够高）。
- 前提：先做 B1 索引版，否则跑不完。

### D. Part 2-4（后续大块）
- Part 2 Transformer（attention / MLP / LayerNorm / 残差）
- Part 3 Loss（交叉熵）+ Optimizer（AdamW）
- Part 4 训练循环 + 数据加载

### E. 环境注意事项
- pytest 一律用 `uv run python -m pytest`（`pytest.exe` 被 Windows 应用控制策略拦截，os error 4551）。
- 运行前先 `cd D:\学习\cs336-assignment1\assignment1-basics`。
- speed 测试对机器负载敏感（空闲 1.0s，负载大 2.7s），跑之前关掉重型应用。

## 三、索引版 merge 循环（完整记录，当前未采用）

思路：维护一张 `pair_words` 索引（pair → 含它的词），每轮只处理 `pair_words[best_pair]` 里的词，`bytesdict` 原地更新。这是 TinyStories 2 分钟目标的关键，参考实现就是这么干的。

```python
# ① 初始化（替代原 pair_counts 循环）
pair_counts = {}
pair_words = {}                                  # 新：pair → {word: freq}
for word, freq in bytesdict.items():
    for i in range(len(word) - 1):
        p = (word[i], word[i + 1])
        pair_counts[p] = pair_counts.get(p, 0) + freq
        pair_words.setdefault(p, {})[word] = freq   # 登记 word 含 pair p

# ② merge 主循环（替代原 while 体）
while len(vocab) < vocab_size:
    best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
    newtoken = best_pair[0] + best_pair[1]
    merge_list.append(best_pair)

    affected = list(pair_words[best_pair].items())   # 只拿含 best_pair 的词
    for word, freq in affected:
        del bytesdict[word]                          # 旧词移出
        for i in range(len(word) - 1):               # 撤旧：旧 pair 减贡献
            p = (word[i], word[i + 1])
            pair_counts[p] = pair_counts.get(p, 0) - freq
            pw = pair_words.get(p)
            if pw is not None:
                pw.pop(word, None)
                if not pw:
                    del pair_words[p]

        merged_word = rebuild(word, best_pair, newtoken)   # 合并
        bytesdict[merged_word] = bytesdict.get(merged_word, 0) + freq  # 新词（累积）

        for i in range(len(merged_word) - 1):        # 加新：新 pair 登记
            p = (merged_word[i], merged_word[i + 1])
            pair_counts[p] = pair_counts.get(p, 0) + freq
            pw2 = pair_words.setdefault(p, {})
            pw2[merged_word] = pw2.get(merged_word, 0) + freq

    vocab[len(vocab)] = newtoken
```

关键点（回忆用）：
- `pair_words[best_pair]` 直接给出受影响的词，不用扫全库。
- 原地改 `bytesdict`：删旧词（`del`）→ 加新词（`get(...,0)+freq` 累积，因为多个旧词可能并出同一新词）。
- `setdefault(p, {})` = 取 p 的内层表，没有就建。
- `pop(word, None)` 不报错地移除；`if not pw` 检测空表（空字典是假值）。
- `affected` 必须 `list()` 复制，因为要边迭代边删原索引。
- 正确性：处理完 `pair_words[best_pair]` 必清空、`pair_counts[best_pair]` 归 0，max 不再选它。
- 用 `test_train_bpe` 验证（merges 必须和检查版完全一致）再上 TinyStories。

## 四、get_tokenizer 路线图

`get_tokenizer` 返回一个带三个方法的类实例。整体管道：

```
text
  → 按特殊 token 切段（最长优先、转义、re.split）
  → 每段：utf-8 编码 → bytes_to_unicode 重映射成字符串 → GPT-2 正则切词
  → 每词：还原成单字节 token → 按 merges 顺序用 rebuild 合并 → 查 vocab 逆表得 id
decode: id → vocab[id] bytes → 拼接 → utf-8 解码 (errors="replace")
```

实现顺序：
1. 类骨架 + `__init__`（建 vocab 逆表 bytes→id、存 merges、存特殊 token bytes）
2. `decode`（最简单，先写）
3. `encode` 普通路径（不含特殊 token）：重映射 → 正则 → merge → 查 id
4. `encode` 加特殊 token 切分
5. `encode_iterable`
