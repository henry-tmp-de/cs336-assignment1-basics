# toy_index.py —— 索引版 merge 的迷你演示（整数版，逻辑与真实代码一致）
# 真实代码里：词 = bytes 的元组（如 ('a','b')）；这里用整数元组代替，机制一模一样

words = {(1, 2, 3): 3, (1, 2): 2, (4, 5): 5}      # 模拟 bytesdict：词元组 -> 频次

def rebuild(word, best_pair, new_token):
    """把 word 里所有相邻的 best_pair 替换成 new_token"""
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == best_pair:
            out.append(new_token)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)

# ① 初始化：建倒排表 pair_words（顺带建 pair_counts）
pair_counts = {}
pair_words = {}
for word, freq in words.items():
    for i in range(len(word) - 1):
        p = (word[i], word[i + 1])
        pair_counts[p] = pair_counts.get(p, 0) + freq
        pair_words.setdefault(p, {})[word] = freq

print("初始 pair_counts:", pair_counts)
print("初始 pair_words :", pair_words)

# ② 合并一轮：best = (1, 2) → 新 token = 12
best_pair = (1, 2)
new_token = 12
affected = list(pair_words[best_pair].items())     # 只拿含 best_pair 的词
print("\n受影响的词:", affected)

for word, freq in affected:
    # 撤旧：把 word 从它所有相邻 pair 的登记里注销，pair_counts 同步减
    for i in range(len(word) - 1):
        p = (word[i], word[i + 1])
        pair_counts[p] = pair_counts.get(p, 0) - freq   # 用 get 兜底，键可能不存在
        pw = pair_words.get(p)
        if pw is not None:
            pw.pop(word, None)
            if not pw:
                del pair_words[p]

    merged_word = rebuild(word, best_pair, new_token)

    # 改 bytesdict：删旧词、加新词（可能已有同名词，累加）
    del words[word]
    words[merged_word] = words.get(merged_word, 0) + freq

    # 加新：把 merged_word 登记进它所有相邻 pair，pair_counts 同步加
    for i in range(len(merged_word) - 1):
        p = (merged_word[i], merged_word[i + 1])
        pair_counts[p] = pair_counts.get(p, 0) + freq
        pw2 = pair_words.setdefault(p, {})
        pw2[merged_word] = pw2.get(merged_word, 0) + freq

print("\n合并后 words      :", words)
print("合并后 pair_counts:", pair_counts)
print("合并后 pair_words :", pair_words)
print("\n自检 pair_counts[(1,2)]==0:", pair_counts[(1, 2)] == 0)
