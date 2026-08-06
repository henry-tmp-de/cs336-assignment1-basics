Phase 1：写 run_train_bpe 的最小正确版本

  目标：先求对，不求快。分 3 个子步骤，每步独立验证：

  子步骤 1：预分词 + 词频统计
  - 用 PDF 第 6 页那个 GPT-2 正则 PAT，配合 regex 库的
  re.finditer 把文本切成词
  - 每个词转成字节序列，统计每个词出现多少次
  - 数据结构建议：dict[tuple[bytes, ...], int]（词的字节元组 →
  频次），PDF 第 7 页的 stylized 例子就是这个结构
  - ✅ 验证方式：拿 PDF 第 7 页那个例子（low low low low low /
  lower lower / widest widest widest /
  newest...）跑一下，看能不能得到 {low: 5, lower: 2, widest:
  3, newest: 6} 这个频次表

  子步骤 2：BPE 合并主循环
  - 数当前所有相邻字节对的频次
  - 找频次最高的对，平局时取字典序更大的（PDF 明说用
  max()，这是对的——因为 Python 元组比较就是字典序）
  - 把这对合并成新 token，记录一条 merge
  - ✅ 验证方式：继续用 PDF 第 7 页例子，第一轮应该选
  ('s','t')（因为 ('e','s') 和 ('s','t') 都是 9
  次，取大的），第二轮是 ('e','st')。你手动对一遍 PDF 给的完整
  merge 序列：
  ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w
  i', 'wi d', 'wid est', 'low e', 'lowe r']

  子步骤 3：组装返回值
  - vocab: dict[int, bytes]：0-255 是字节，之后是特殊
  token，再之后是每次 merge 产生的新 token
  - merges: list[tuple[bytes, bytes]]：按创建顺序
  - 返回 (vocab, merges)

  子步骤 4：跑测试验收
  uv run pytest tests/test_train_bpe.py -x
  先关注 test_train_bpe（比对 corpus.en 上训出的 merges 和
  vocab 是否和标准答案一致）。

  ---
  先别碰的三件事（Phase 1 不做）

  1. 速度优化——test_train_bpe_speed（<1.5s）最后再攻，先用小语
  料跑对
  2. 特殊 token
  边界处理——先写主逻辑，test_train_bpe_special_tokens 放到
  Phase 2
  3. multiprocessing / 大文件分块——等全量训练 TinyStories
  时再用

  ---
  Phase 2（跑通基本后）：三件收尾
  3. multiprocessing / 大文件分块——等全量训练 TinyStories 时再用

  ---
  Phase 2（跑通基本后）：三件收尾

  1. 特殊 token：训练时用 re.split 按特殊 token 切段（不跨文档合并），特殊
  token 本身进词表但不参与计数 → 过 test_train_bpe_special_tokens
  2. 速度优化：增量更新 pair 计数（只更新和刚合并的对重叠的部分）→ 过
  test_train_bpe_speed
  3. 全量训练：TinyStories 上 vocab_size=10000 + <|endoftext|>，保存结果 →
  回答 PDF 2.5 的书面题

  ---
  调试小技巧

  - 写完后先用 PDF 第 7
  页的例子手动测，别直接跑测试文件——那个例子小，能一步步对
  - uv run pytest tests/test_train_bpe.py::test_train_bpe -x -v 只跑单个测试
  - 卡住了把报错或输出贴给我，我帮你分析思路（不替你写代码）

  ---
  现在就从子步骤 1 开始吧——在 adapters.py 的 run_train_bpe 里，先只做"读文件
  + 预分词 + 词频统计"，打印出来验证。写好了或者卡住了告诉我。
