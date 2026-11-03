# QuantLightningIndexerV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Ascend 950PR&950DT 系列产品</term>|      √     |
|<term>Atlas A3 系列产品</term>|      √     |
|<term>Atlas A2 系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：QuantLightningIndexerV2是推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

- 计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为Attention的输入。

## A2/A3 候选消费者快速路径

`candidate_mode=4` 接收与 mode=2 相同的候选块输入，紧凑收集候选分数后排序。
候选重复或有效位置不足 TopK 时，逐行使用原有 mask 路径；因果尾块只拷贝可见位置。
局部及最终归并按请求 TopK 的对齐长度处理，保留原 workspace 布局。

mode=4 支持对多个请求沿 K 维分配任务，再归并各分片的局部 TopK。
各请求的 query 分别按 4 行分组，总组数为 `sum(ceil(query_count[b] / 4))`。
请求数和总组数均小于 AIC 核数、单 KV head，且每个有 query 的请求压缩后 K 长度
至少为 4096 时启用。多请求还要求总组数不超过 AIC 核数的一半，保证每组至少可用
两个核，避免未分核的组限制整个 batch 的耗时。每个组独立处理所属请求的页表、因果范围和输出位置；TND 支持
不同 query 长度及空请求。多请求 BSND 存在填充行时保留原调度。

这里的 batch 是当前算子调用中的请求数，与 DP 的卡数无关。例如 8 个请求各有
1 个 query 时共 8 组，在 24 个 AIC 上每组可分配 3 个核；16 个请求各有 6 个
query 时共 32 组，继续使用原 metadata 调度。mode=1 的候选生成仍使用原路径。

DeepSeek-V4.1 可通过 `VLLM_ASCEND_QLI_CANDIDATE_SKIP=1` 将候选消费者切换到
mode=4；默认值为 `0`，保持 mode=2。该开关复用已有候选层配置，不改变候选数或块大小。
mode=4 本身仍计算所有可见 K。下面的复用路径通过提前收集候选 K 减少消费者的计算量。

## A3 跨层候选 K 复用

DeepSeek-V4.1 可通过 `VLLM_ASCEND_QLI_CANDIDATE_REUSE=1` 启用候选 K/scale
复用，默认值为 `0`。该开关同时启用 mode=4 回退，无需另设 `VLLM_ASCEND_QLI_CANDIDATE_SKIP`。
当前模型接入仅在 A3 启用；A2 未验证，A5 继续使用其 packed-cache 后端。

候选源在每次 forward 中调用 [QliCandidateGather](../qli_candidate_gather/README.md)，
按原块号升序打包各 query 可见的候选 K/scale、原始位置和有效长度。后续消费者分别使用自己的
Q 和 weights，调用 `candidate_mode=5`，再将 packed 索引恢复成原始位置，并执行原有索引整理。
mode=5 要求 `TND/PA_BBND`、`mask_mode=0`、`cmp_ratio=1`、每个 packed batch 一个 query，
块大小为 8。它保留 mode=4 的块降序、块内升序同分优先级，不需要传入候选块列表。

模型仅为同一 PP 阶段内、相同 KV owner、压缩比为 1 的候选源及至少两个 index-source
消费者分配共享缓冲。只复用 TopK 的普通层不参与。候选必须来自原生候选源、互不重复；
容量必须保证尾块只有一个可见 token 时仍足够选 TopK，短 K 则由候选源覆盖全部位置。
不满足拓扑、布局或行容量约束时，消费者使用 mode=4。

四个持久 payload 缓冲合计最多 512 MiB；候选数 2048 时最多容纳 244 行，超过时回退。
静态 packed metadata 在初始化时生成，另有少量 metadata、页表和临时排序空间。
source 每次执行都覆盖当前行，包括 graph replay；没有 Python ready 标记。
query 的 TND 前缀和必须覆盖输入行，padding 请求使用零 K 长度。

此开关面向长上下文。短上下文中打包成本可能超过减少计算的收益，因此默认关闭。
不能用 Python 当前 K 长度在 capture 时选择路径：replay 会改变设备长度而不会重跑此判断。
模型最大长度同时受 FP32 块号排序的精确表示范围约束，必须小于 `(2^24-1)*8`。
性能应比较完整的“一次准备 + 多消费者 + 索引恢复”，不得只计 packed QLI。

## 参数说明

| 参数名                     | 输入/输出/属性 | 描述  | 数据类型       | 数据格式   |
|----------------------------|-----------|----------------------------------------------------------------------|----------------|------------|
| query                     | 输入      | 公式中的$Q_{index}^{Quant}\in\R^{g\times d}$，表示输入Index Query，不支持非连续。| INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1 | ND         |
| key                   | 输入      | 公式中的$K_{index}^{Quant}\in\R^{S_{k}\times d}$，表示压缩后的输入Index Key，支持0轴非连续。| INT8、FLOAT8_e4m3fn、HIFLOAT8、FLOAT4_e2m1 | ND |
| weights                 | 输入      | 公式中的$W$，表示权重系数，不支持非连续。 | FLOAT16、FLOAT32 | ND |
| query_dequant_scale             | 输入      | 公式中的$Scale_Q$，表示Index Query的反量化系数，不支持非连续。`quant_mode`为3/5时，shape为将`query`的D轴替换为(D/64, 2)；`quant_mode`为4时，shape为(1,)；其他场景shape与weights一致 | FLOAT16、FLOAT32、FLOAT8_e8m0     | ND         |
| key_dequant_scale            | 输入      | 公式中的$Scale_K$，表示Index Key的反量化系数，支持0轴非连续。`quant_mode`为3/5时，shape为将`key`的D轴替换为(D/64, 2)；`quant_mode`为4时，shape为(1,)；其他场景shape为移除`key`的D轴 | FLOAT16、FLOAT32、FLOAT8_e8m0       | ND         |
| cu_seqlens_q                    | 可选输入      | layout_q为TND时必须传入，表示每个Batch中`query`的有效token数前缀和。；layout_q为BSND时不能传入 | INT32       | ND         |
| cu_seqlens_k                    | 可选输入      | layout_k为TND时必须传入，表示每个Batch中`key`的有效token数前缀和；layout_k为PA_BSND或BSND时不能传入 | INT32       | ND         |
| seqused_q                    | 可选输入      | layout_q为BSND时可选传入，表示每个Batch中`query`的有效token数 | INT32       | ND         |
| seqused_k                    | 可选输入      | layout_k为PA_BSND或BSND时使用，表示每个Batch中`key`的有效token数。| INT32       | ND         |
| cmp_residual_k                    | 可选输入      | 压缩场景下Key的残余长度，需满足0 \<= cmp_residual_k\[i\] \< cmp_ratio。| INT32       | ND         |
| block_table                    | 可选输入      | 表示PageAttention中KV存储使用的block映射表。 | INT32       | ND         |
| output_idx_offset                    | 可选输入      | 输出索引的偏移量 | INT32       | ND         |
| metadata                    | 可选输入      | QuantLightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容。 | INT32       | ND         |
| quant_mode                 | 属性      | 用于标识输入的量化模式。 | INT32          | -         |
| max_seqlen_q                 | 可选属性| Query的最大序列长度，默认值-1表示任意可能长度 | INT32 | -         |
| layout_q                 | 可选属性| 用于标识输入`query`的数据排布格式，默认值"BSND"。 | STRING | -         |
| layout_k      | 可选属性      | 用于标识输入`key`的数据排布格式，默认值"BSND"。| STRING          | -         |
| topk  | 属性      | 代表topK阶段需要保留的索引数量，默认值2048。 | INT32          | -         |
| mask_mode | 可选属性      | 表示mask的模式，默认值0。 | INT32          | -         |
| cmp_ratio      | 可选属性      | 用于稀疏计算，表示key的压缩倍数，默认值1。 | INT32          | -         |
| return_value      |  可选属性     | 表示是否输出`sparse_values`，默认值0。 | INT32          | -         |
| sparse_indices     | 输出      | 公式中的输出Out，参与稀疏attention计算的token索引值。 | INT32          | ND         |
| sparse_values           | 输出      | 公式中的Indices输出对应的value值。`return_value`为1时shape与`sparse_indices`一致，`return_value`为0时shape为(0,) | BFLOAT16         | ND          |

## 约束说明

- <term>Ascend 950PR&950DT 系列产品</term>：
    - `query`、`key`在`quant_mode`为1/3时支持FLOAT8_e4m3fn，`quant_mode`为2时支持INT8，`quant_mode`为4时支持HIFLOAT8，`quant_mode`为5时支持FLOAT4_e2m1。
    - `query_dequant_scale`和`key_dequant_scale`在`quant_mode`为1/4时支持FLOAT32，`quant_mode`为2时支持FLOAT16，`quant_mode`为3/5时支持FLOAT8_e8m0。
    - `weights`在`quant_mode`为2时支持FLOAT16，`quant_mode`为1/3/4/5时支持FLOAT32。
    - `quant_mode`为3/5时，`query_dequant_scale`和`key_dequant_scale`的维数分别比`query`和`key`多1，前缀维度保持一致，末两维为(D/64,2)；D必须为64的倍数，每个scale对应连续32个D轴逻辑元素。
    - `query`的N支持[1, 64]，`key`的N仅支持1。
    - `topk`支持[1, 8192]。
    - 当传入的参数layout_query为TND时，必须传入cu_seqlens_q，如果也传入seqused_q，应保证由seqused_q传入的各个batch的query长度不超过根据cu_seqlens_q计算出的各个batch的q序列长度。当某个batch由seqused_q传入的q序列长度seqlen1小于由cu_seqlens_q计算出的query长度seqlen2时，会启用TND Padding功能，将该batch的seqlen2与seqlen1差值部分的query输出的sparse_indices和sparse_values全部置为无效值。部分长序列场景下，如果需要填充的无效数据过多，由于硬件限制可能会导致aicore执行超时，可以通过(seqlen2 - seqlen1) * topk来计算需要填充的数据量，建议将这个数据量控制在4亿以内。

- <term>Atlas A3 系列产品</term>、<term>Atlas A2 系列产品</term>：
    - `quant_mode`仅支持2。
    - `query`、`key`支持INT8，不支持FLOAT8_e4m3fn、HIFLOAT8和FLOAT4_e2m1。
    - `query_dequant_scale`和`key_dequant_scale`支持FLOAT16，不支持FLOAT32和FLOAT8_e8m0。
    - `weights`支持FLOAT16，不支持FLOAT32。
    - 不支持`output_idx_offset`和`return_value`。
    - `query`的N仅支持64，`key`的N仅支持1。
    - `topk`支持[1, 2048]。

## 调用示例

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| PyTorch API | - | 通过[torch.ops.cann_ops_transformer.quant_lightning_indexer](../../torch_extension/cann_ops_transformer/docs/zh/quant_lightning_indexer.md)接口调用QuantLightningIndexerV2算子。 |
| aclnn API | [test_aclnn_quant_lightning_indexer_v2](examples/test_aclnn_quant_lightning_indexer_v2.cpp) | 通过[aclnnQuantLightningIndexerV2](docs/aclnnQuantLightningIndexerV2.md)两段式接口调用QuantLightningIndexerV2算子。 |
