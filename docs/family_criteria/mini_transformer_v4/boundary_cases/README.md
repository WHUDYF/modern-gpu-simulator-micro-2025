# Boundary Cases

本目录用于存放第一轮关键边界 case 文档。

## Role

边界 case 文档是第一轮 family 判据生长的主舞台。  
它们的职责不是直接生成最终 family，而是优先回答：

- 哪些 kernel 看起来相似
- 哪些区分点阻止它们被轻易并类
- 当前阶段应给出怎样的分级结论
- 当前是否建议共享后续验证主线

## Version-1 Required Cases

第一版至少覆盖两组：

- `gemm_tiled vs attention_score`
- `softmax_kernel vs context_mul`

## Required Output Shape

每份边界 case 文档必须包含：

- shared points
- distinguishing points
- graded conclusion：`强共享 / 弱共享 / 边界未定`
- current execution advice：
  - 以 family 划分建议为主
  - 验证组织建议为辅
- 至少两处明确证据引用
