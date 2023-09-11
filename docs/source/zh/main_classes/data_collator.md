<!--版权 2020 HuggingFace 团队。保留所有权利。

根据 Apache 许可证第 2.0 版（“许可证”）获得许可；除非符合许可证，否则不得使用此文件。您可以在以下地址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言的权限和限制。

⚠️ 请注意，此文件是 Markdown 格式，但包含了我们的文档生成器（类似于 MDX）的特定语法，可能在您的 Markdown 查看器中无法正确呈现。

-->

# 数据整理器

数据整理器是使用数据集元素列表作为输入来形成批次的对象。这些元素与 `train_dataset` 或 `eval_dataset` 的元素类型相同。

为了能够构建批次，数据整理器可能会应用一些处理（如填充）。其中一些整理器（如 [`DataCollatorForLanguageModeling`]）还会对形成的批次应用一些随机数据增强（如随机掩码）。

可以在[示例脚本](../examples)或[示例笔记本](../notebooks)中找到使用示例。


## 默认数据整理器

[[autodoc]] data.data_collator.default_data_collator

## 默认数据整理器

[[autodoc]] data.data_collator.DefaultDataCollator

## 带填充的数据整理器

[[autodoc]] data.data_collator.DataCollatorWithPadding

## 用于标记分类的数据整理器

[[autodoc]] data.data_collator.DataCollatorForTokenClassification

## 用于 Seq2Seq 的数据整理器

[[autodoc]] data.data_collator.DataCollatorForSeq2Seq

## 用于语言模型的数据整理器

[[autodoc]] data.data_collator.DataCollatorForLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## 用于全词屏蔽的数据整理器

[[autodoc]] data.data_collator.DataCollatorForWholeWordMask
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens

## 用于置换语言建模的数据整理器

[[autodoc]] data.data_collator.DataCollatorForPermutationLanguageModeling
    - numpy_mask_tokens
    - tf_mask_tokens
    - torch_mask_tokens
