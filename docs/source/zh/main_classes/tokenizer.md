<!--版权所有2020年The HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）使用此文件除非符合许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本：

http$：//www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，无论是明示的还是暗示的。请参阅许可证以获得特定语言下的权限和限制。

⚠的注意事项：此文件虽然使用Markdown，但包含我们的文档生成器的特定语法（类似于MDX），可能无法正确呈现在您的Markdown查看器中。

-->

# 分词器

分词器负责为模型准备输入。该库包含所有模型的分词器。大多数分词器有两种版本：一种是完整的Python实现，另一种是基于Rust库的“快速”实现 [🤗 Tokenizers](https$：//github.com/huggingface/tokenizers)。 “快速”实现允许

1. 在批量分词时显著加速
2. 提供附加方法，以在原始字符串（字符和单词）与标记空间之间进行映射（例如，获取包含给定字符的标记的索引或与给定标记对应的字符范围）。 

[`PreTrainedTokenizer`]和[`PreTrainedTokenizerFast`]实现了在模型输入中对字符串输入进行编码的常用方法（请参见下文），以及从本地文件或目录或从库提供的预训练分词器实例化/保存Python和“快速”分词器的方法（从HuggingFace的AWS S3存储库下载）。它们都依赖于[`~tokenization_utils_base.PreTrainedTokenizerBase`]，其中包含常用方法，以及[`~tokenization_utils_base.SpecialTokensMixin`]。

因此，[`PreTrainedTokenizer`]和[`PreTrainedTokenizerFast`]实现了使用所有分词器的主要方法：

- 分词（将字符串拆分为子词标记字符串），将标记字符串转换为id并进行反向转换，进行编码/解码（即进行分词并转换为整数）。
- 以与底层结构（BPE，SentencePiece等）无关的方式添加新标记到词汇表中。
- 管理特殊标记（如掩码，句子开头等）：将它们添加到分词器的属性中以便轻松访问，并确保它们在分词过程中不会被分割。

[`BatchEncoding`]保存了[`~tokenization_utils_base.PreTrainedTokenizerBase`]的编码方法（`__call__`，`encode_plus`和`batch_encode_plus`）的输出，它是从Python字典派生的。当分词器是纯Python分词器时，此类的行为与标准Python字典完全相同，并保存这些方法计算的各种模型输入（`input_ids`，`attention_mask`等）。当分词器是“快速”分词器（即由HuggingFace的[tokenizers库](https$：//github.com/huggingface/tokenizers)支持）时，此类还提供了几种高级对齐方法，可以用于在原始字符串（字符和单词）与标记空间之间进行映射（例如，获取包含给定字符的标记的索引或与给定标记对应的字符范围）。


## 预训练分词器

[[autodoc]] 预训练分词器
    - __call__
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## 快速预训练分词器

[`PreTrainedTokenizerFast`]依赖于[tokenizers](https$：//huggingface.co/docs/tokenizers)库。从[tokenizers库](https$：//github.com/huggingface/tokenizers)获取的分词器可以很简单地加载到transformers中。请参阅[从[tokenizers库](https$：//github.com/huggingface/tokenizers)使用分词器](../fast_tokenizers)页面以了解如何操作。

[[autodoc]] PreTrainedTokenizerFast
    - __call__
    - batch_decode
    - decode
    - encode
    - push_to_hub
    - all

## BatchEncoding

[[autodoc]] BatchEncoding
