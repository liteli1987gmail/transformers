<!--版权 2020 年 HuggingFace 团队。版权所有。

根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证，否则不得使用此文件。您可以在以下网址获取许可证副本

http：//www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，以“现状”方式分发的软件无任何形式的明示或暗示担保或条件。请参阅许可证以获取特定语言的权限和限制。

⚠️ 请注意，此文件是 Markdown 文件，但包含特定于我们的 doc-builder（类似于 MDX）的语法，可能在您的 Markdown 查看器中无法正确渲染。

-->

# 处理器

处理器在 Transformers 库中可以有两个不同的含义:
- 处理多模态模型（如 [Wav2Vec2](../model_doc/wav2vec2)（语音和文本）或 [CLIP](../model_doc/clip)（文本和视觉））的输入的对象
- 在旧版本的库中用于预处理 GLUE 或 SQUAD 数据的弃用对象

## 多模态处理器

任何多模态模型都需要一个编码或解码数据的对象，该对象将多个模态（文本、视觉和音频）分组。这由称为处理器的对象处理，处理器将两个或多个处理对象（如文本模态的标记器、视觉的图像处理器和音频的特征提取器）分组在一起。

这些处理器继承了实现保存和加载功能的以下基类:

[[autodoc]] ProcessorMixin

## 弃用的处理器

所有处理器都遵循相同的架构，即 [`~data.processors.utils.DataProcessor`] 的架构。处理器返回 [`~data.processors.utils.InputExample`] 的列表。这些 [`~data.processors.utils.InputExample`] 可以转换为 [`~data.processors.utils.InputFeatures`] 以供模型使用。

[[autodoc]] data.processors.utils.DataProcessor

[[autodoc]] data.processors.utils.InputExample

[[autodoc]] data.processors.utils.InputFeatures

## GLUE

[通用语言理解评估（GLUE）](https：//gluebenchmark.com/) 是一个评估模型在多样化的现有 NLU 任务上性能的基准。它与论文 [GLUE: 用于自然语言理解的多任务基准和分析平台](https：//openreview.net/pdf?id=rJ4km2R5t7) 一同发布。

该库为以下任务提供了合计 10 个处理器：MRPC、MNLI、MNLI（不匹配）、CoLA、SST2、STSB、QQP、QNLI、RTE 和 WNLI。

这些处理器是：

- [`~data.processors.utils.MrpcProcessor`]
- [`~data.processors.utils.MnliProcessor`]
- [`~data.processors.utils.MnliMismatchedProcessor`]
- [`~data.processors.utils.Sst2Processor`]
- [`~data.processors.utils.StsbProcessor`]
- [`~data.processors.utils.QqpProcessor`]
- [`~data.processors.utils.QnliProcessor`]
- [`~data.processors.utils.RteProcessor`]
- [`~data.processors.utils.WnliProcessor`]

此外，还可以使用以下方法从数据文件加载值并将其转换为 [`~data.processors.utils.InputExample`] 的列表。

[[autodoc]] data.processors.glue.glue_convert_examples_to_features


## XNLI

[跨语言 NLI 语料库（XNLI）](https：//www.nyu.edu/projects/bowman/xnli/) 是一个评估跨语言文本表示质量的基准。XNLI 是基于 [*MultiNLI*](http：//www.nyu.edu/projects/bowman/multinli/) 的众包数据集：以 15 种不同语言（包括英语等高资源语言和斯瓦希里语等低资源语言）对文本进行文本蕴涵注释。

它与论文 [XNLI: 评估跨语言句子表示](https：//arxiv.org/abs/1809.05053) 一同发布。

该库提供了加载 XNLI 数据的处理器：

- [`~data.processors.utils.XnliProcessor`]

请注意，由于测试集上可用黄金标签，因此评估将在测试集上执行。

在 [run_xnli.py](https：//github.com/huggingface/transformers/tree/main/examples/legacy/text-classification/run_xnli.py) 脚本中提供了使用这些处理器的示例。


## SQuAD

[斯坦福问答数据集（SQuAD）](https：//rajpurkar.github.io/SQuAD-explorer//) 是一个评估模型在问答任务上性能的基准。有两个版本可用，v1.1 和 v2.0。第一个版本（v1.1）与论文 [SQuAD: 用于文本机器理解的 100,000+ 个问题](https：//arxiv.org/abs/1606.05250) 一同发布。第二个版本（v2.0）与论文 [了解你不知道的内容: SQuAD 的无法回答问题](https：//arxiv.org/abs/1806.03822) 一同发布。

该库为两个版本都提供了处理器：

### 处理器

这些处理器是：

- [`~data.processors.utils.SquadV1Processor`]
- [`~data.processors.utils.SquadV2Processor`]

它们都继承自抽象类 [`~data.processors.utils.SquadProcessor`]

[[autodoc]] data.processors.squad.SquadProcessor
    - all

此外，还可以使用以下方法将 SQuAD 示例转换为 [`~data.processors.utils.SquadFeatures`]，以用作模型输入。

[[autodoc]] data.processors.squad.squad_convert_examples_to_features


这些处理器以及前面提到的方法都可以用于包含数据的文件以及 *tensorflow_datasets* 包。以下是示例。


### 示例用法

以下示例演示了使用处理器以及使用数据文件的转换方法：

```python
# 加载 V2 处理器
processor = SquadV2Processor()
examples = processor.get_dev_examples(squad_v2_data_dir)

# 加载 V1 处理器
processor = SquadV1Processor()
examples = processor.get_dev_examples(squad_v1_data_dir)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

使用 *tensorflow_datasets* 与使用数据文件一样简单：

```python
# tensorflow_datasets 仅处理 Squad V1.
tfds_examples = tfds.load("squad")
examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)

features = squad_convert_examples_to_features(
    examples=examples,
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    doc_stride=args.doc_stride,
    max_query_length=max_query_length,
    is_training=not evaluate,
)
```

在 [run_squad.py](https：//github.com/huggingface/transformers/tree/main/examples/legacy/question-answering/run_squad.py) 脚本中还提供了使用这些处理器的另一个示例。
