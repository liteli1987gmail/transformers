<!--版权所有2020 HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证，否则不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件按“原样”分发，无论是明示还是暗示。有关特定语言的权限和限制，请参阅许可证。

⚠️请注意，此文件是Markdown格式的，但包含我们的文档生成器（类似于MDX）的特定语法，可能无法在Markdown查看器中正确呈现。

-->

# 管道

管道是使用模型进行推理的一种很好且简便的方法。这些管道是抽象出库中大部分复杂代码的对象，提供了一个专门用于多个任务的简单API，包括命名实体识别、掩码语言建模、情感分析、特征提取和问答。有关使用示例，请参阅[任务摘要](../task_summary)。

有两种管道抽象类别需要注意：

- [`pipeline`] 是封装了所有其他管道的最强大的对象。
- 针对[音频](#audio)、[计算机视觉](#computer-vision)、[自然语言处理](#natural-language-processing)和[多模式](#multimodal)任务提供了特定任务的管道。

## 管道抽象

*管道*抽象是围绕所有其他可用管道的封装器。它像任何其他管道一样实例化，但可以提供额外的生活质量。

对一个项目进行简单调用:

```python
>>> pipe = pipeline("text-classification")
>>> pipe("这家餐厅很棒")
[{'label': 'POSITIVE', 'score': 0.9998743534088135}]
```

如果您想要使用[hub](https://huggingface.co)上的特定模型，则可以忽略任务，如果hub上的模型已经定义了它:

```python
>>> pipe = pipeline(model="roberta-large-mnli")
>>> pipe("这家餐厅很棒")
[{'label': 'NEUTRAL', 'score': 0.7313136458396912}]
```

要对多个项目调用管道，可以使用*list*进行调用。

```python
>>> pipe = pipeline("text-classification")
>>> pipe(["这家餐厅很棒", "这家餐厅很糟糕"])
[{'label': 'POSITIVE', 'score': 0.9998743534088135},
 {'label': 'NEGATIVE', 'score': 0.9996669292449951}]
```

建议使用`dataset`直接迭代整个数据集。这意味着您不需要一次性分配整个数据集，也不需要自行批处理。这应该与在GPU上的自定义循环一样快。如果不是，请创建一个问题。

```python
import datasets
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

pipe = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h", device=0)
dataset = datasets.load_dataset("superb", name="asr", split="test")

# KeyDataset（仅 *pt*）将简单地返回数据集项返回的字典中的项目，因为我们对数据集的*target*部分不感兴趣。对于句子对，请使用KeyPairDataset
for out in tqdm(pipe(KeyDataset(dataset, "file"))):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

为了方便起见，也可以使用生成器:


```python
from transformers import pipeline

pipe = pipeline("text-classification")


def data():
    while True:
        # 这可以来自数据集、数据库、队列或HTTP请求在服务器上
        # 注意因为这是迭代的，所以无法使用`num_workers > 1`变量
        # 使用多个线程对数据进行预处理。您仍然可以有1个线程进行
        # 预处理，而主线程运行大型推理
        yield "这是一个测试"


for out in pipe(data()):
    print(out)
    # {"text": "NUMBER TEN FRESH NELLY IS WAITING ON YOU GOOD NIGHT HUSBAND"}
    # {"text": ....}
    # ....
```

[[autodoc]] 管道

## 管道批处理

所有管道都可以使用批处理。这将在管道使用其流式处理能力时起作用（因此当传递列表、`Dataset`或`generator`时）。

```python
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

dataset = datasets.load_dataset("imdb", name="plain_text", split="unsupervised")
pipe = pipeline("text-classification", device=0)
for out in pipe(KeyDataset(dataset, "text"), batch_size=8, truncation="only_first"):
    print(out)
    # [{'label': 'POSITIVE', 'score': 0.9998743534088135}]
    # 与之前完全相同的输出，但是内容作为批次传递给模型
```

<Tip warning={true}>

但是，这对性能来说不是自动获胜。它可能是10倍的加速或5倍的放慢，这取决于硬件、数据和实际使用的模型。

主要是速度更快的示例:

</Tip>

```python
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm

pipe = pipeline("text-classification", device=0)


class MyDataset(Dataset):
    def __len__(self):
        return 5000

    def __getitem__(self, i):
        return "这是一个测试"


dataset = MyDataset()

for batch_size in [1, 8, 64, 256]:
    print("-" * 30)
    print(f"流式处理 batch_size={batch_size}")
    for out in tqdm(pipe(dataset, batch_size=batch_size), total=len(dataset)):
        pass
```

```
# On GTX 970
------------------------------
流式处理无批处理
100%|██████████████████████████████████████████████████████████████████████████| 5000/5000 [00:26<00:00, 187.52it/s]
------------------------------
流式处理 batch_size=8
100%|████████████████████████████████████████████████████████████████████| 5000/5000 [00:04<00:00, 1205.95it/s]
------------------------------
流式处理 batch_size=64
100%|████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2478.24it/s]
------------------------------
流式处理 batch_size=256
100%|█████████████████████████████████████████████████████████████████████| 5000/5000 [00:01<00:00, 2554.43it/s]
（收益递减，GPU饱和）
```

大数据集上没有好（通用）的解决方案，因此您的使用情况可能会有所不同。准则是：

对于用户来说，经验准则是:

- **使用您的负载、您的硬件测量性能。测量、测量，并持续测量。真实的数字是唯一的方式。**
- 如果您受到延迟限制（进行实时推理的实时产品），请不要批处理
- 如果使用的是CPU，请不要批处理。
- 如果使用的是GPU，且使用吞吐量（希望在一堆静态数据上运行模型），则:

  - 如果您对sequence_length的大小没有概念（"自然"数据），则默认不要批处理，测量并
    尝试暂时添加它，添加OOM检查以在失败时恢复（如果您不
    控制sequence_length，它在某些时候将失败）
  - 如果您的sequence_length非常规则，则批处理更有可能非常有趣，测量并推
    它，直到您遇到OOM。
  - GPU越大，批处理越有可能更有趣
- 一旦启用批处理，请确保可以很好地处理OOM。

## 管道块批处理

`zero-shot-classification`和`question-answering`在某种程度上是特殊的，因为单个输入可能会产生模型的多次前向传递。在正常情况下，这会导致`batch_size`参数出现问题。

为了避免这个问题，这两个管道有点特殊，它们是`ChunkPipeline`而不是常规的`Pipeline`。简而言之:


```python
preprocessed = pipe.preprocess(inputs)
model_outputs = pipe.forward(preprocessed)
outputs = pipe.postprocess(model_outputs)
```

现在变成了:


```python
all_model_outputs = []
for preprocessed in pipe.preprocess(inputs):
    model_outputs = pipe.forward(preprocessed)
    all_model_outputs.append(model_outputs)
outputs = pipe.postprocess(all_model_outputs)
```

这对于您的代码来说应该是非常透明的，因为管道的使用方式是一样的。

这只是一个简化的视图，因为管道可以自动处理批处理！这意味着您无需关心输入实际上会触发多少次前向传递，您可以独立优化`batch_size`而不影响输入。前一节的注意事项仍然适用。

## 管道自定义代码

如果您想要覆盖特定管道，请随时为您的任务创建问题，管道的目标是易于使用并支持大多数情况，因此`transformers`可能可以支持您的用例。


如果要简单尝试，可以:

- 子类化您选择的管道

```python
class MyPipeline(TextClassificationPipeline):
    def postprocess():
        # 在这里添加您的代码
        scores = scores * 100
        # 还可以在这里添加代码


my_pipeline = MyPipeline(model=model, tokenizer=tokenizer, ...)
# 或者如果使用*pipeline*函数，那么:
my_pipeline = pipeline(model="xxxx", pipeline_class=MyPipeline)
```

这样应该使您能够编写所有自定义代码。


## 实现管道

[实现新管道](../add_new_pipeline)

## 音频

音频任务的可用管道包括以下内容。

### AudioClassificationPipeline

[[autodoc]] AudioClassificationPipeline
    - __call__
    - all

### AutomaticSpeechRecognitionPipeline

[[autodoc]] AutomaticSpeechRecognitionPipeline
    - __call__
    - all

### ZeroShotAudioClassificationPipeline

[[autodoc]] ZeroShotAudioClassificationPipeline
    - __call__
    - all

## 计算机视觉

计算机视觉任务的可用管道包括以下内容。

### DepthEstimationPipeline
[[autodoc]] DepthEstimationPipeline
    - __call__
    - all

### ImageClassificationPipeline

[[autodoc]] ImageClassificationPipeline
    - __call__
    - all

### ImageSegmentationPipeline

[[autodoc]] ImageSegmentationPipeline
    - __call__
    - all

### ObjectDetectionPipeline

[[autodoc]] ObjectDetectionPipeline
    - __call__
    - all

### VideoClassificationPipeline

[[autodoc]] VideoClassificationPipeline
    - __call__
    - all

### ZeroShotImageClassificationPipeline

[[autodoc]] ZeroShotImageClassificationPipeline
    - __call__
    - all

### ZeroShotObjectDetectionPipeline

[[autodoc]] ZeroShotObjectDetectionPipeline
    - __call__
    - all

## 自然语言处理

自然语言处理任务的可用管道包括以下内容。

### ConversationalPipeline

[[autodoc]] Conversation

[[autodoc]] ConversationalPipeline
    - __call__
    - all

### FillMaskPipeline

[[autodoc]] FillMaskPipeline
    - __call__
    - all

### NerPipeline

[[autodoc]] NerPipeline

有关所有详细信息，请参阅[`TokenClassificationPipeline`]。

### QuestionAnsweringPipeline

[[autodoc]] QuestionAnsweringPipeline
    - __call__
    - all

### SummarizationPipeline

[[autodoc]] SummarizationPipeline
    - __call__
    - all

### TableQuestionAnsweringPipeline

[[autodoc]] TableQuestionAnsweringPipeline
    - __call__

### TextClassificationPipeline

[[autodoc]] TextClassificationPipeline
    - __call__
    - all

### TextGenerationPipeline

[[autodoc]] TextGenerationPipeline
    - __call__
    - all

### Text2TextGenerationPipeline

[[autodoc]] Text2TextGenerationPipeline
    - __call__
    - all

### TokenClassificationPipeline

[[autodoc]] TokenClassificationPipeline
    - __call__
    - all

### TranslationPipeline

[[autodoc]] TranslationPipeline
    - __call__
    - all

### ZeroShotClassificationPipeline

[[autodoc]] ZeroShotClassificationPipeline
    - __call__
    - all

## 多模态

多模态任务的可用管道包括以下内容。

### DocumentQuestionAnsweringPipeline

[[autodoc]] DocumentQuestionAnsweringPipeline
    - __call__
    - all

### FeatureExtractionPipeline

[[autodoc]] FeatureExtractionPipeline
    - __call__
    - all

### ImageToTextPipeline

[[autodoc]] ImageToTextPipeline
    - __call__
    - all

### VisualQuestionAnsweringPipeline

[[autodoc]] VisualQuestionAnsweringPipeline
    - __call__
    - all

## 父类：`Pipeline`

[[autodoc]] Pipeline
