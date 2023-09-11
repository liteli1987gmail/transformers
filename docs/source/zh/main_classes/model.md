<!--版权2020年The HuggingFace团队。版权所有。

根据Apache许可证第2版（“许可证”）许可; 除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http：//www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是按“原样”分发的，不附带任何形式的明示或暗示的保证或条件。请参阅许可证以获取特定语言的权限和限制。

⚠️ 请注意，此文件是Markdown格式，但包含我们的doc-builder的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确呈现。

-->

# 模型

基类[`PreTrainedModel`]，[`TFPreTrainedModel`]和[`FlaxPreTrainedModel`]实现了加载/保存模型的通用方法，可以从本地文件或目录加载模型，也可以从库提供的预训练模型配置加载模型（从HuggingFace的AWS
S3存储库下载）。

[`PreTrainedModel`]和[`TFPreTrainedModel`]还实现了一些在所有模型中都常见的方法，如：

- 在词汇表中添加新标记时调整输入标记嵌入
- 修剪模型的注意力头。

每个模型都定义了其他通用方法，这些方法在[`~modeling_utils.ModuleUtilsMixin`]中定义（适用于PyTorch模型），以及[`~modeling_tf_utils.TFModuleUtilsMixin`]（适用于TensorFlow模型），或者用于文本生成的[`~generation.GenerationMixin`]（适用于PyTorch模型），[`~generation.TFGenerationMixin`]（适用于TensorFlow模型）和[`~generation.FlaxGenerationMixin`]（适用于Flax/JAX模型）。


## PreTrainedModel

[[autodoc]] PreTrainedModel
    - push_to_hub
    - all

<a id='from_pretrained-torch-dtype'></a>

### 大模型加载

在Transformers 4.20.0中，[`~PreTrainedModel.from_pretrained`]方法已经重新设计以适应使用[Accelerate](https：//huggingface.co/docs/accelerate/big_modeling)的大模型。这需要Accelerate> = 0.9.0和PyTorch> = 1.9.0。不再创建完整模型，然后在其中加载预训练权重（这需要两倍于内存中模型大小的内存，一半用于随机初始化的模型，一半用于权重），而是可以选择创建模型作为空壳，然后仅在加载预训练权重时才实现其参数。

可以通过`low_cpu_mem_usage=True`来激活此选项。模型首先在元设备上创建（带有空权重），然后在其中加载状态字典（在分片检查点的情况下逐个分片）。这样，最大使用的RAM仅为模型的完整大小。

```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", low_cpu_mem_usage=True)
```

此外，如果模型无法完全适应RAM（目前仅适用于推断），您还可以直接将模型放置在不同的设备上。使用`device_map="auto"`，Accelerate将确定如何将每个层放置在最快设备（GPU）上，并将其余部分卸载到CPU，甚至硬盘，如果您没有足够的GPU RAM（或CPU RAM）。即使模型分布在多个设备上，它也将按照您通常期望的方式运行。

在传递`device_map`时，`low_cpu_mem_usage`会自动设置为`True`，因此您不需要指定它：

```py
from transformers import AutoModelForSeq2SeqLM

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

您可以通过查看其`hf_device_map`属性来查看模型如何在设备上分割：

```py
t0pp.hf_device_map
```

```python out
{'shared': 0,
 'decoder.embed_tokens': 0,
 'encoder': 0,
 'decoder.block.0': 0,
 'decoder.block.1': 1,
 'decoder.block.2': 1,
 'decoder.block.3': 1,
 'decoder.block.4': 1,
 'decoder.block.5': 1,
 'decoder.block.6': 1,
 'decoder.block.7': 1,
 'decoder.block.8': 1,
 'decoder.block.9': 1,
 'decoder.block.10': 1,
 'decoder.block.11': 1,
 'decoder.block.12': 1,
 'decoder.block.13': 1,
 'decoder.block.14': 1,
 'decoder.block.15': 1,
 'decoder.block.16': 1,
 'decoder.block.17': 1,
 'decoder.block.18': 1,
 'decoder.block.19': 1,
 'decoder.block.20': 1,
 'decoder.block.21': 1,
 'decoder.block.22': 'cpu',
 'decoder.block.23': 'cpu',
 'decoder.final_layer_norm': 'cpu',
 'decoder.dropout': 'cpu',
 'lm_head': 'cpu'}
```

您还可以根据相同的格式编写自己的设备映射（字典层名称到设备）。它应将模型的所有参数映射到给定设备，但如果该层完全位于同一设备上，则无需详细说明该层的所有子模块放在何处。例如，以下设备映射将适用于T0pp（只要您具有GPU内存）：

```python
device_map = {"shared": 0, "encoder": 0, "decoder": 1, "lm_head": 1}
```

减小模型对内存的影响的另一种方法是以较低的精度dtype（如`torch.float16`）实例化模型，或使用下面描述的直接量化技术。

### 模型实例化dtype

在PyTorch中，模型通常以`torch.float32`格式实例化。如果尝试加载权重为fp16的模型，则可能会有问题，因为它需要两倍的内存。为了克服这个限制，您可以使用`torch_dtype`参数显式传递所需的`dtype`：

```python
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype=torch.float16)
```
$或者，如果希望模型始终以最佳内存模式加载，则可以使用特殊值`"auto"`，然后`dtype`将自动从模型的权重中推导出：

```python
model = T5ForConditionalGeneration.from_pretrained("t5", torch_dtype="auto")
```

从头开始实例化的模型也可以告知使用哪个`dtype`：

```python
config = T5Config.from_pretrained("t5")
model = AutoModel.from_config(config)
```

由于PyTorch的设计，此功能仅适用于浮点dtype。


## ModuleUtilsMixin

[[autodoc]] modeling_utils.ModuleUtilsMixin

## TFPreTrainedModel

[[autodoc]] TFPreTrainedModel
    - push_to_hub
    - all

## TFModelUtilsMixin

[[autodoc]] modeling_tf_utils.TFModelUtilsMixin

## FlaxPreTrainedModel

[[autodoc]] FlaxPreTrainedModel
    - push_to_hub
    - all

## 推送到Hub

[[autodoc]] utils.PushToHubMixin

## 分片检查点

[[autodoc]] modeling_utils.load_sharded_checkpoint
