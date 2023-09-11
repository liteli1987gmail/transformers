<!--版权2023年HuggingFace团队。版权所有。

根据Apache许可证2.0版本（“许可证”），您除非符合许可证，否则不得使用此文件。您可以在
许可证。您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是按“现状”分发的，没有任何明示或暗示的担保或条件。请参阅许可证的
特定语言管理权限和限制条件的限制。

⚠️请注意，此文件是Markdown格式的，但包含我们doc-builder的特定语法（类似于MDX），在您的Markdown查看器中可能无法正确呈现。

-->

# 量化 🤗 Transformers 模型

## `bitsandbytes` 集成

🤗 Transformers 与 `bitsandbytes` 上使用最多的模块密切集成。您可以在几行代码中以8位精度加载模型。
从`bitsandbytes`发布的`0.37.0`版本开始，大多数GPU硬件都支持此功能。

有关量化方法的更多信息，请参阅[LLM.int8()](https://arxiv.org/abs/2208.07339)论文，或者关于合作的[博客文章](https://huggingface.co/blog/hf-bitsandbytes-integration)。
从`0.39.0`版本开始，您可以使用4位量化加载支持`device_map`的任何模型，利用FP4数据类型。

以下是使用`bitsandbytes`集成可以执行的操作

### FP4量化 

#### 要求

在运行下面的代码片段之前，请确保已安装以下要求。

- 最新的`bitsandbytes`库
`pip install bitsandbytes>=0.39.0`

- 从源代码安装最新的`accelerate`
`pip install git+https://github.com/huggingface/accelerate.git`

- 从源代码安装最新的`transformers` 
`pip install git+https://github.com/huggingface/transformers.git`

#### 使用4位加载大型模型

在调用`.from_pretrained`方法时使用`load_in_4bit=True`，可以将内存使用量减少4倍（大致）。

```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_4bit=True)
```

<Tip warning={true}>

请注意，一旦模型以4位加载，目前无法将量化的权重推送到Hub。请注意，尚不支持训练4位权重。但是，您可以使用4位模型来训练额外的参数，这将在下一节中介绍。

</Tip>

### 使用8位加载大型模型

通过在调用`.from_pretrained`方法时使用`load_in_8bit=True`参数，可以将内存要求大致减半。


```python
# pip install transformers accelerate bitsandbytes
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bigscience/bloom-1b7"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", load_in_8bit=True)
```

然后，像通常使用[`PreTrainedModel`]一样使用模型。

您可以使用`get_memory_footprint`方法检查模型的内存占用。

```python
print(model.get_memory_footprint())
```

通过此集成，我们能够在较小的设备上加载大型模型并无缝运行它们。

<Tip warning={true}>

请注意，一旦模型以8位加载，目前无法将量化的权重推送到Hub，除非您使用最新的`transformers`和`bitsandbytes`。请注意，尚不支持训练8位权重。但是，您可以使用8位模型来训练额外的参数，这将在下一节中介绍。
还要注意`device_map`是可选的，但为了推理效果好，设置`device_map = 'auto'`是首选，它将有效地将模型分派到可用的资源上。

</Tip>

#### 高级用例

这里将介绍一些使用FP4量化的高级用例

##### 更改计算数据类型

计算数据类型用于更改计算过程中将使用的数据类型。例如，隐藏状态可以为`float32`，但计算可以设置为bf16以提高速度。默认情况下，计算数据类型设置为`float32`。

```python
import torch
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

##### 使用NF4（Normal Float 4）数据类型 

您还可以使用NF4数据类型，这是一种针对使用正态分布初始化的权重的新的4位数据类型。为此运行:

```python
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
```

##### 使用嵌套量化进行更高效的内存推断

我们还建议用户使用嵌套量化技术。这样可以节省更多的内存，而不会增加额外的性能-根据我们的经验观察，这使得在具有1024个序列长度、1个批次大小和4个梯度累积步骤的NVIDIA-T4 16GB上微调llama-13b模型成为可能。

```python
from transformers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

model_double_quant = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=double_quant_config)
```


### 将量化模型推送到🤗 Hub

您可以通过简单地使用`push_to_hub`方法将量化模型推送到Hub。这首先会推送量化配置文件，然后推送量化模型权重。
请确保使用`bitsandbytes>0.37.2`（在撰写本文时，我们在`bitsandbytes==0.38.0.post1`上进行了测试）以便使用此功能。 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m", device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")

model.push_to_hub("bloom-560m-8bit")
```

<Tip warning={true}>

强烈建议将8位模型推送到Hub以用于大型模型。这将使社区能够从内存占用减少和加载的好处中受益，例如在Google Colab上加载大型模型。

</Tip>

### 从🤗 Hub加载量化模型

您可以使用`from_pretrained`方法从Hub加载量化模型。通过检查模型配置对象中是否存在属性`quantization_config`，确保推送的权重经过量化。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{your_username}/bloom-560m-8bit", device_map="auto")
```
在这种情况下，您不需要指定参数`load_in_8bit=True`，但您需要确保已安装`bitsandbytes`和`accelerate`。
还要注意`device_map`是可选的，但为了推理效果好，设置`device_map = 'auto'`是首选，它将有效地将模型分派到可用的资源上。

### 高级用例

本节面向高级用户，希望探索在加载和运行8位模型之外可以做什么。

#### 在`cpu`和`gpu`之间进行卸载

这其中一个高级用例是能够加载模型并在`CPU`和`GPU`之间分派权重。请注意，将在CPU上分派的权重**不会**转换为8位，因此保持为`float32`。此功能适用于希望适应非常大的模型并在GPU和CPU之间分派模型的用户。

首先，从`transformers`加载`BitsAndBytesConfig`并将属性`llm_int8_enable_fp32_cpu_offload`设置为`True`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
```

假设您想要加载`bigscience/bloom-1b7`模型，并且您的GPU RAM刚好足够容纳整个模型，除了`lm_head`。因此，编写自定义的device_map如下:
```python
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
}
```

并按以下方式加载模型:
```python
model_8bit = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-1b7",
    device_map=device_map,
    quantization_config=quantization_config,
)
```

就是这样！享受您的模型吧！

#### 使用`llm_int8_threshold`进行调整

您可以使用`llm_int8_threshold`参数来更改异常值的阈值。"异常值"是大于某个特定阈值的隐藏状态值。
这对应于`LLM.int8()`论文中描述的异常值检测的异常值阈值。任何超过此阈值的隐藏状态值都将被视为异常值，并且在这些值上的操作将以fp16进行。这些值通常服从正态分布，即大多数值在[-3.5, 3.5]范围内，但对于大型模型，有一些例外的系统异常值分布非常不同。这些异常值通常在区间[-60, -6]或[6, 60]内。对于大小约为5的值，int8量化效果很好，但超过这个范围，性能会明显下降。一个很好的默认阈值是6，但对于更不稳定的模型（小型模型，微调），可能需要较低的阈值。
此参数会影响模型的推理速度。我们建议尝试不同的参数值，找到最适合您用例的参数值。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_threshold=10,
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### 跳过某些模块的转换

某些模型具有多个模块，这些模块在量化时不应转换为8位，以确保稳定性。例如，Jukebox模型具有多个应跳过的`lm_head`模块。使用`llm_int8_skip_modules`进行尝试 

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "bigscience/bloom-1b7"

quantization_config = BitsAndBytesConfig(
    llm_int8_skip_modules=["lm_head"],
)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

#### 在8位加载的模型上进行微调

在Hugging Face生态系统中支持适配器的官方支持下，您可以微调已经以8位加载的模型。
这使得可以在单个Google Colab中微调诸如`flan-t5-large`或`facebook/opt-6.7b`之类的大型模型。请参阅[`peft`](https://github.com/huggingface/peft)库以了解更多详细信息。

请注意，在加载模型进行训练时不需要传递`device_map`参数。它将自动将模型加载到您的GPU上。如果需要，您还可以将设备映射设置为特定设备（例如`cuda:0`，`0`，`torch.device('cuda:0')`）。请注意，仅推理时应使用`device_map=auto`。 

### BitsAndBytesConfig

[[autodoc]] BitsAndBytesConfig


## 使用🤗 `optimum` 进行量化 

请查看[Optimum文档](https://huggingface.co/docs/optimum/index)以了解有关`optimum`支持的量化方法的更多信息，并查看这些方法是否适用于您的用例。

