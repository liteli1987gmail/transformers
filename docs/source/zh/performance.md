<!---版权所有 2021 年 The HuggingFace 团队。保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）授权; 除非符合许可证的规定，否则不得使用此文件。您可以在以下网址获取许可证副本
    http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，否则依照许可证分发的软件在“按原样”基础上分发，不附带任何明示或暗示的担保或条件。有关特定语言的权限和许可证的详细信息，请参阅许可证。许可下的限制。
⚠️请注意，此文件是 Markdown 格式，但包含我们的文档生成器（类似于 MDX）的特定语法，可能无法在您的 Markdown 查看器中正确显示。
-->

# 性能和可扩展性

训练越来越大的转换模型并将其部署到生产环境中会面临一系列挑战。在训练期间，您的模型可能需要更多的 GPU 内存，或者训练速度非常慢，当您将其部署进行推理时，可能会被生产环境中所需的吞吐量所压倒。本文档旨在帮助您应对这些挑战，找到适合您用例的最佳设置。我们将指南分为训练和推理，因为它们具有不同的挑战和解决方案。然后，在每个部分中，我们都为不同类型的硬件设置（例如，单 GPU 与多 GPU 训练，或 CPU 与 GPU 推理）提供单独的指南。
![perf_overview](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/perf_overview.png)
本文档旨在为您的场景提供有用的方法概述和入口。

## 训练

有效地训练转换模型需要像 GPU 或 TPU 这样的加速器。最常见的情况是您只有一张 GPU，但也有关于多 GPU 和 CPU 训练的部分（更多内容即将推出）。
<Tip>

 注意：在单 GPU 部分介绍的大多数策略（例如混合精度训练或梯度累积）是通用的，并适用于一般的模型训练，因此在深入研究多 GPU 或 CPU 训练等其他部分之前，请务必查看它。
</Tip>

### 单 GPU

在单个 GPU 上训练大型模型可能具有挑战性，但有许多工具和方法可使其可行。本节讨论了混合精度训练、梯度累积和检查点、高效优化器以及确定最佳批量大小的策略等方法。

[转到单 GPU 训练部分](perf_train_gpu_one)

### 多 GPU

在某些情况下，单个 GPU 上的训练仍然过慢或无法容纳大型模型。转向多 GPU 设置是合乎逻辑的一步，但同时在多个 GPU 上训练涉及到新的决策：每个 GPU 是否拥有模型的完整副本，还是模型本身也进行了分布式处理？在本节中，我们将研究数据、张量和管道并行性。

[转到多 GPU 训练部分](perf_train_gpu_many)
### CPU

[转到 CPU 训练部分](perf_train_cpu)

### TPU

[_即将推出_](perf_train_tpu)

### 专用硬件

[_即将推出_](perf_train_special)

## 推理

在生产环境中使用大型模型进行高效的推理与训练它们一样具有挑战性。在以下各节中，我们将介绍在 CPU 和单/多 GPU 设置上运行推理的步骤。
### CPU

[转到 CPU 推理部分](perf_infer_cpu)

### 单 GPU

[转到单 GPU 推理部分](perf_infer_gpu_one)

### 多 GPU

[转到多 GPU 推理部分](perf_infer_gpu_many)

### 专用硬件

[_即将推出_](perf_infer_special)

## 硬件

在硬件部分，您可以找到在构建自己的深度学习设备时的提示和技巧。
[转到硬件部分](perf_hardware)

## 贡献

这份文档还远未完成，还需要添加许多内容，因此，如果您有任何补充或更正意见，请不要犹豫，打开一个 PR，或者如果您不确定，请发起一个 Issue，我们可以在那里讨论细节。

在提出 A 优于 B 的贡献时，请尽量包含可复制的基准测试和/或该信息来源的链接（除非信息直接来自您）。