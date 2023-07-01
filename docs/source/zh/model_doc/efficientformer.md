<!-- 版权所有 2022 年 HuggingFace 团队保留所有权利。
根据 Apache 许可证第 2.0 版（“许可证”）许可；除非符合许可证，否则您不得使用此文件。您可以在以下位置获取许可证的副本
http://www.apache.org/licenses/LICENSE-2.0
除非适用法律要求或书面同意，根据许可证进行的软件分发基于“按原样”基础，没有任何明示或暗示的保证或条件。请参阅许可证以了解特定语言下的权限和限制。
⚠️请注意，此文件是 Markdown 格式的，但包含有关我们的 doc-builder（类似于 MDX）的特定语法，这些语法可能无法在您的 Markdown 查看器中正确显示。
-->
# EfficientFormer
## 概述
EfficientFormer 模型是由 Yanyu Li，Geng Yuan，Yang Wen，Eric Hu，Georgios Evangelidis，Sergey Tulyakov，Yanzhi Wang，Jian Ren 在 [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191) 中提出的。EfficientFormer 提出了一种在移动设备上运行的维度一致的纯 Transformer，用于密集预测任务，如图像分类，物体检测和语义分割。

从论文中摘录的摘要如下所示：

*Vision Transformers（ViT）在计算机视觉任务中取得了快速进展，在各种基准测试中取得了有希望的结果。但是，由于参数和模型设计的大规模数量，例如注意机制，基于 ViT 的模型通常比轻量级卷积网络慢得多。因此，将 ViT 部署到实时应用程序中尤为具有挑战性，特别是在资源受限的硬件上，如移动设备。最近的努力尝试通过网络架构搜索或与 MobileNet 块的混合设计来减少 ViT 的计算复杂性，但推理速度仍然不尽人意。这引出了一个重要的问题：Transformer 能否与 MobileNet 一样快速，同时实现高性能？为了回答这个问题，我们首先重新审视在基于 ViT 的模型中使用的网络架构和运算符，并确定了低效的设计。然后，我们引入了一个维度一致的纯 Transformer（不包含 MobileNet 块）作为设计范例。最后，我们进行基于延迟的精简，得到了一系列称为 EfficientFormer 的最终模型。广泛的实验证明了 EfficientFormer 在移动设备上的性能和速度的优势。我们最快的模型 EfficientFormer-L1，在 ImageNet-1K 上实现了 79.2%的 top-1 准确率，仅需 1.6 毫秒的推理延迟在 iPhone 12 上（使用 CoreML 编译），与 MobileNetV2 × 1.4（1.6 毫秒，74.7% top-1）一样快，我们最大的模型 EfficientFormer-L7 在仅需 7.0 毫秒的延迟下实现了 83.3%的准确率。我们的工作证明了适当设计的 Transformer 可以在移动设备上实现极低的延迟，同时保持高性能。*

该模型由 [novice03](https://huggingface.co/novice03) 和 [Bearnardd](https://huggingface.co/Bearnardd) 贡献。可以在 [此处](https://github.com/snap-research/EfficientFormer) 找到原始代码。此模型的 TensorFlow 版本由 [D-Roberts](https://huggingface.co/D-Roberts) 添加。

## 文档资源
- [图像分类任务指南](../tasks/image_classification)
## Documentation resources

- [Image classification task guide](../tasks/image_classification)

## EfficientFormerConfig

[[autodoc]] EfficientFormerConfig

## EfficientFormerImageProcessor

[[autodoc]] EfficientFormerImageProcessor
    - preprocess

## EfficientFormerModel

[[autodoc]] EfficientFormerModel
    - forward

## EfficientFormerForImageClassification

[[autodoc]] EfficientFormerForImageClassification
    - forward

## EfficientFormerForImageClassificationWithTeacher

[[autodoc]] EfficientFormerForImageClassificationWithTeacher
    - forward

## TFEfficientFormerModel

[[autodoc]] TFEfficientFormerModel
    - call

## TFEfficientFormerForImageClassification

[[autodoc]] TFEfficientFormerForImageClassification
    - call

## TFEfficientFormerForImageClassificationWithTeacher

[[autodoc]] TFEfficientFormerForImageClassificationWithTeacher
    - call