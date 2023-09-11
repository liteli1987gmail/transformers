<!--版权所有2022年The HuggingFace Team。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可；除非符合许可证的规定，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http：//www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于“按原样”基础分发的，不附带任何明示或暗示的担保或条件。有关特定语言规定权限和许可限制的详细信息，请参阅许可证。

⚠️请注意，此文件使用Markdown编写，但包含我们的doc-builder（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确渲染。

-->

# 图像处理器

图像处理器负责为视觉模型准备输入特征并对其输出进行后处理。这包括诸如调整大小、归一化和转换为PyTorch、TensorFlow、Flax和Numpy张量的转换。它还可以包括模型特定的后处理，如将对数转换为分割掩码。


## 图像处理混合

[[autodoc]] image_processing_utils.ImageProcessingMixin
    - from_pretrained
    - save_pretrained

## BatchFeature

[[autodoc]] BatchFeature

## BaseImageProcessor

[[autodoc]] image_processing_utils.BaseImageProcessor
