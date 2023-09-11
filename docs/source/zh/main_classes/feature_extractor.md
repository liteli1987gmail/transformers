<!--版权所有2021年The HuggingFace Team。保留所有权利。

根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言下的权限和限制。

⚠️ 请注意，此文件是Markdown格式的，但包含我们的文档构建器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确渲染。

-->

# 特征提取器

特征提取器负责为音频或视觉模型准备输入特征。这包括从序列中提取特征，例如将音频文件预处理为Log-Mel Spectrogram特征，从图像中提取特征，例如裁剪图像文件，还包括填充、归一化和转换为Numpy、PyTorch和TensorFlow张量。


## 特征提取混合类

[[autodoc]] feature_extraction_utils.FeatureExtractionMixin
    - from_pretrained
    - save_pretrained

## 序列特征提取器

[[autodoc]] SequenceFeatureExtractor
    - pad

## 批次特征

[[autodoc]] BatchFeature

## 图像特征提取混合类

[[autodoc]] image_utils.ImageFeatureExtractionMixin
