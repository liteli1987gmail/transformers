<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）许可;您不得使用此文件，除非符合许可证的规定。您可以在以下网址获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据本许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的保证或条件。请参阅许可证以获取特定语言下的权限和限制。

⚠️请注意，此文件为Markdown格式，但包含我们doc-builder的特定语法（类似于MDX），可能无法在Markdown查看器中正确渲染。

-->

# 将 🤗 Transformers 模型导出为 ONNX

🤗 Transformers 提供了一个`transformers.onnx`包，通过利用配置对象，您可以将模型检查点转换为ONNX图形。

有关导出🤗 Transformers模型的详细信息，请参阅[指南](../serialization)。

## ONNX配置

我们提供了三个抽象类，您应该从中继承，具体取决于您希望导出的模型架构类型：

* 基于编码器的模型继承自[`~onnx.config.OnnxConfig`]
* 基于解码器的模型继承自[`~onnx.config.OnnxConfigWithPast`]
* 编码器-解码器模型继承自[`~onnx.config.OnnxSeq2SeqConfigWithPast`]

### OnnxConfig

[[autodoc]] onnx.config.OnnxConfig

### OnnxConfigWithPast

[[autodoc]] onnx.config.OnnxConfigWithPast

### OnnxSeq2SeqConfigWithPast

[[autodoc]] onnx.config.OnnxSeq2SeqConfigWithPast

## ONNX功能

每个ONNX配置与一组_功能_相关联，使您能够为不同类型的拓扑或任务导出模型。

### FeaturesManager

[[autodoc]] onnx.features.FeaturesManager

