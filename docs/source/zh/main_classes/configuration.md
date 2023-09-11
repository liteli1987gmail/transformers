<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）进行许可；除非符合许可证，否则您不得使用此文件。您可以在以下网址获取许可证副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则依据许可证分发的软件是在“按原样”基础上提供的，不附带任何明示或暗示的担保或条件。请参阅许可证以了解特定语言下的权限和限制。

⚠️ 请注意，此文件是Markdown格式的，但包含了我们的文档构建器（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。

-->

# 配置

基类[`PretrainedConfig`]实现了加载/保存配置的常用方法，无论是从本地文件或目录加载，还是从库中提供的预训练模型配置（从HuggingFace的AWS S3存储库下载）。

每个派生的配置类都实现了模型特定的属性。所有配置类中的通用属性是:
`hidden_size`，`num_attention_heads`和`num_hidden_layers`。文本模型还实现了:
`vocab_size`。


## PretrainedConfig

[[autodoc]] PretrainedConfig
    - push_to_hub
    - all
