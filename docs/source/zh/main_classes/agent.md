<!--版权2023 The HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）许可；除非符合许可证，否则不得使用此文件。您可以在以下位置获取许可证的副本：

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，软件根据许可证分发，以“按原样”基础，不附带任何明示或暗示的担保或条件。请参阅许可证以了解管理权限和限制的具体语言。

⚠️请注意，该文件虽然是用Markdown编写的，但包含了我们的文档生成器（类似于MDX）的特定语法，这可能不会在您的Markdown查看器中正确呈现。

-->

# 代理与工具

<Tip warning={true}>

Transformers代理是一个实验性的API，随时可能发生变化。代理返回的结果也可能因API或底层模型的更改而有所不同。

</Tip>

要了解更多关于代理和工具，请确保阅读[入门指南](../transformers_agents)。本页面包含底层类的API文档。

## 代理

我们提供三种类型的代理：[`HfAgent`]使用开源模型的推理端点，[`LocalAgent`]在本地使用您选择的模型，[`OpenAiAgent`]使用OpenAI的封闭模型。

### HfAgent

[[autodoc]] HfAgent

### LocalAgent

[[autodoc]] LocalAgent

### OpenAiAgent

[[autodoc]] OpenAiAgent

### AzureOpenAiAgent

[[autodoc]] AzureOpenAiAgent

### Agent

### Agent

[[autodoc]] Agent
    - chat
    - run
    - prepare_for_new_chat

## 工具

### load_tool

[[autodoc]] load_tool

### Tool

[[autodoc]] Tool

### PipelineTool

[[autodoc]] PipelineTool

### RemoteTool

[[autodoc]] RemoteTool

### launch_gradio_demo

[[autodoc]] launch_gradio_demo

## 代理类型

代理可以处理工具之间的任何类型的对象；工具本身是完全多模态的，可以接受并返回文本、图像、音频、视频等多种类型。为了增加工具之间的兼容性，以及在ipython（jupyter, colab, ipython notebooks, ...）中正确呈现这些返回结果，我们围绕这些类型实现了包装器类。

被包装的对象应继续按照最初的方式行为；一个文本对象应仍然表现为字符串，一个图像对象应仍然表现为`PIL.Image`。

这些类型有三个特定的目的：

- 调用类型上的`to_raw`应返回底层对象
- 调用类型上的`to_string`应返回对象作为字符串：在`AgentText`的情况下可能是字符串，但在其他情况下将是对象序列化版本的路径
- 在ipython内核中显示它应正确地显示对象

### AgentText

[[autodoc]] transformers.tools.agent_types.AgentText

### AgentImage

[[autodoc]] transformers.tools.agent_types.AgentImage

### AgentAudio

[[autodoc]] transformers.tools.agent_types.AgentAudio
