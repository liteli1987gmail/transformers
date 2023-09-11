<!--版权所有2022年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）获得许可；除非符合许可证的要求，否则您不得使用此文件。您可以在以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，不附带任何明示或暗示的担保或条件。请参阅许可证以获取特定语言下的权限和限制

⚠️请注意，此文件是Markdown格式，但包含我们的doc-builder的特定语法（类似于MDX），可能在您的Markdown查看器中无法正确呈现。

-->

# 生成

每个框架都有一个用于文本生成的generate方法，实现在各自的`GenerationMixin`类中:

- PyTorch [`~generation.GenerationMixin.generate`] 实现在 [`~generation.GenerationMixin`] 中。
- TensorFlow [`~generation.TFGenerationMixin.generate`] 实现在 [`~generation.TFGenerationMixin`] 中。
- Flax/JAX [`~generation.FlaxGenerationMixin.generate`] 实现在 [`~generation.FlaxGenerationMixin`] 中。

无论您选择的框架如何，您都可以使用[`~generation.GenerationConfig`]类实例参数化generate方法。请参考此类以获取生成方法的完整参数列表，这些参数控制生成方法的行为。

要了解如何检查模型的生成配置，了解默认值，如何临时更改参数以及如何创建和保存自定义生成配置，请参考[text generation strategies guide](../generation_strategies)。该指南还解释了如何使用相关功能，例如令牌流。

## 生成配置

[[autodoc]] generation.GenerationConfig
	- from_pretrained
	- from_model_config
	- save_pretrained

## 生成混合类

[[autodoc]] generation.GenerationMixin
	- generate
	- compute_transition_scores
	- greedy_search
	- sample
	- beam_search
	- beam_sample
	- contrastive_search
	- group_beam_search
	- constrained_beam_search

## TF生成混合类

[[autodoc]] generation.TFGenerationMixin
	- generate
	- compute_transition_scores

## Flax生成混合类

[[autodoc]] generation.FlaxGenerationMixin
	- generate
