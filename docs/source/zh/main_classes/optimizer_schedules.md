<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”），您不得使用此文件，除非符合许可证的规定。您可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则按“原样”分发的软件不附带任何明示或暗示的担保或条件。请参阅许可证中的特定语言管理权限和限制。

⚠️请注意，此文件以Markdown格式，但包含我们的doc-builder（类似于MDX）的特定语法，可能无法在您的Markdown查看器中正确呈现。

-->

# 优化

`.optimization`模块提供了:

- 一个带有固定权重衰减的优化器，可用于微调模型，以及
- 几种以schedule对象形式提供的调度

## AdamW（PyTorch）

[[autodoc]] AdamW

## AdaFactor（PyTorch）

[[autodoc]] Adafactor

## AdamWeightDecay（TensorFlow）

[[autodoc]] AdamWeightDecay

[[autodoc]] create_optimizer

## 调度

### 学习率调度（Pytorch）

[[autodoc]] SchedulerType

[[autodoc]] get_scheduler

[[autodoc]] get_constant_schedule

[[autodoc]] get_constant_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_constant_schedule.png"/>

[[autodoc]] get_cosine_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_schedule.png"/>

[[autodoc]] get_cosine_with_hard_restarts_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_cosine_hard_restarts_schedule.png"/>

[[autodoc]] get_linear_schedule_with_warmup

<img alt="" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/warmup_linear_schedule.png"/>

[[autodoc]] get_polynomial_decay_schedule_with_warmup

[[autodoc]] get_inverse_sqrt_schedule

### 热身（TensorFlow）

[[autodoc]] WarmUp

## 梯度策略

### GradientAccumulator（TensorFlow）

[[autodoc]] GradientAccumulator
