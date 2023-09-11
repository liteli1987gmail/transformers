<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证2.0版（“许可证”）获得许可；除非符合
许可证的规定，否则您不能使用此文件。您可以在
以下网址获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则以“AS IS”方式分发的软件
在明示或暗示的情况下均不附带任何担保或条件。请参阅许可证
关于特定语言的使用权限和限制。

⚠️请注意，此文件是Markdown格式，但包含特定语法，用于我们的doc-builder（类似于MDX），可能无法
在您的Markdown查看器中正确渲染。

-->

# 模型输出

所有模型都有作为[`~utils.ModelOutput`]子类实例的输出。这些是
data结构，其中包含模型返回的所有信息，但也可以用作元组或
dictionaries。

让我们来看一个例子:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1]).unsqueeze(0)  # 批大小 1
outputs = model(**inputs, labels=labels)
```

`outputs`对象是一个[`~modeling_outputs.SequenceClassifierOutput`]，如我们在
documentation该类下面可以看到，它意味着它具有一个可选的`loss`，一个`logits`，一个可选的`hidden_states`和
an可选的`attentions`属性。在这里，我们有`loss`，因为我们传递了`labels`，但我们没有$`hidden_states`和`attentions`，因为我们没有传递`output_hidden_states=True`或$`output_attentions=True`。

您可以像通常那样访问每个属性，如果该属性没有被模型返回，您将得到`None`。在这里，例如，
`outputs.loss`是模型计算的损失，`outputs.attentions`是$`None`。

当将我们的`outputs`对象视为元组时，它仅考虑没有`None`值的属性。在这里，例如，它有$两个元素，`loss`然后是`logits`，所以

```python
outputs[:2]
```

将返回元组`(outputs.loss, outputs.logits)`。

当将我们的`outputs`对象视为字典时，它仅考虑没有`None`的属性
值。在这里，例如，它有两个键，它们是`loss`和`logits`。

我们在这里记录了通用模型输出，这些通用模型输出被多个模型类型使用。特定输出类型的
记录在相应的模型页面上。

## ModelOutput

[[autodoc]] utils.ModelOutput
    - to_tuple

## BaseModelOutput

[[autodoc]] modeling_outputs.BaseModelOutput

## BaseModelOutputWithPooling

[[autodoc]] modeling_outputs.BaseModelOutputWithPooling

## BaseModelOutputWithCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithCrossAttentions

## BaseModelOutputWithPoolingAndCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions

## BaseModelOutputWithPast

[[autodoc]] modeling_outputs.BaseModelOutputWithPast

## BaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_outputs.BaseModelOutputWithPastAndCrossAttentions

## Seq2SeqModelOutput

[[autodoc]] modeling_outputs.Seq2SeqModelOutput

## CausalLMOutput

[[autodoc]] modeling_outputs.CausalLMOutput

## CausalLMOutputWithCrossAttentions

[[autodoc]] modeling_outputs.CausalLMOutputWithCrossAttentions

## CausalLMOutputWithPast

[[autodoc]] modeling_outputs.CausalLMOutputWithPast

## MaskedLMOutput

[[autodoc]] modeling_outputs.MaskedLMOutput

## Seq2SeqLMOutput

[[autodoc]] modeling_outputs.Seq2SeqLMOutput

## NextSentencePredictorOutput

[[autodoc]] modeling_outputs.NextSentencePredictorOutput

## SequenceClassifierOutput

[[autodoc]] modeling_outputs.SequenceClassifierOutput

## Seq2SeqSequenceClassifierOutput

[[autodoc]] modeling_outputs.Seq2SeqSequenceClassifierOutput

## MultipleChoiceModelOutput

[[autodoc]] modeling_outputs.MultipleChoiceModelOutput

## TokenClassifierOutput

[[autodoc]] modeling_outputs.TokenClassifierOutput

## QuestionAnsweringModelOutput

[[autodoc]] modeling_outputs.QuestionAnsweringModelOutput

## Seq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_outputs.Seq2SeqQuestionAnsweringModelOutput

## Seq2SeqSpectrogramOutput

[[autodoc]] modeling_outputs.Seq2SeqSpectrogramOutput

## SemanticSegmenterOutput

[[autodoc]] modeling_outputs.SemanticSegmenterOutput

## ImageClassifierOutput

[[autodoc]] modeling_outputs.ImageClassifierOutput

## ImageClassifierOutputWithNoAttention

[[autodoc]] modeling_outputs.ImageClassifierOutputWithNoAttention

## DepthEstimatorOutput

[[autodoc]] modeling_outputs.DepthEstimatorOutput

## Wav2Vec2BaseModelOutput

[[autodoc]] modeling_outputs.Wav2Vec2BaseModelOutput

## XVectorOutput

[[autodoc]] modeling_outputs.XVectorOutput

## Seq2SeqTSModelOutput

[[autodoc]] modeling_outputs.Seq2SeqTSModelOutput

## Seq2SeqTSPredictionOutput

[[autodoc]] modeling_outputs.Seq2SeqTSPredictionOutput

## SampleTSPredictionOutput

[[autodoc]] modeling_outputs.SampleTSPredictionOutput

## TFBaseModelOutput

[[autodoc]] modeling_tf_outputs.TFBaseModelOutput

## TFBaseModelOutputWithPooling

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPooling

## TFBaseModelOutputWithPoolingAndCrossAttentions

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPoolingAndCrossAttentions

## TFBaseModelOutputWithPast

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPast

## TFBaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_tf_outputs.TFBaseModelOutputWithPastAndCrossAttentions

## TFSeq2SeqModelOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqModelOutput

## TFCausalLMOutput

[[autodoc]] modeling_tf_outputs.TFCausalLMOutput

## TFCausalLMOutputWithCrossAttentions

[[autodoc]] modeling_tf_outputs.TFCausalLMOutputWithCrossAttentions

## TFCausalLMOutputWithPast

[[autodoc]] modeling_tf_outputs.TFCausalLMOutputWithPast

## TFMaskedLMOutput

[[autodoc]] modeling_tf_outputs.TFMaskedLMOutput

## TFSeq2SeqLMOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqLMOutput

## TFNextSentencePredictorOutput

[[autodoc]] modeling_tf_outputs.TFNextSentencePredictorOutput

## TFSequenceClassifierOutput

[[autodoc]] modeling_tf_outputs.TFSequenceClassifierOutput

## TFSeq2SeqSequenceClassifierOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqSequenceClassifierOutput

## TFMultipleChoiceModelOutput

[[autodoc]] modeling_tf_outputs.TFMultipleChoiceModelOutput

## TFTokenClassifierOutput

[[autodoc]] modeling_tf_outputs.TFTokenClassifierOutput

## TFQuestionAnsweringModelOutput

[[autodoc]] modeling_tf_outputs.TFQuestionAnsweringModelOutput

## TFSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_tf_outputs.TFSeq2SeqQuestionAnsweringModelOutput

## FlaxBaseModelOutput

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutput

## FlaxBaseModelOutputWithPast

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPast

## FlaxBaseModelOutputWithPooling

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPooling

## FlaxBaseModelOutputWithPastAndCrossAttentions

[[autodoc]] modeling_flax_outputs.FlaxBaseModelOutputWithPastAndCrossAttentions

## FlaxSeq2SeqModelOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqModelOutput

## FlaxCausalLMOutputWithCrossAttentions

[[autodoc]] modeling_flax_outputs.FlaxCausalLMOutputWithCrossAttentions

## FlaxMaskedLMOutput

[[autodoc]] modeling_flax_outputs.FlaxMaskedLMOutput

## FlaxSeq2SeqLMOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqLMOutput

## FlaxNextSentencePredictorOutput

[[autodoc]] modeling_flax_outputs.FlaxNextSentencePredictorOutput

## FlaxSequenceClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxSequenceClassifierOutput

## FlaxSeq2SeqSequenceClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqSequenceClassifierOutput

## FlaxMultipleChoiceModelOutput

[[autodoc]] modeling_flax_outputs.FlaxMultipleChoiceModelOutput

## FlaxTokenClassifierOutput

[[autodoc]] modeling_flax_outputs.FlaxTokenClassifierOutput

## FlaxQuestionAnsweringModelOutput

[[autodoc]] modeling_flax_outputs.FlaxQuestionAnsweringModelOutput

## FlaxSeq2SeqQuestionAnsweringModelOutput

[[autodoc]] modeling_flax_outputs.FlaxSeq2SeqQuestionAnsweringModelOutput
