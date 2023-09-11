<!--版权所有2020年HuggingFace团队。保留所有权利。

根据Apache许可证第2.0版（“许可证”）发布；除非符合
许可证的规定，否则您不得使用此文件。您可以在以下位置获取许可证的副本

http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，否则根据许可证
分发的软件按“原样”分发，不附带任何明示或暗示的保证或条件。请参阅
许可证中的特定语言规定权限和限制。

⚠️请注意，此文件采用Markdown格式，但包含特定于我们的doc-builder（类似于MDX）的语法，可能无法在您的Markdown查看器中正确渲染。

-->

# 回调

回调是可以自定义PyTorch中的训练循环行为的对象[$`Trainer`]（此功能尚未在TensorFlow中实现），可以检查训练循环状态（用于进度报告，在TensorBoard或其他ML平台上记录日志等）并作出决策（例如提前停止）。

回调是“只读”代码块，除了它们返回的[`TrainerControl`]对象外，它们不能更改训练循环中的任何内容。对于需要更改训练循环的自定义操作，应子类化[`Trainer`]并覆盖所需的方法（有关示例，请参阅[trainer](trainer)）。

默认情况下，[`Trainer`]将使用以下回调:

- [`DefaultFlowCallback`]用于处理日志记录、保存和评估的默认行为。
- [`PrinterCallback`]或[`ProgressCallback`]用于显示进度并打印日志（如果通过[`TrainingArguments`]取消激活tqdm，则使用第一个，否则使用第二个）。
- [`~integrations.TensorBoardCallback`]（如果通过PyTorch >= 1.4或tensorboardX可访问）。
- [`~integrations.WandbCallback`]（如果已安装[wandb](https://www.wandb.com/)）。
- [`~integrations.CometCallback`]（如果已安装[comet_ml](https://www.comet.ml/site/)）。
- [`~integrations.MLflowCallback`]（如果已安装[mlflow](https://www.mlflow.org/)）。
- [`~integrations.NeptuneCallback`]（如果已安装[neptune](https://neptune.ai/)）。
- [`~integrations.AzureMLCallback`]（如果已安装[azureml-sdk](https://pypi.org/project/azureml-sdk/)）。
- [`~integrations.CodeCarbonCallback`]（如果已安装[codecarbon](https://pypi.org/project/codecarbon/)）。
- [`~integrations.ClearMLCallback`]（如果已安装[clearml](https://github.com/allegroai/clearml)）。
- [`~integrations.DagsHubCallback`]（如果已安装[dagshub](https://dagshub.com/)）。
- [`~integrations.FlyteCallback`]（如果已安装[flyte](https://flyte.org/)）。

实现回调的主要类是[`TrainerCallback`]。它获取用于实例化[`Trainer`]的[`TrainingArguments`]，可以通过[`TrainerState`]访问该Trainer的内部状态，并可以通过[
[`TrainerControl`]对训练循环采取某些操作。


## 可用回调

以下是库中可用的[`TrainerCallback`]列表:

[[autodoc]] integrations.CometCallback
    - setup

[[autodoc]] DefaultFlowCallback

[[autodoc]] PrinterCallback

[[autodoc]] ProgressCallback

[[autodoc]] EarlyStoppingCallback

[[autodoc]] integrations.TensorBoardCallback

[[autodoc]] integrations.WandbCallback
    - setup

[[autodoc]] integrations.MLflowCallback
    - setup

[[autodoc]] integrations.AzureMLCallback

[[autodoc]] integrations.CodeCarbonCallback

[[autodoc]] integrations.NeptuneCallback

[[autodoc]] integrations.ClearMLCallback

[[autodoc]] integrations.DagsHubCallback

[[autodoc]] integrations.FlyteCallback

## TrainerCallback

[[autodoc]] TrainerCallback

以下是如何使用PyTorch [`Trainer`]注册自定义回调的示例:

```python
class MyCallback(TrainerCallback):
    "在训练开始时打印一条消息的回调"

    def on_train_begin(self, args, state, control, **kwargs):
        print("开始训练")


trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MyCallback],  # 我们可以通过这种方式传递回调类，也可以传递实例（MyCallback()）
)
```

注册回调的另一种方法是调用`trainer.add_callback()`，如下所示:

```python
trainer = Trainer(...)
trainer.add_callback(MyCallback)
# 或者，我们可以传递回调类的实例
trainer.add_callback(MyCallback())
```

## TrainerState

[[autodoc]] TrainerState

## TrainerControl

[[autodoc]] TrainerControl
