from transformers import TrainerCallback
from tqdm import tqdm

from transformers import TrainerCallback
from tqdm import tqdm

class CustomProgressBarCallback(TrainerCallback):
    """
    Augmented class of `transformers.TrainerCallback` that displays a custom progress bar for training and evaluation.
    """
    def __init__(self, num_epochs):
        super().__init__()
        self.training_bar = None
        self.prediction_bar = None
        self.num_epochs = num_epochs

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar = tqdm(
                total=state.max_steps,
                desc=f"Epoch 1/{self.num_epochs} [Train]",
                bar_format="{desc}\n{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]\n{postfix}",
                postfix=dict(loss="0.0000", lr="0.0"),
                dynamic_ncols=True,
                initial=0
            )

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.update(1)
            current_epoch = int(state.epoch) + 1
            self.training_bar.set_description(f"Epoch {current_epoch}/{self.num_epochs} [Train]")

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader),
                    desc=f"Epoch {int(state.epoch) + 1}/{self.num_epochs} [Valid]",
                    bar_format="{desc}\n{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]\n{postfix}",
                    postfix=dict(loss="0.0000"),
                    dynamic_ncols=True,
                    initial=0,
                    leave=self.training_bar is None
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            if logs and state.log_history:
                loss = logs.get('loss', 0)
                lr = logs.get('learning_rate', 0)
                self.training_bar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.6f}")
            self.training_bar.write(str(logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.training_bar.close()
            self.training_bar = None
