from transformers import TrainerCallback
from tqdm import tqdm

class CustomProgressBarCallback(TrainerCallback):
    def __init__(self, num_epochs):
        super().__init__()
        self.progress_bar = None
        self.num_epochs = num_epochs

    def on_epoch_begin(self, _, state, *_args, **_kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()
            
        self.progress_bar = tqdm(
            total=state.max_steps,
            desc=f"Epoch {int(state.epoch) + 1}/{self.num_epochs}",
            bar_format="{desc}\n{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]\n{postfix}",
            postfix=dict(loss="0.0000"),
            dynamic_ncols=True,
            initial=0
        )

    def on_step_end(self, _, state, *_args, **_kwargs):
        if self.progress_bar is not None:
            self.progress_bar.update(1)
            if state.log_history:
                loss = state.log_history[-1].get('loss', 0)
                self.progress_bar.set_postfix(loss=f"{loss:.4f}")

    def on_epoch_end(self, *_args, **_kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()
