import transformers
from torch import nn
from transformers import TrainerControl, TrainerState, TrainingArguments

import os
import wandb
import torch
import time
import psutil
from spft.modules import SparseModule
from spft.utils.io import rank0_print

__all__ = ["SPFTCallback", "MemoryTimeCallback"]


class SPFTCallback(transformers.TrainerCallback):
    def __init__(self, start_step: float = 0, end_step: float = 1) -> None:
        self.start_step = start_step
        self.end_step = end_step
        self.prev_state = False

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ) -> None:
        start_step = self.start_step * state.max_steps if self.start_step <= 1 else self.start_step
        end_step = self.end_step * state.max_steps if self.end_step <= 1 else self.end_step

        
        for module in model.modules():
            if isinstance(module, SparseModule):
                module.enabled = start_step <= state.global_step < end_step
                current_state = module.enabled
                
        if current_state and not self.prev_state:
            rank0_print(f"SPFT: Enabled at step {state.global_step} / {state.max_steps}")
            self.prev_state = True
            
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: nn.Module,
        **kwargs,
    ) -> None:
        if wandb.run is None:
            return
        for name, module in model.named_modules():
            if isinstance(module, SparseModule):
                for key, val in module.stats.items():
                    wandb.log({f"stats/{name}/{key}": val}, step=state.global_step)
                module.stats.clear()

class EvaluateFirstStepCallback(transformers.integrations.integration_utils.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1:
            control.should_evaluate = True


class MemoryTimeCallback(transformers.TrainerCallback):
    def __init__(self):
        super().__init__()
        self.batch_times = []
        self.max_memory_allocated = 0
        self.process = psutil.Process(os.getpid())

    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
    def on_step_begin(self, args, state, control, **kwargs):
        self.batch_start_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        batch_time = time.time() - self.batch_start_time
        self.batch_times.append(batch_time)
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  
            self.max_memory_allocated = max(self.max_memory_allocated, memory_allocated)
            
        if state.global_step % args.logging_steps == 0:
            print(f"Step {state.global_step}: "
                  f"Batch time: {batch_time:.3f}s, "
                  f"Max GPU memory: {self.max_memory_allocated:.2f} GB")
    
    def on_train_end(self, args, state, control, **kwargs):
        if len(self.batch_times) > 0:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            print(f"\nTraining completed!")
            print(f"Average batch time: {avg_batch_time:.3f}s")
            print(f"Max GPU memory used: {self.max_memory_allocated:.2f} GB")
            print(f"Total training steps: {state.global_step}")
