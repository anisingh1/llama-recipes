# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="/home/anisingh/models/LlamaGuard-7b"
    enable_fsdp: bool=True
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=4
    batching_strategy: str="packing" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=1
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=42
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "llamaguard_dataset"
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=True
    output_dir: str = "/home/anisingh/models/trained"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="/home/anisingh/models/fsdp" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    save_metrics: bool = True # saves training metrics to a json file for later plotting
