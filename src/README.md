# Fine Tuning a Llama based LLM

This document shows how to fine tuned Llama mode with HuggingFace's model weights.

## Fine Tuning
Go to src/llama_recipes/configs and set config parameters there in the python files. Afterwards, run following command to run distributed fine tuning on your machine.

```
torchrun --nnodes 1 --nproc_per_node 8 src/finetuning.py
```

## Merging the weights (Only required if LoRA method was used) 
In case the model was fine tuned with LoRA method we need to merge the weights of the base model with the adapter weight. For this we can use the script `merge_lora_weights.py` which is located in the same folder as this README file.

The script takes the base model, the peft weight folder as well as an output as arguments:

```
python src/merge_lora_weights.py --base_model PATH/or/HF/base_model_name --peft_model PATH/to/PEFT/Checkpoints --output_dir PATH/to/save/merged/HF/checkpoints
```

## FSDP model weights to HF model weights
In case the model was fine tuned without PEFT we will need to convert the FSDP checkpoint to Huggingface checkpoint. For this we can use the script `fsdp_checkpoint_to_hf.py` which is located in the same folder as this README file.

The script takes the base model, the fsdp weight folder as well as an output as arguments:

```
 python src/fsdp_checkpoint_to_hf.py --fsdp_checkpoint_path PATH/to/FSDP/Checkpoints --consolidated_model_path PATH/to/save/HF/checkpoints --HF_model_path_or_name PATH/or/HF/base_model_name
```








