from dataclasses import dataclass
from typing import Optional, Iterator, Dict, List
import json
from pathlib import Path
import gc
import argparse
from typing_extensions import Literal
import warnings
import inspect
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import os

from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging
)

@dataclass
class ModelConfig:
    model_name: str
    model_family: str

@dataclass
class QuantConfig:
    # base config
    load_in_4bit: Optional[bool]
    load_in_8bit: Optional[bool]
    bnb_4bit_use_double_quant: Optional[bool]
    bnb_4bit_quant_storage: Optional[torch.dtype]
    bnb_4bit_compute_dtype: Optional[torch.dtype]
    bnb_4bit_quant_type: Optional[str] = None  
    
@dataclass
class LayerConfig:
    layer_quant_list: Optional[List[QuantConfig]]

def create_dtype_map() -> Dict[str, torch.device]:
    mapping = {
        ("float16", "fp16") : torch.float16,
        ("bfloat16",)       : torch.bfloat16,
        ("float32", "fp32") : torch.float32,
    }
    dtype_map = {}
    for keys, value in mapping.items():
        for key in keys:
            dtype_map[key] = value
    return dtype_map


def load_config_from_json(
    json_file: Path,
    config_type: Literal["model", "quant", "layer_swap"]
) -> Iterator[QuantConfig]:
    print(json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        configs = json.load(f)
    match config_type:
        case "model":
            for config in configs:
                yield ModelConfig(**config)
        case "quant":
            for config in configs:
                yield QuantConfig(**config)
        case "layer_swap":
            for config in configs:
                yield LayerSwapConfig(**config)

def generate_decoder_map():
    decoder_map: Dict[str, str] = {
        "llama": "model.layers",
        "gpt_neox": "gpt_neox.layers",
        "mistral": "model.layers",
        "mixtral": "model.layers",
    }
    return decoder_map

DecoderMap = generate_decoder_map()

def set_seed(seed: int = 42):
    random.seed(seed)                          # Python random module
    np.random.seed(seed)                       # NumPy
    torch.manual_seed(seed)                    # PyTorch CPU
    torch.cuda.manual_seed(seed)               # PyTorch GPU
    torch.cuda.manual_seed_all(seed)           # PyTorch multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # Avoid non-deterministic optimizations

def quantize_transformer_blocks(
    model: AutoModelForCausalLM,
    model_config: ModelConfig,
    layerwise_config: LayerConfig,
    ):
    
    if layerwise_config.layer_quant_list is None:
        print("None layer-wise config given, returning the same model")
        return model

    base_model_dtype = model.dtype
    try:
        base_model_quant_config = model.config.quantization_config
    except AttributeError as e:
        base_model_quant_config = None

    decoder_map = generate_decoder_map()

    try:
        blocks = eval(f"model.{decoder_map[model_config.model_family]}")
    except AttributeError as e:
        raise ValueError(f"Unsupported model family '{model_config.model_family}' or invalid layer attribute: {e}")

    dtype_map = create_dtype_map()

    quant_config_list = []

    for i, layer in enumerate(blocks):
        load_in = "fp32"
        dtype = dtype_map[layerwise_config.layer_quant_list[i].bnb_4bit_compute_dtype]
        if layerwise_config.layer_quant_list[i].load_in_4bit:
            load_in = "4bit/" + layerwise_config.layer_quant_list[i].bnb_4bit_quant_type
            if layerwise_config.layer_quant_list[i].bnb_4bit_use_double_quant:
                load_in += "/double"
            else:
                load_in += "/single" 

        elif layerwise_config.layer_quant_list[i].load_in_8bit:
            load_in = "8bit"

        if layerwise_config.layer_quant_list[i].bnb_4bit_quant_type:
            layer_quant_config = BitsAndBytesConfig(
                load_in_4bit = layerwise_config.layer_quant_list[i].load_in_4bit,
                bnb_4bit_quant_type = layerwise_config.layer_quant_list[i].bnb_4bit_quant_type,
                load_in_8bit = layerwise_config.layer_quant_list[i].load_in_8bit,
                bnb_4bit_use_double_quant = layerwise_config.layer_quant_list[i].bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype = dtype
            )
        else:
            layer_quant_config = BitsAndBytesConfig(
                load_in_4bit = layerwise_config.layer_quant_list[i].load_in_4bit,
                load_in_8bit = layerwise_config.layer_quant_list[i].load_in_8bit,
                bnb_4bit_use_double_quant = layerwise_config.layer_quant_list[i].bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype = dtype
            )
        print(layer_quant_config)
        exit()
        # quant_config_list.append(str())
  

    
def quantize_attention_layers(

    ):

    return 0

def quantize_mlp_layers():
    return 0 

if __name__ == '__main__':
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        device_map = "cpu"
    )
    print('meow')