from dataclasses import dataclass
from typing import Optional, Iterator, Dict, List
from typing_extensions import Literal
import torch
import numpy as np
import random

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
    load_in_4bit: Optional[bool]
    load_in_8bit: Optional[bool]
    bnb_4bit_use_double_quant: Optional[bool]
    bnb_4bit_quant_storage: Optional[torch.dtype]
    bnb_4bit_compute_dtype: Optional[torch.dtype]
    bnb_4bit_quant_type: Optional[str] = None  
    

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


def load_config(
    configs: Dict,
    config_type: Literal["model", "quant"]
) -> Iterator[QuantConfig]:
    match config_type:
        case "model":
            for config in configs:
                yield ModelConfig(**config)
        case "quant":
            for config in configs:
                yield QuantConfig(**config)

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
    layerwise_config: List[QuantConfig],
    ):
    
    if layerwise_config is None:
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
    quant_config_str_list = []

    for i, layer in enumerate(blocks):
        load_in = "fp32"
        dtype = dtype_map[layerwise_config[i].bnb_4bit_compute_dtype]
        if layerwise_config[i].load_in_4bit:
            load_in = "4bit/" + layerwise_config[i].bnb_4bit_quant_type
            if layerwise_config[i].bnb_4bit_use_double_quant:
                load_in += "/double"
            else:
                load_in += "/single" 

        elif layerwise_config[i].load_in_8bit:
            load_in = "8bit"

        if layerwise_config[i].bnb_4bit_quant_type:
            layer_quant_config = BitsAndBytesConfig(
                load_in_4bit = layerwise_config[i].load_in_4bit,
                bnb_4bit_quant_type = layerwise_config[i].bnb_4bit_quant_type,
                load_in_8bit = layerwise_config[i].load_in_8bit,
                bnb_4bit_use_double_quant = layerwise_config[i].bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype = dtype
            )
        else:
            layer_quant_config = BitsAndBytesConfig(
                load_in_4bit = layerwise_config[i].load_in_4bit,
                load_in_8bit = layerwise_config[i].load_in_8bit,
                bnb_4bit_use_double_quant = layerwise_config[i].bnb_4bit_use_double_quant,
                bnb_4bit_compute_dtype = dtype
            )
        quant_config_list.append(layer_quant_config)
        quant_config_str_list.append(str(layer_quant_config))

    unique_configs = set(quant_config_str_list)
    
    for config in unique_configs:
        temp_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_config.model_name,
            quantization_config = quant_config_list[quant_config_str_list.index(config)],
            device_map = model.device
        )
        try:
            decoders_1 = eval(f"model.{DecoderMap[model_config.model_family]}")
            decoders_2 = eval(f"temp_model.{DecoderMap[model_config.model_family]}")
        except AttributeError as e:
            raise ValueError(f"Unsupported model family '{model_config.model_family}' or invalid layer attribute: {e}")

        layer_change_positions = [i for i, j in enumerate(quant_config_str_list) if j == config]

        

        

    return 0
      

    
def quantize_attention_layers(

    ):

    return 0

def quantize_mlp_layers():
    return 0 

if __name__ == '__main__':

    ## example

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-1B",
        # quantization_config = 
        # device_map = "cuda:0"
        device_map = "cpu"
    )

    model_config_list = [
    {
        "model_name": "meta-llama/Llama-3.2-1B",
        "model_family": "llama",
    }
    ]

    decoder_map = generate_decoder_map()

    quant_config_list = [
        {
        "load_in_4bit": False,
        "load_in_8bit": False,
        "bnb_4bit_use_double_quant": False,
        "bnb_4bit_quant_storage": "float16",
        "bnb_4bit_compute_dtype": "float16"
    }
    ]*len(eval(f"model.{decoder_map[model_config_list[0]['model_family']]}"))
    
    layerwise_quant_configs = load_config(quant_config_list, config_type = "quant")

    layerwise_model_quant_config = [] 

    for qconfig in layerwise_quant_configs:
        layerwise_model_quant_config.append(qconfig)

    model_configs = load_config(configs = model_config_list, config_type = "model")

    for model_config in model_configs:
        updated_model = quantize_transformer_blocks(model = model, model_config = model_config, layerwise_config = layerwise_model_quant_config)

    # print('meow')