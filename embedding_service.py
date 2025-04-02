import torch
from models.graphsgpt.modeling_graphsgpt import GraphsGPTForCausalLM
from data.tokenizer import GraphsGPTTokenizer
from utils.operations.operation_tensor import move_tensors_to_device
from utils.operations.operation_list import split_list_with_yield
from typing import List, Optional
import numpy as np
from tqdm import tqdm

class State:
    model: Optional[GraphsGPTForCausalLM] = None
    tokenizer: Optional[GraphsGPTTokenizer] = None

    model_name: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[str] = None

    initialized: bool = False


def setup(model_name: str, device: str, batch_size: int) -> None:
    State.model = GraphsGPTForCausalLM.from_pretrained(model_name)
    State.model.to(device)
    State.model.eval()

    State.tokenizer = GraphsGPTTokenizer.from_pretrained(model_name)
    
    State.model_name = model_name
    State.device = device
    State.batch_size = batch_size
    
    State.initialized = True

def encode(smiles: List[str]) -> np.ndarray:
    if not State.initialized:
        raise RuntimeError("Service is not setup, call 'setup' before 'encode'.")
    outputs = []
    with torch.no_grad():
        for batched_smiles in tqdm(split_list_with_yield(smiles, State.batch_size), f"Encoding with {State.model_name}"):
            inputs = State.tokenizer.batch_encode(batched_smiles, return_tensors="pt")
            move_tensors_to_device(inputs, State.device)

            fingerprint_tokens = State.model.encode_to_fingerprints(**inputs)  # (batch_size, num_fingerprints, hidden_dim)
            outputs.append(fingerprint_tokens.mean(dim=-2))
    outputs = torch.concat(outputs)
    outputs = outputs.cpu().numpy()
    return outputs
