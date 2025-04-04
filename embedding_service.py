import torch
from models.graphsgpt.modeling_graphsgpt import GraphsGPTForCausalLM
from data.tokenizer import GraphsGPTTokenizer
from utils.operations.operation_tensor import move_tensors_to_device
from typing import List, Optional
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Union
from rdkit import Chem

HF_MODEL_NAMES = {
    "GraphsGPT-1W": "DaizeDong/GraphsGPT-1W",
    "GraphsGPT-2W": "DaizeDong/GraphsGPT-2W",
    "GraphsGPT-4W": "DaizeDong/GraphsGPT-4W",
    "GraphsGPT-8W": "DaizeDong/GraphsGPT-8W",}

def batch_encode(tokenizer: GraphsGPTTokenizer, smiles_or_mol_list: List[Union[str, Chem.Mol]]) -> Union[Dict[str, torch.LongTensor], Dict[str, List[List]]]:
    batched_tokens = []
    for smiles_or_mol in smiles_or_mol_list:
        result = tokenizer.encode(smiles_or_mol, return_tensors="pt")
        if result is None:
            result = {
                "input_ids": torch.tensor([], dtype=torch.long),
                "graph_position_ids_1": torch.tensor([], dtype=torch.long),
                "graph_position_ids_2": torch.tensor([], dtype=torch.long),
                "identifier_ids": torch.tensor([], dtype=torch.bool),}
        batched_tokens.append(result)
        
    batched_tokens = tokenizer._pad_encoded_tensor(batched_tokens)
    return batched_tokens


class State:
    model: Optional[GraphsGPTForCausalLM] = None
    tokenizer: Optional[GraphsGPTTokenizer] = None

    model_name: Optional[str] = None
    device: Optional[str] = None
    batch_size: Optional[str] = None

    initialized: bool = False


def setup(model_name: str, device: str, batch_size: int) -> None:
    State.model = GraphsGPTForCausalLM.from_pretrained(HF_MODEL_NAMES[model_name])
    State.model.to(device)
    State.model.eval()

    State.tokenizer = GraphsGPTTokenizer.from_pretrained(HF_MODEL_NAMES[model_name])
    
    State.model_name = model_name
    State.device = device
    State.batch_size = batch_size
    
    State.initialized = True

def encode(smiles: List[str]) -> np.ndarray:
    if not State.initialized:
        raise RuntimeError("Service is not setup, call 'setup' before 'encode'.")
    outputs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles), State.batch_size), f"Encoding with {State.model_name}"):
            smiles_batch = smiles[i:i+State.batch_size]
            inputs = batch_encode(State.tokenizer, smiles_batch)
            move_tensors_to_device(inputs, State.device)

            fingerprint_tokens = State.model.encode_to_fingerprints(**inputs)  # (batch_size, num_fingerprints, hidden_dim)
            outputs.append(fingerprint_tokens.mean(dim=-2))
    outputs = torch.concat(outputs)
    outputs = outputs.cpu().numpy()
    return outputs
