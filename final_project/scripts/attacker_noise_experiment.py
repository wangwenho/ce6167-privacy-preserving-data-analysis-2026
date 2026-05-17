import argparse
import json
import os
import sys

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# Override CUDA device before any project imports
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attacker_evaluation_gpt import eval_on_batch
from data_process import get_sent_list

MODEL_CARDS = {
    "mpnet": "all-mpnet-base-v1",
    "sent_roberta": "all-roberta-large-v1",
    "sent_t5_large": "sentence-t5-large",
    "sent_t5_base": "sentence-t5-base",
    "simcse_bert": "princeton-nlp/sup-simcse-bert-large-uncased",
    "simcse_roberta": "princeton-nlp/sup-simcse-roberta-large",
}


class LinearProjection(nn.Module):
    """Projection layer: 768 -> 1280 (matches GPT-2 token embedding dim)"""

    def __init__(self, in_num=768, out_num=1280):
        super().__init__()
        self.fc1 = nn.Linear(in_num, out_num)

    def forward(self, x):
        return self.fc1(x)


class PersonaChatDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate(self, unpacked_data):
        return unpacked_data


def run_test_with_noise(config, noise_scale):
    device = torch.device("cuda:0")
    batch_size = config["batch_size"]

    # Data
    sent_list = get_sent_list(config)
    dataset = PersonaChatDataset(sent_list)
    dataloader = DataLoader(
        dataset, shuffle=False, batch_size=batch_size, collate_fn=dataset.collate
    )
    print(f"Data: {len(sent_list)} sentences, {len(dataloader)} batches")

    # Victim embedder
    embedder = SentenceTransformer(config["embed_model_path"], device="cuda:0")
    embedder.eval()

    # Projection
    proj_path = (
        f"models/projection_gpt2_large_{config['dataset']}_{config['embed_model']}"
    )
    projection = LinearProjection(in_num=768, out_num=1280)
    projection.load_state_dict(torch.load(proj_path, map_location=device))
    projection.to(device)
    projection.eval()
    print("Projection loaded")

    # Attacker
    attacker_path = (
        f"models/attacker_gpt2_large_{config['dataset']}_{config['embed_model']}"
    )
    config["model"] = AutoModelForCausalLM.from_pretrained(attacker_path).to(device)
    config["tokenizer"] = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    config["model"].eval()
    print("Attacker loaded")

    # Output path
    save_path = (
        f"models/attacker_gpt2_large_{config['dataset']}"
        f"_{config['embed_model']}_beam_noise_{noise_scale}.log"
    )
    print(f"Output: {save_path}")

    # Evaluation loop
    sent_dict = {"gt": [], "pred": []}
    with torch.no_grad():
        for idx, batch_text in enumerate(dataloader):
            # Victim encoding
            embeddings = embedder.encode(batch_text, convert_to_tensor=True).to(device)
            # Projection
            embeddings = projection(embeddings)
            # Noise injection (after projection, before decoding)
            if noise_scale > 0:
                embeddings = embeddings + torch.randn_like(embeddings) * noise_scale
            # Decode
            pred_list, gt_list = eval_on_batch(
                batch_X=embeddings,
                batch_D=batch_text,
                model=config["model"],
                tokenizer=config["tokenizer"],
                device=device,
                config=config,
            )
            sent_dict["pred"].extend(pred_list)
            sent_dict["gt"].extend(gt_list)
            print(f"  Batch {idx} done ({idx * batch_size} samples)")

    with open(save_path, "w") as f:
        json.dump(sent_dict, f, indent=4)
    print(f"Done → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GEIA evaluation with noise")
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.0,
        help="Gaussian noise std dev added to embeddings",
    )
    parser.add_argument("--dataset", type=str, default="personachat")
    parser.add_argument(
        "--embed_model", type=str, default="mpnet", choices=list(MODEL_CARDS.keys())
    )
    parser.add_argument(
        "--decode", type=str, default="beam", choices=["beam", "sampling"]
    )
    parser.add_argument("--model_dir", type=str, default="microsoft/DialoGPT-large")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    config = {
        "model_dir": args.model_dir,
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "data_type": "test",
        "embed_model": args.embed_model,
        "decode": args.decode,
        "embed_model_path": MODEL_CARDS[args.embed_model],
        "device": torch.device("cuda"),
        "eos_token": "<|endoftext|>",
        "use_opt": False,
    }

    run_test_with_noise(config, noise_scale=args.noise_scale)
