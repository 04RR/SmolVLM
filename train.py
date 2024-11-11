import os
import wandb
import torch
import warnings
import pandas as pd
from PIL import Image
from tqdm import tqdm
from smol import SmolVLM, SAM2Transforms
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")


# Dataset Class
class SmolVLMDataset(Dataset):
    def __init__(self, csv_file, transforms, tokenizer, max_length=1024):
        self.data = pd.read_csv(csv_file)
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_path = self.data.loc[idx, "image"]
            img_path = f"data/pretrain/images/{img_path}"
            input_string = self.data.loc[idx, "input_string"]
            output_string = self.data.loc[idx, "output_string"]

            image = Image.open(img_path).convert("RGB")
            image = self.transforms(image)

            output_tokens = self.tokenizer(
                output_string,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {
                "image": image,
                "input_string": input_string,
                "labels": output_tokens["input_ids"].squeeze(),
            }
        except:
            random_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(random_idx)


# Collate Function
def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    input_strings = [item["input_string"] for item in batch]
    labels = torch.stack([item["labels"] for item in batch])
    return {"images": images, "input_strings": input_strings, "labels": labels}


# Data Loader Creation
def create_data_loaders(
    csv_path, tokenizer, batch_size=16, shuffle=True, num_workers=8
):
    transforms = SAM2Transforms(resolution=1024, mask_threshold=0.5)
    dataset = SmolVLMDataset(csv_path, transforms=transforms, tokenizer=tokenizer)
    print(f"Dataset length: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


# Model Evaluation
def evaluate_model(model, dataloader, device="cuda"):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["images"].to(device)
            input_strings = batch["input_strings"]
            labels = batch["labels"].to(device)

            _, loss = model.generate(
                prompts=input_strings, image_tensors=images, labels=labels
            )
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# Model Training with WandB Logging, Gradient Accumulation, and Saving Checkpoints
def train_model(
    model,
    train_dataloader,
    val_dataloader,
    epochs=5,
    lr=1e-4,
    device="cuda",
    run_name=None,
    accumulation_steps=1,
    checkpoint_dir="checkpoints",
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        list(model.image_encoder.parameters())
        + list(model.mlp_projector.parameters())
        + list(model.language_model.parameters()),
        lr=lr,
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 1 - epoch / epochs
    )

    wandb.init(
        project="SmolVLM-Training",
        config={"epochs": epochs, "learning_rate": lr},
        name=run_name,
    )
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        epoch_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_ckpt_path, exist_ok=True)
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        total_train_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            images = batch["images"].to(device)
            input_strings = batch["input_strings"]
            labels = batch["labels"].to(device)

            _, loss = model.generate(
                prompts=input_strings, image_tensors=images, labels=labels
            )
            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Training Loss: {avg_train_loss:.4f}")

        avg_val_loss = evaluate_model(model, val_dataloader, device)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        current_lr = scheduler.get_last_lr()[0]
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
            }
        )

        scheduler.step()

        torch.save(
            model.image_encoder.state_dict(),
            os.path.join(epoch_ckpt_path, f"image_encoder.pth"),
        )
        torch.save(
            model.mlp_projector.state_dict(),
            os.path.join(epoch_ckpt_path, f"mlp_projector.pth"),
        )
        model.language_model.save_pretrained(
            os.path.join(epoch_ckpt_path, f"language_model")
        )
        model.tokenizer.save_pretrained(
            os.path.join(epoch_ckpt_path, f"language_model")
        )
        print(f"Checkpoints saved for epoch {epoch + 1}")

    wandb.finish()


# Main Script
if __name__ == "__main__":
    model = SmolVLM(
        model_cfg="configs/sam2.1/sam2.1_hiera_base_plus.yaml",
        sam2_full_ckpt_path="/mnt/d/rr/smol/model_ckpts/sam2.1_hiera_base_plus.pt",
        image_encoder_ckpt_path="/mnt/d/rr/smol/model_ckpts/sam2.1_base_plus_image_encoder.pth",
        language_model_ckpt="HuggingFaceTB/SmolLM2-135M",
        device="cuda",
        dtype=torch.float16,
    )
    batch_size = 4
    n = 0
    for m in [model.image_encoder, model.mlp_projector, model.language_model]:
        for param in m.parameters():
            param.requires_grad = True
            n += param.numel()

    print(f"Trainable Parameters: {n}")

    exp_name = "SmolLM2-135M_with_SAM2.1-HieraBasePlus_run1"

    train_dataloader = create_data_loaders(
        "data/pretrain/train.csv", model.tokenizer, batch_size=batch_size
    )
    val_dataloader = create_data_loaders(
        "data/pretrain/val.csv", model.tokenizer, batch_size=batch_size
    )

    train_model(
        model,
        train_dataloader,
        val_dataloader,
        epochs=5,
        lr=1e-4,
        accumulation_steps=4,
        device="cuda",
        run_name=exp_name,
    )
