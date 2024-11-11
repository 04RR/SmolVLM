import os
import time
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision.transforms import Normalize, Resize

warnings.filterwarnings("ignore")


class SAM2Transforms(nn.Module):
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = nn.Sequential(
            Resize((self.resolution, self.resolution)),
            Normalize(self.mean, self.std),
        )

    def forward_batch(self, img_list):
        img_batch = [self.transforms(self.to_tensor(img)) for img in img_list]
        img_batch = torch.stack(img_batch, dim=0)
        return img_batch


class SmolVLM(nn.Module):
    def __init__(
        self,
        model_cfg="configs/sam2.1/sam2.1_hiera_base_plus.yaml",
        sam2_full_ckpt_path="/mnt/d/rr/smol/model_ckpts/sam2.1_hiera_base_plus.pt",
        image_encoder_ckpt_path="/mnt/d/rr/smol/model_ckpts/sam2.1_base_plus_image_encoder.pth",
        language_model_ckpt="HuggingFaceTB/SmolLM2-135M",
        device="cuda",
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.image_encoder = self.load_image_encoder(
            model_cfg, sam2_full_ckpt_path, image_encoder_ckpt_path
        )
        self.mlp_projector = torch.nn.Sequential(
            torch.nn.Linear(4096, 1152),
            torch.nn.GELU(),
            torch.nn.Linear(1152, 576),
            torch.nn.GELU(),
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(language_model_ckpt)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_ckpt, device_map=self.device, torch_dtype=self.dtype
        )

    def get_tokenizer(self):
        return self.tokenizer

    def load_from_checkpoint(self, epoch_ckpt_path):

        image_encoder_path = os.path.join(epoch_ckpt_path, "image_encoder.pth")
        if os.path.exists(image_encoder_path):
            self.image_encoder.load_state_dict(
                torch.load(image_encoder_path, map_location=self.device)
            )
            print(f"Loaded image encoder from {image_encoder_path}")
        else:
            print(f"Image encoder checkpoint not found at {image_encoder_path}")

        mlp_projector_path = os.path.join(epoch_ckpt_path, "mlp_projector.pth")
        if os.path.exists(mlp_projector_path):
            self.mlp_projector.load_state_dict(
                torch.load(mlp_projector_path, map_location=self.device)
            )
            print(f"Loaded MLP projector from {mlp_projector_path}")
        else:
            print(f"MLP projector checkpoint not found at {mlp_projector_path}")

        language_model_path = os.path.join(epoch_ckpt_path, "language_model")
        if os.path.exists(language_model_path):
            self.language_model = AutoModelForCausalLM.from_pretrained(
                language_model_path, device_map=self.device, torch_dtype=self.dtype
            )
            self.tokenizer = AutoTokenizer.from_pretrained(language_model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Loaded language model and tokenizer from {language_model_path}")
        else:
            print(f"Language model checkpoint not found at {language_model_path}")

    def load_image_encoder(
        self, model_cfg, sam2_full_ckpt_path, image_encoder_ckpt_path
    ):
        sam2 = build_sam2(
            model_cfg,
            sam2_full_ckpt_path,
            device=self.device,
            apply_postprocessing=False,
        )
        image_encoder = sam2.image_encoder

        state_dict = torch.load(image_encoder_ckpt_path, map_location=self.device)
        image_encoder.load_state_dict(state_dict)
        image_encoder = image_encoder.to(self.device)

        return image_encoder.to(self.device)

    def preprocess_images(self, images):
        if (
            images.dim() != 4
            or images.size(1) != 3
            or images.size(2) != 1024
            or images.size(3) != 1024
        ):
            raise ValueError(
                "Images should be a tensor of shape (batch_size, 3, 1024, 1024)"
            )

        images = images.to(self.device)
        return images

    def generate(
        self, prompts, image_tensors=None, max_length=1024, temperature=0.0, labels=None
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        st = time.time()

        inputs = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # print(f"Tokenizer time: {time.time() - st}")
        st = time.time()

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"].to(self.device)

        input_embeds = self.language_model.model.embed_tokens(input_ids).to(self.dtype)

        # print(f"Embedding time: {time.time() - st}")
        st = time.time()

        if image_tensors is not None:
            assert len(image_tensors) == len(
                prompts
            ), "Number of images and prompts should match"

            images = self.preprocess_images(image_tensors)
            batch_size = images.size(0)

            # print(f"Preprocess time: {time.time() - st}")
            st = time.time()

            image_features = self.image_encoder(images)["vision_features"]
            image_features = image_features.view(batch_size, image_features.size(1), -1)
            image_features = self.mlp_projector(image_features).to(self.dtype)

            # print(f"Image encoder time: {time.time() - st}")
            st = time.time()

            input_embeds = torch.cat([image_features, input_embeds], dim=1)
            image_seq_len = image_features.size(1)

            image_attention_mask = torch.ones(
                (batch_size, image_seq_len), dtype=attn_mask.dtype
            ).to(self.device)

            attn_mask = torch.cat([image_attention_mask, attn_mask], dim=1)

            # print(f"Concat time: {time.time() - st}")
            st = time.time()

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn_mask,
            temperature=temperature,
            do_sample=True if temperature > 0.0 else False,
            max_new_tokens=max_length,
        )

        # print(f"Generate time: {time.time() - st}")
        st = time.time()

        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        loss = None
        if labels is not None:
            if isinstance(labels[0], str):
                labels_input = self.tokenizer(
                    labels,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                    return_attention_mask=True,
                ).to(self.device)
                labels = labels_input["input_ids"]

            logits = self.language_model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=labels,
            ).logits

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )

        # print(f"Loss time: {time.time() - st}")

        return output_texts, loss


class SAM2Transforms:
    def __init__(
        self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0
    ):
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = Compose(
            [
                Resize((self.resolution, self.resolution)),
                ToTensor(),
                Normalize(self.mean, self.std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


# model = SmolVLM(
#     model_cfg="configs/sam2.1/sam2.1_hiera_base_plus.yaml",
#     sam2_full_ckpt_path="/mnt/d/rr/smol/model_ckpts/sam2.1_hiera_base_plus.pt",
#     image_encoder_ckpt_path="/mnt/d/rr/smol/model_ckpts/sam2.1_base_plus_image_encoder.pth",
#     language_model_ckpt="HuggingFaceTB/SmolLM2-135M",
#     device="cuda",
# )
