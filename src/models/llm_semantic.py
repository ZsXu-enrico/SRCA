"""LLM-based Semantic Representation Module using TinyLlama.

This module implements the semantic representation component of SRCA:
1. Loads TinyLlama model for text understanding
2. Applies RPM/FPA prompts for description unification
3. Extracts semantic features from LLM hidden states
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging

from ..utils.prompts import format_rpm_prompt, format_fpa_prompt, format_chat_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMSemanticEncoder(nn.Module):
    """
    LLM-based semantic encoder for service descriptions.

    Uses TinyLlama to:
    - Generate unified descriptions via prompt engineering
    - Extract semantic features from hidden states
    """

    def __init__(
        self,
        model_path: str,
        max_length: int = 512,
        semantic_dim: int = 1024,
        device_map: str = 'auto',
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        freeze_llm: bool = True
    ):
        """
        Initialize LLM semantic encoder.

        Args:
            model_path: Path to LLaMA model (e.g., '../Llama3.1-8B-hf')
            max_length: Maximum sequence length
            semantic_dim: Output semantic feature dimension (Paper uses 1024)
            device_map: Device mapping for model loading
            load_in_8bit: Whether to use 8-bit quantization
            load_in_4bit: Whether to use 4-bit quantization
            freeze_llm: Whether to freeze LLM parameters (recommended)
        """
        super().__init__()

        self.model_path = model_path
        self.max_length = max_length
        self.semantic_dim = semantic_dim
        self.freeze_llm = freeze_llm

        logger.info(f"Loading LLaMA model from: {model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # Set pad token if not exists (LLaMA doesn't have pad token by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with quantization if specified
        load_kwargs = {
            'trust_remote_code': True,
            'torch_dtype': torch.float16,  # Use fp16 for efficiency
        }

        # Only add device_map if specified (let PyTorch Lightning manage otherwise)
        if device_map is not None:
            load_kwargs['device_map'] = device_map

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization")
        elif load_in_8bit:
            load_kwargs['load_in_8bit'] = True
            logger.info("Using 8-bit quantization")

        self.llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs
        )

        # Freeze LLM parameters if specified (recommended for SRCA)
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
            logger.info("LLM parameters frozen (recommended for SRCA)")

        # Get LLM hidden size (LLaMA-3-8B has hidden_size=4096)
        self.llm_hidden_size = self.llm.config.hidden_size
        logger.info(f"LLaMA hidden size: {self.llm_hidden_size}")

        # Dimension mapping layer (Paper Section 4.1.2, Eq. 8)
        # Paper: e^(0)_i = W * e_i + b
        # Maps from LLaMA hidden dim (4096) to semantic dim (1024)
        self.projection = nn.Linear(self.llm_hidden_size, semantic_dim)

        # Match projection dtype to LLM dtype (fp16 if LLM uses fp16)
        if load_kwargs.get('torch_dtype') == torch.float16:
            self.projection = self.projection.half()
            logger.info(f"Projection layer converted to float16")

        logger.info(f"Dimension mapping: {self.llm_hidden_size} -> {semantic_dim}")
        logger.info(f"Output semantic dimension: {semantic_dim}")
        logger.info("✓ LLM Semantic Encoder initialized successfully")

    def encode_text(
        self,
        texts: List[str],
        use_generation: bool = False
    ) -> torch.Tensor:
        """
        Encode text inputs to semantic features.

        CRITICAL FIX: Implements correct two-stage process from SRCA paper Section 4.1.1-4.1.2
        - Stage 1: Generate unified description D' = M(concat(T, D))  [Eq. 3-4]
        - Stage 2: Re-encode D' to extract features e_D' = 1/n Σ h_i  [Eq. 5]

        Args:
            texts: List of text strings (prompts or raw descriptions)
            use_generation: If True, use two-stage generation → re-encoding (paper method)
                          If False, directly extract features from input

        Returns:
            Tensor of shape [batch_size, semantic_dim]
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Move to model device
        device = next(self.llm.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad() if self.freeze_llm else torch.enable_grad():
            if use_generation:
                # ============================================================
                # CRITICAL FIX: Two-stage process as described in paper
                # ============================================================

                # Stage 1: Generate unified description D' (Paper Eq. 3-4)
                logger.info("Stage 1: Generating unified descriptions...")
                generation_outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,  # Deterministic for reproducibility
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

                # generation_outputs.sequences contains the full sequence (prompt + generation)
                # We need to extract only the generated part for re-encoding
                generated_ids = generation_outputs  # [batch, seq_len]

                # Stage 2: Re-encode the generated description to extract features (Paper Eq. 5)
                logger.info("Stage 2: Re-encoding generated descriptions to extract semantic features...")

                # Re-encode the generated text
                re_encode_outputs = self.llm(
                    input_ids=generated_ids,
                    output_hidden_states=True,
                    return_dict=True
                )

                # Extract hidden states from the last layer
                hidden_states = re_encode_outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

                # Average pooling over the sequence (Paper Eq. 5: e_D' = 1/n * Σ h_i)
                # Exclude padding tokens from pooling
                attention_mask = (generated_ids != self.tokenizer.pad_token_id).float().unsqueeze(-1)  # [batch, seq_len, 1]
                masked_hidden = hidden_states * attention_mask  # [batch, seq_len, hidden_size]

                # Mean pooling: sum over sequence and divide by number of non-padding tokens
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)  # [batch, hidden_size]

                logger.info(f"✓ Two-stage semantic extraction completed: {pooled.shape}")

            else:
                # Directly extract features from input encoding (no generation)
                outputs = self.llm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True
                )

                # Extract last layer hidden states
                hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]

                # Average pooling over sequence (excluding padding)
                attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [batch, seq_len, 1]
                masked_hidden = hidden_states * attention_mask
                pooled = masked_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)  # [batch, hidden_size]

        # Project to semantic dimension (Paper Eq. 8: e^(0)_i = W * e_i + b)
        if self.projection is not None:
            # Ensure projection layer is on the same device as the hidden states
            if self.projection.weight.device != pooled.device:
                self.projection = self.projection.to(pooled.device)
            semantic_features = self.projection(pooled)  # [batch, semantic_dim]
        else:
            # No projection needed, dimensions already match
            semantic_features = pooled

        return semantic_features

    def encode_mashups(
        self,
        descriptions: List[str],
        categories_list: List[List[str]],
        names: Optional[List[str]] = None,
        use_generation: bool = True
    ) -> torch.Tensor:
        """
        Encode mashup descriptions using RPM prompt.

        Args:
            descriptions: List of mashup descriptions
            categories_list: List of category lists for each mashup
            names: List of mashup names (optional)
            use_generation: Whether to generate unified descriptions

        Returns:
            Tensor of shape [batch_size, semantic_dim]
        """
        # Use default names if not provided
        if names is None:
            names = ["Unknown"] * len(descriptions)

        # Format RPM prompts
        prompts = [
            format_chat_prompt(format_rpm_prompt(desc, cats, name))
            for desc, cats, name in zip(descriptions, categories_list, names)
        ]

        return self.encode_text(prompts, use_generation=use_generation)

    def encode_apis(
        self,
        names: List[str],
        descriptions: List[str],
        categories_list: List[List[str]],
        use_generation: bool = True
    ) -> torch.Tensor:
        """
        Encode API descriptions using FPA prompt.

        Args:
            names: List of API names
            descriptions: List of API descriptions
            categories_list: List of category lists for each API
            use_generation: Whether to generate feature summaries

        Returns:
            Tensor of shape [batch_size, semantic_dim]
        """
        # Format FPA prompts
        prompts = [
            format_chat_prompt(format_fpa_prompt(name, desc, cats))
            for name, desc, cats in zip(names, descriptions, categories_list)
        ]

        return self.encode_text(prompts, use_generation=use_generation)

    def forward(
        self,
        texts: List[str],
        prompt_type: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass for encoding texts.

        Args:
            texts: List of text strings
            prompt_type: 'rpm' for mashups, 'fpa' for APIs, None for raw text
            **kwargs: Additional arguments for specific prompt types

        Returns:
            Semantic features [batch_size, semantic_dim]
        """
        if prompt_type == 'rpm':
            return self.encode_mashups(
                texts,
                kwargs.get('categories_list', [[] for _ in texts]),
                kwargs.get('names', None),
                kwargs.get('use_generation', True)
            )
        elif prompt_type == 'fpa':
            return self.encode_apis(
                kwargs.get('names', [''] * len(texts)),
                texts,
                kwargs.get('categories_list', [[] for _ in texts]),
                kwargs.get('use_generation', True)
            )
        else:
            return self.encode_text(texts, kwargs.get('use_generation', False))


def test_llm_encoder():
    """Test function for LLM semantic encoder."""
    print("Testing LLM Semantic Encoder...")

    # Initialize encoder
    encoder = LLMSemanticEncoder(
        model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        semantic_dim=768,
        freeze_llm=True
    )

    # Test mashup encoding
    print("\n1. Testing mashup encoding (RPM)...")
    mashup_descs = [
        "A social media aggregator that combines Twitter, Facebook, and Instagram feeds",
        "Real-time weather dashboard using multiple weather API sources"
    ]
    mashup_cats = [
        ["Social", "Media"],
        ["Weather", "Visualization"]
    ]

    mashup_features = encoder.encode_mashups(mashup_descs, mashup_cats, use_generation=False)
    print(f"   Output shape: {mashup_features.shape}")
    print(f"   Feature range: [{mashup_features.min():.4f}, {mashup_features.max():.4f}]")

    # Test API encoding
    print("\n2. Testing API encoding (FPA)...")
    api_names = ["Twitter API", "OpenWeather API"]
    api_descs = [
        "Access tweets, timelines, and user information from Twitter",
        "Current weather data and forecasts for any location worldwide"
    ]
    api_cats = [["Social"], ["Weather"]]

    api_features = encoder.encode_apis(api_names, api_descs, api_cats, use_generation=False)
    print(f"   Output shape: {api_features.shape}")
    print(f"   Feature range: [{api_features.min():.4f}, {api_features.max():.4f}]")

    print("\n✓ LLM Semantic Encoder test passed!")


if __name__ == "__main__":
    test_llm_encoder()
