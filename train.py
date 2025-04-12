import os
import torch
import argparse
import json
import logging
from datetime import datetime
from typing import List, Dict
from transformers import (
    set_seed,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from trl import SFTTrainer
from torch.utils.data import Dataset

# Import the SciqData processing functions
from SciqData import (
    download_sciq_dataset,
    process_sciq_dataset,
    extract_topics_and_subtopics,
    create_training_files
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log")
    ]
)
logger = logging.getLogger(__name__)

# Constants
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
LORA_R = 16  # LoRA attention dimension
LORA_ALPHA = 32  # Alpha parameter for LoRA scaling
LORA_DROPOUT = 0.05  # Dropout probability for LoRA layers
TARGET_MODULES = [  # Modules to apply LoRA to
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]


class SciQuizDataset(Dataset):
    """Dataset class for science quiz training examples"""

    def __init__(self, data_path, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Tokenize the full text (prompt + completion)
        full_text = example["prompt"] + "\n" + example["completion"]

        full_text_encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize just the prompt to find where the completion starts
        prompt_encoding = self.tokenizer(
            example["prompt"] + "\n",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Create labels: -100 for prompt tokens (ignored in loss),
        # actual token IDs for completion
        labels = full_text_encoding.input_ids.clone()

        # Find the length of the prompt in tokens
        prompt_len = 0
        for i in range(min(len(prompt_encoding.input_ids[0]), len(labels[0]))):
            if prompt_encoding.input_ids[0][i] != labels[0][i]:
                break
            prompt_len += 1

        # Set labels for prompt part to -100 (ignored in loss calculation)
        labels[0, :prompt_len] = -100

        return {
            "input_ids": full_text_encoding.input_ids[0],
            "attention_mask": full_text_encoding.attention_mask[0],
            "labels": labels[0]
        }


def setup_model_and_tokenizer(
        model_name=MODEL_NAME,
        load_in_8bit=False,
        load_in_4bit=True,
        use_flash_attn=True,
        device="auto"
):
    """
    Loads the base model and tokenizer with optimized settings for fine-tuning
    """
    logger.info(f"Loading base model: {model_name}")

    # Set device map
    if device == "auto":
        device_map = "auto"
    else:
        device_map = {"": device}

    # Set up quantization config for reduced memory usage
    compute_dtype = torch.float16

    # Configure quantization
    if load_in_4bit:
        logger.info("Loading model in 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif load_in_8bit:
        logger.info("Loading model in 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    # Load model with quantization config
    model_kwargs = {
        "device_map": device_map,
        "quantization_config": quantization_config,
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2" if use_flash_attn else "eager"
    }

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=True,
    )

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

    return model, tokenizer


def setup_lora(model, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
               lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES):
    """
    Applies LoRA adapters to the model for parameter-efficient fine-tuning
    """
    logger.info("Setting up LoRA for efficient fine-tuning")

    # Prepare model for kbit training if using quantization
    model = prepare_model_for_kbit_training(model)

    # Define LoRA Config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA adapters
    model = get_peft_model(model, peft_config)

    # Print trainable vs frozen parameters
    model.print_trainable_parameters()

    return model


def create_trainer(model, tokenizer, train_dataset, eval_dataset=None, output_dir="./results",
                   num_epochs=3, batch_size=4, gradient_accumulation_steps=2):
    """
    Creates a SFTTrainer for model fine-tuning
    """
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        logging_steps=10,
        logging_dir=f"{output_dir}/logs",
        fp16=True,
        weight_decay=0.01,
        save_total_limit=2,
        report_to="tensorboard",
        push_to_hub=False
    )

    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=tokenizer.model_max_length,
        packing=False
    )

    return trainer


def prepare_ollama_model(output_dir, model_name="mistral-sciquiz"):
    """
    Prepares Ollama Modelfile for easy deployment
    """
    system_prompt = """You are SciQuizPro, an educational AI assistant specialized in creating science quizzes for students. 
You were trained on the SciQ dataset to generate accurate and educationally appropriate multiple-choice science questions. 
You can create quizzes for different grade levels (elementary school, middle school, high school, college) 
across various science categories (Earth Science, Biology, Chemistry, Physics, Space Science).

When asked to create a quiz:
1. Generate factually accurate science questions with exactly 4 multiple-choice options
2. Format your response as a valid JSON array of question objects
3. Each question should include the question text, the options, and the correct answer
4. Ensure content is appropriate for the requested grade level
5. Provide clear, educational content that helps students learn science concepts

Always respond with properly structured, valid JSON when generating quizzes."""

    modelfile_content = f"""
FROM {MODEL_NAME}
PARAMETER temperature 0.2
PARAMETER num_predict 4000
SYSTEM {system_prompt}
"""

    # Write Modelfile
    modelfile_path = os.path.join(output_dir, "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    logger.info(f"Modelfile created at {modelfile_path}")
    logger.info(f"To create the Ollama model, run: ollama create {model_name} -f {modelfile_path}")

    # Create instructions for importing the model to Ollama
    instructions = f"""
# Instructions for importing fine-tuned model to Ollama

1. Transfer all model files from '{output_dir}' to your Ollama-compatible environment.

2. Create the Ollama model using:
   ollama create {model_name} -f {modelfile_path}

3. Test the model with the web application by updating the model name in app.py.
"""

    # Write instructions
    instructions_path = os.path.join(output_dir, "ollama_instructions.txt")
    with open(instructions_path, "w") as f:
        f.write(instructions)

    logger.info(f"Deployment instructions saved to {instructions_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Fine-tune Mistral-7B for science quiz generation")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1",
                        help="Base model to fine-tune")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="Load model in 8-bit mode")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                        help="Load model in 4-bit mode")
    parser.add_argument("--use_flash_attn", action="store_true", default=True,
                        help="Use Flash Attention for faster training")

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout probability")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    # Data arguments
    parser.add_argument("--data_cache_dir", type=str, default="./data_cache",
                        help="Directory to cache the dataset")
    parser.add_argument("--training_file", type=str, default="training_examples.jsonl",
                        help="Path to the processed training file")
    parser.add_argument("--skip_data_processing", action="store_true",
                        help="Skip data processing if already done")

    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./sciquiz_model",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--ollama_model_name", type=str, default="mistral-sciquiz",
                        help="Name for the Ollama model")

    # CPU-Only mode (for hardware limitations mentioned in project)
    parser.add_argument("--cpu_only", action="store_true",
                        help="Force using CPU only (for systems without compatible GPUs)")

    args = parser.parse_args()
    return args


def prepare_data_splits(training_file, tokenizer, val_ratio=0.1, seed=42):
    """
    Prepare training and validation splits from a JSONL file
    """
    logger.info(f"Loading training data from {training_file}")

    # Load all examples
    examples = []
    with open(training_file, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))

    # Set seed for reproducibility
    set_seed(seed)

    # Shuffle and split
    import random
    random.shuffle(examples)

    val_size = int(len(examples) * val_ratio)
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]

    # Write split files
    train_file = training_file.replace(".jsonl", "_train.jsonl")
    val_file = training_file.replace(".jsonl", "_val.jsonl")

    with open(train_file, 'w', encoding='utf-8') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(val_file, 'w', encoding='utf-8') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    # Create datasets
    train_dataset = SciQuizDataset(train_file, tokenizer)
    val_dataset = SciQuizDataset(val_file, tokenizer)

    logger.info(f"Created {len(train_dataset)} training examples and {len(val_dataset)} validation examples")

    return train_dataset, val_dataset


def main():
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Log initialization
    logger.info("Starting Mistral-7B fine-tuning for science quiz generation")
    logger.info(f"Output directory: {output_dir}")

    # Process SciQ dataset if needed
    if not args.skip_data_processing and not os.path.exists(args.training_file):
        logger.info("Processing SciQ dataset...")
        data_folder = download_sciq_dataset()
        processed_data, difficulties = process_sciq_dataset(data_folder)
        topic_labeled_data = extract_topics_and_subtopics(processed_data)
        create_training_files(topic_labeled_data)
        logger.info(f"Data processing complete. Training file created at {args.training_file}")

    # Set device configuration based on args
    device = "cpu" if args.cpu_only else "auto"
    if args.cpu_only:
        logger.warning("Running in CPU-only mode. Training will be very slow!")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        # Setup model and tokenizer
        logger.info("Setting up model and tokenizer")
        model, tokenizer = setup_model_and_tokenizer(
            model_name=args.model_name,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            use_flash_attn=args.use_flash_attn,
            device=device
        )

        # Prepare data
        logger.info("Preparing training and validation datasets")
        train_dataset, val_dataset = prepare_data_splits(
            args.training_file,
            tokenizer,
            seed=args.seed
        )

        # Setup LoRA
        logger.info("Applying LoRA adapters")
        model = setup_lora(
            model,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )

        # Create trainer
        logger.info("Setting up training configuration")
        trainer = create_trainer(
            model,
            tokenizer,
            train_dataset,
            val_dataset,
            output_dir=output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )

        # Train the model
        logger.info("Starting training process")
        trainer.train()

        # Save the trained model
        logger.info("Saving fine-tuned model")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Prepare Ollama configuration
        logger.info("Creating Ollama configuration")
        prepare_ollama_model(output_dir, model_name=args.ollama_model_name)

        logger.info(f"Training completed! Model saved to {output_dir}")
        logger.info(
            f"Use 'ollama create {args.ollama_model_name} -f {output_dir}/Modelfile' to create your Ollama model")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()