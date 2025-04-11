"""
Configuration settings for the Ollama Mistral 7B model fine-tuned on the SciQ dataset.
"""

# Ollama API configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"  # Local Ollama endpoint

# Model configuration
MODEL_NAME = "mistral-7b-sciq-finetuned"  # Name of our fine-tuned model in Ollama

# Generation parameters
TEMPERATURE = 0.2      # Lower values make output more deterministic (0.0-1.0)
TOP_P = 0.95           # Nucleus sampling - only consider tokens with this cumulative probability
MAX_TOKENS = 4000      # Maximum length of generated response
STOP_SEQUENCES = ["</s>", "user:", "User:"]  # Sequences that stop generation

# Fine-tuning details (for documentation)
FINETUNING_INFO = {
    "base_model": "Mistral 7B",
    "dataset": "SciQ - 13,679 crowdsourced science exam questions",
    "training_epochs": 3,
    "training_method": "LORA fine-tuning with 8-bit quantization",
    "specialization": "Science education content for K-12 and college levels"
}

# Science categories
SCIENCE_CATEGORIES = {
    "earth": ["Geology", "Weather", "Climate", "Natural Disasters", "Earth's Structure", "Ecosystems"],
    "biology": ["Human Body", "Plants", "Animals", "Cells", "Genetics", "Ecosystems"],
    "chemistry": ["Elements", "Chemical Reactions", "States of Matter", "Periodic Table", "Acids and Bases"],
    "physics": ["Forces", "Motion", "Energy", "Electricity", "Magnetism", "Light and Sound"],
    "space": ["Solar System", "Stars", "Planets", "Space Exploration", "Galaxies"]
}