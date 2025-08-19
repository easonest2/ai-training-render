"""
Configuration file for MicroGPT Pro.
Modify these settings to customize your AI assistant.
"""

# Model Configuration
MODEL_CONFIG = {
    'vocab_size': 256,      # Vocabulary size (256 for byte-level tokenizer)
    'block_size': 1024,     # Maximum sequence length
    'n_layer': 6,           # Number of transformer layers
    'n_head': 8,            # Number of attention heads
    'n_embd': 384,          # Embedding dimension
    'dropout': 0.1,         # Dropout rate
    'bias': True            # Use bias in linear layers
}

# Generation Configuration
GENERATION_CONFIG = {
    'max_tokens': 100,      # Maximum tokens to generate
    'temperature': 0.8,     # Sampling temperature (0.1-2.0)
    'top_k': 50             # Top-k sampling parameter
}

# Server Configuration
SERVER_CONFIG = {
    'host': '127.0.0.1',    # Server host
    'port': 5000,           # Server port
    'debug': False,          # Debug mode (set to True for development)
    'memory_file': 'memory.json'  # Chat history file
}

# Model Path
MODEL_PATH = 'artifacts/model_final.pt'

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',        # Log level (DEBUG, INFO, WARNING, ERROR)
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}

# Memory Configuration
MEMORY_CONFIG = {
    'max_conversations': 1000,  # Maximum number of conversations to keep
    'auto_cleanup': True,       # Automatically clean old conversations
    'cleanup_threshold': 800    # Start cleanup when reaching this number
}
