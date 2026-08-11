# config.py

# Tumor board configuration
TUMOR_TYPE = "thoracic"   # later we can set from CLI, env var, or GUI dropdown
NOISE_CODES_PATH = "noise_codes.json"  # path to noise codes file for XML cleaning

# LLM processing configuration
XML_CHUNK_SIZE = 30_000   # max chars per XML chunk for LLM processing (reduced to avoid large event counts)
API_DELAY_SECONDS = 10  # delay between API calls to avoid rate limits

# Model selection configuration
# GPT-5 is the default for all operations (can be overridden via UI or CLI)
DEFAULT_MODEL = "gemma4:31b-it-q8_0"

MODELS = {
    "json_generation": "gemma4:31b-it-q8_0",        # Default for ToA and JSON generation
    "chat": "gemma4:31b-it-q8_0",                   # Default for dashboard chat
    "default": "gemma4:31b-it-q8_0"                 # Default for all operations
}

# TOA-specific model configuration (granular control for different extraction stages)
TOA_MODELS = {
    "chunk_extraction": "gemma4:31b-it-q8_0",     # XML chunk → timeline events (fast, granular extraction)
    "episode_splitting": "gemma4:31b-it-q8_0"       # Events → clinical episodes (needs reasoning, use gpt-5 or gemini-2.5-pro)
}

# Model-specific token limits (output tokens)
# Note: Reasoning models (gpt-5, gemini-2.5-pro) need higher limits due to internal reasoning overhead
# Gemini 2.5 Flash supports up to 65,536 output tokens
MODEL_MAX_TOKENS = {
    "gemma4:31b-it-q8_0": 65536, # 
    "gpt-5": 32768,          # TESTING: High limit for reasoning tokens (o1 supports up to 100k output)
    "gpt-5-mini": 4096,      # Mid-range, good for most tasks
    "gpt-5-nano": 2048,      # Smaller, faster responses
    "gpt-4.1": 8192,         # Traditional model, proven stable
    "gpt-4.1-mini": 8192,    # Fast, cost-effective GPT-4.1 variant (32k max output)
    "gpt-4o": 16384,         # GPT-4o supports 16k output tokens
    "gpt-4o-mini": 16384,    # GPT-4o-mini supports 16k output tokens
    "gemini-2.5-flash": 32768,  # Gemini 2.5 Flash via Vertex AI (supports up to 65k output)
    "gemini-2.5-pro": 32768,  # Gemini 2.5 Pro via Vertex AI (supports up to 65k output)
    "gemini-2.5-flash-lite": 32768,  # Gemini 2.5 Flash Lite via Vertex AI (supports up to 65k output)
    "gemini-3-pro-preview": 32768,  # Gemini 3 Pro - high output capacity
    "gemini-2.0-flash": 8192,  # Gemini 2.0 Flash (1M context, 8k output)
    "gemini-1.5-pro": 8192,  # Large context window
}

# Model-specific context window sizes (input tokens)
MODEL_CONTEXT_LIMITS = {
    "gemma4:31b-it-q8_0": 256000, # 256k token context
    "gpt-5": 200000,         # 200k token context    
    "gpt-5-mini": 128000,    # 128k token context
    "gpt-5-nano": 128000,    # 128k token context
    "gpt-4.1": 128000,       # 128k token context
    "gpt-4.1-mini": 128000,  # 128k token context
    "gpt-4o": 128000,        # 128k token context
    "gpt-4o-mini": 128000,   # 128k token context
    "gemini-2.5-flash": 1048576,  # 1M token context
    "gemini-2.5-pro": 1048576,  # 1M token context
    "gemini-2.5-flash-lite": 1048576,  # 1M token context
    "gemini-3-pro-preview": 1000000,  # 1M token context (estimated)
    "gemini-2.0-flash": 1048576,  # 1M token context
    "gemini-1.5-pro": 2000000,  # 2M token context
}

# Available models (for future UI dropdown)
AVAILABLE_MODELS = list(MODEL_MAX_TOKENS.keys())

# Default max tokens (for backward compatibility)
MAX_TOKENS = 8192

# Token budget for chat context (Rate limit: 150k tokens/min)
# Target: ~30-35k tokens input + 8k output = ~40k per request → ~3-4 queries/min
# Context breakdown per chat query:
#   - System prompt: ~800 tokens
#   - Chat history (last 8 msgs): ~2-3k tokens
#   - TOA episodes: ~2-3k tokens
#   - Full TOA timeline: ~10-15k tokens (priority - NOT limited)
#   - Patient JSONs: ~5k tokens
#   - Recent EHR (10k chars): ~3-4k tokens
#   - User question: ~500 tokens
# Total estimated: ~25-35k tokens input
