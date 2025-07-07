# LLM Task Configuration
import os
from dotenv import load_dotenv
load_dotenv()

# Data settings
BASE_DIR = "CrysMTM"
MODALITIES = ["text", "image"]
MAX_ROTATIONS = None

# Temperature settings
TRAIN_RANGE = range(0, 851, 50)
TRAIN_TEMPS = [T for T in TRAIN_RANGE if T not in [250, 450, 650, 750, 800]]
ID_TEMPS = [250, 450, 650, 750, 800]
OOD_TEMPS = [0, 50, 100, 900, 950, 1000]
ALL_TEMPS = set(range(0, 1001, 50))
FEW_SHOT_EXAMPLES = 3  # Number of random training examples to include
ID_SAMPLES_PER_TEMP = 3
OOD_SAMPLES_PER_TEMP = 3

# LLM API settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TEMPERATURE = 0.2
MAX_TOKENS = 1000

# Model configurations
MULTIMODAL_MODELS = {
    "mistralai/mistral-medium-3",
    "x-ai/grok-2-vision-1212",
    "meta-llama/llama-4-maverick",
    "openai/gpt-4.1-mini",
    "google/gemini-2.5-flash-preview-05-20",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
}

MODELS_TO_TEST = [
    {'name': "deepseek/deepseek-chat", 'use_openrouter': True},
    {'name': "x-ai/grok-2-vision-1212", 'use_openrouter': True},
    {'name': "x-ai/grok-2-1212", 'use_openrouter': True},
    {'name': "meta-llama/llama-4-maverick", 'use_openrouter': True},
    {'name': "mistralai/mistral-medium-3", 'use_openrouter': True},
    {'name': "openai/gpt-4.1-mini", 'use_openrouter': True},
    {'name': "google/gemini-2.5-flash-preview-05-20", 'use_openrouter': True},
    {'name': "anthropic/claude-opus-4", 'use_openrouter': True},
    {'name': "anthropic/claude-sonnet-4", 'use_openrouter': True}
]

# Target properties for prediction
TARGET_PROPERTIES = ["HOMO", "LUMO", "Eg", "Ef", "Et", "Eta", "disp", "vol", "bond"]

# Processing settings
MAX_RETRIES = 3
MAX_WORKERS = 4
TIMEOUT = 60 