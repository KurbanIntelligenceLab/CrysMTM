import base64
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from dataloaders.llm_regression_dataloader import LLMLoader

load_dotenv()

MAX_TOKENS = 1000
TEMPERATURE = 0.2
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

multimodal_models = {
    "mistralai/mistral-medium-3",
    "x-ai/grok-2-vision-1212",
    "meta-llama/llama-4-maverick",
    "openai/gpt-4.1-mini",
    "google/gemini-2.5-flash-preview-05-20",
    "anthropic/claude-opus-4",
    "anthropic/claude-sonnet-4",
}
models_to_test = [
    {"name": "deepseek/deepseek-chat", "use_openrouter": True},
    {"name": "x-ai/grok-2-vision-1212", "use_openrouter": True},
    {"name": "x-ai/grok-2-1212", "use_openrouter": True},
    {"name": "meta-llama/llama-4-maverick", "use_openrouter": True},
    {"name": "mistralai/mistral-medium-3", "use_openrouter": True},
    {"name": "openai/gpt-4.1-mini", "use_openrouter": True},
    {"name": "google/gemini-2.5-flash-preview-05-20", "use_openrouter": True},
    {"name": "anthropic/claude-opus-4", "use_openrouter": True},
    {"name": "anthropic/claude-sonnet-4", "use_openrouter": True},
]

ID_TEMPS = [200, 400, 600]
OOD_TEMPS = [0, 50, 150, 900, 950, 1000]
ALL_TEMPS = set(range(0, 1001, 50))
ID_SAMPLES_PER_TEMP = 1
OOD_SAMPLES_PER_TEMP = 1
OOD_PREV_TEMPS = sorted(list(set(range(150, 851, 50)) - set(ID_TEMPS)))


def call_openrouter_llm(messages, model, max_retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            response_json = response.json()
            if "error" in response_json:
                raise Exception(f"OpenRouter API Error: {response_json['error']}")
            return response_json["choices"][0]["message"]["content"]
        except Exception:
            print(
                f"OpenRouter API error (attempt {attempt + 1}/{max_retries})"
            )  # noqa: T201
            if attempt == max_retries - 1:
                return None
            time.sleep(2**attempt)


def highlight_values_in_template(template_summary):
    # Highlight numbers and values in double curly braces
    # This will match numbers, ranges, coordinates, and values like 1.02:1.01:1.00
    def replacer(match):
        return f"{{{{{match.group(0)}}}}}"

    # Numbers (integers, floats, scientific notation)
    pattern = r"-?\d+\.\d+|-?\d+|\d+\.\d+|\d+|\d+\.\d+e[+-]?\d+"
    highlighted = re.sub(pattern, replacer, template_summary)
    # Also highlight tuples like (x, y, z)
    highlighted = re.sub(
        r"\(([^)]+)\)", lambda m: f"{{{{({m.group(1)})}}}}", highlighted
    )
    # Also highlight ranges like 1.80–2.33 or 0.85–13.24
    highlighted = re.sub(
        r"(\d+\.\d+–\d+\.\d+)", lambda m: f"{{{{{m.group(1)}}}}}", highlighted
    )
    # Also highlight axis ratios like 1.02:1.01:1.00
    highlighted = re.sub(
        r"(\d+\.\d+:\d+\.\d+:\d+\.\d+)", lambda m: f"{{{{{m.group(1)}}}}}", highlighted
    )
    return highlighted


def build_summary_generation_prompt(
    context_summaries, context_temps, target_temp, model_name, image_path
):
    system_content = (
        "You are a materials science expert. "
        "Given two example summaries for a nanoparticle at different temperatures, and an image of the nanoparticle at a new temperature, "
        "generate a summary for the nanoparticle at the new temperature. "
        "Use the same structure and style as the examples, but fill in all values for the new temperature and image."
    )
    context_str = ""
    for temp, summary in zip(context_temps, context_summaries):
        context_str += f"Example summary (for temperature {temp} K):\n{highlight_values_in_template(summary)}\n\n"
    user_text = (
        f"{context_str}"
        f"Now, generate a summary for the nanoparticle at temperature {target_temp} K, using the image provided. "
        f"Fill in all values in double curly braces for the new temperature and image."
    )
    if model_name in multimodal_models:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
        user_content = [
            {"type": "text", "text": user_text},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_b64}",
                    "detail": "low",
                },
            },
        ]
    else:
        user_content = user_text
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return messages


def process_sample_summary(sample, temp_set, phase, temp, summary_lookup, model_name):
    # Gather all available context temps for this phase/rotation
    available_contexts = [
        (t, summary_lookup[(phase, t, sample["rotation"])])
        for t in OOD_PREV_TEMPS
        if (phase, t, sample["rotation"]) in summary_lookup
    ]
    if len(available_contexts) < 2:
        return None
    chosen_contexts = random.sample(available_contexts, 2)
    context_temps, context_summaries = zip(*chosen_contexts)
    messages = build_summary_generation_prompt(
        context_summaries=context_summaries,
        context_temps=context_temps,
        target_temp=temp,
        model_name=model_name,
        image_path=sample["image_path"],
    )
    try:
        llm_output = call_openrouter_llm(messages, model=model_name)
        print(
            f"Model: {model_name} | Phase: {phase} | Temp: {temp} | Rot: {sample['rotation']} | Output: {llm_output}"
        )
    except Exception:
        print(f"Error for sample {sample.get('image_path', '')}")
        llm_output = None
    result = {
        "phase": phase,
        "temperature": temp,
        "rotation": sample["rotation"],
        "set": temp_set,
        "model": model_name,
        "llm_output": llm_output,
        "reference_summary": sample["text"],
        "context_summaries": context_summaries,
        "context_temps": context_temps,
    }
    return result


def main():
    dataset = LLMLoader(label_dir="data_revised", modalities=["text", "image"])
    # Build summary lookup: only one summary per (phase, temp, rotation)
    summary_lookup = {}
    for sample in dataset:
        key = (sample["phase"], sample["temperature"], sample["rotation"])
        if key not in summary_lookup:
            summary_lookup[key] = sample["text"]
    # Group by phase and temperature
    phase_temp_samples = {}
    for idx, sample in enumerate(dataset):
        key = (sample["phase"], sample["temperature"])
        phase_temp_samples.setdefault(key, []).append((idx, sample))
    for model_cfg in tqdm(models_to_test, desc="Models"):
        model_name = model_cfg["name"]
        model_results = []
        all_samples = []
        for temp_set, temps, n_samples in [
            ("ID", ID_TEMPS, ID_SAMPLES_PER_TEMP),
            ("OOD", OOD_TEMPS, OOD_SAMPLES_PER_TEMP),
        ]:
            for phase in ["anatase", "brookite", "rutile"]:
                for temp in temps:
                    key = (phase, temp)
                    samples = phase_temp_samples.get(key, [])
                    if not samples:
                        continue
                    chosen = random.sample(samples, min(n_samples, len(samples)))
                    for idx, sample in chosen:
                        all_samples.append((sample, temp_set, phase, temp))
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(
                    process_sample_summary,
                    sample,
                    temp_set,
                    phase,
                    temp,
                    summary_lookup,
                    model_name,
                )
                for sample, temp_set, phase, temp in all_samples
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"{model_name} (all samples)",
            ):
                result = future.result()
                if result is not None:
                    model_results.append(result)
        output_dir = os.path.join(
            "results", "llm_summary", model_name.replace("/", "_")
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "llm_summary_generation_outputs.json")
        with open(output_path, "w") as f:
            json.dump(model_results, f, indent=2)
    print("Saved per-sample summary generation outputs for each model.")


if __name__ == "__main__":
    main()
