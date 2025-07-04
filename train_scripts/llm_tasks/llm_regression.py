import os
from dotenv import load_dotenv
load_dotenv()
import json
import requests
import time
import random
from tqdm import tqdm
from dataloaders.llm_regression_dataloader import LLMLoader
from PIL import Image
from io import BytesIO
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========== CONFIG ==========
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TEMPERATURE = 0.2
MAX_TOKENS = 1000

# Model lists
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

ID_TEMPS = [200, 400, 600]
OOD_TEMPS = [0, 50, 150, 900, 950, 1000]
ALL_TEMPS = set(range(0, 1001, 50))
OOD_PREV_TEMPS = sorted(list(set(range(150, 851, 50)) - set(ID_TEMPS)))
ID_SAMPLES_PER_TEMP = 1
OOD_SAMPLES_PER_TEMP = 1

# ========== LLM CALL ==========
def call_openrouter_llm(messages, model, max_retries=3):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            response_json = response.json()
            if 'error' in response_json:
                raise Exception(f"OpenRouter API Error: {response_json['error']}")
            return response_json['choices'][0]['message']['content']
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Failed to get response from OpenRouter after retries.")

# ========== PROMPT GENERATION ==========
def build_property_prediction_prompt(text, image_path, curr_temp, prev_temp, prev_props, model_name):
    system_content = (
        "You are a materials science expert. "
        "Given a nanoparticle's summary, the current temperature, and property values at a different temperature, "
        "predict the following properties: HOMO, LUMO, Eg, Ef, Et (in eV). "
        "Return your answer as a Python dict with keys: HOMO, LUMO, Eg, Ef, Et. "
        "Do not include any explanation or extra text."
    )
    prev_str = (
        f"At {prev_temp} K, the properties were: "
        f"HOMO: {prev_props['HOMO']}, LUMO: {prev_props['LUMO']}, "
        f"Eg: {prev_props['Eg']}, Ef: {prev_props['Ef']}, Et: {prev_props['Et']}."
    )
    user_text = (
        f"Current temperature: {curr_temp} K\n"
        f"Summary:\n{text}\n\n"
        f"{prev_str}\n"
        "Predict the properties and return as a Python dict."
    )
    if model_name in multimodal_models:
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}", "detail": "low"}}
        ]
    else:
        user_content = user_text
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    return messages

# ========== PROPERTY EXTRACTION ==========
def extract_properties_from_llm_output(output):
    """
    Parse the LLM output and return a dict with keys: HOMO, LUMO, Eg, Ef, Et (as floats).
    """
    try:
        props = eval(output, {"__builtins__": {}})
        return {k: float(props[k]) for k in ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et']}
    except Exception:
        # Fallback: try to parse numbers from text
        import re
        keys = ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et']
        props = {}
        for k in keys:
            match = re.search(rf"{k}\s*[:=]\s*([-+]?\d*\.?\d+)", output)
            if match:
                props[k] = float(match.group(1))
        return props

# ========== ERROR COMPUTATION ==========
def compute_error(pred, true):
    """Compute absolute error for each property."""
    return {k: abs(pred.get(k, 0) - true[k]) for k in true}

# ========== MAIN ==========
def process_sample(sample, model_name, temp_set, phase, temp, sample_lookup, ood_prev_lookup):
    gt = {k: sample[k] for k in ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et']}
    curr_temp = sample['temperature']
    rotation = sample['rotation']
    if temp_set == 'ID':
        prev_temp = curr_temp - 50
        prev_key = (phase, prev_temp, rotation)
        prev_props = sample_lookup.get(prev_key, None)
        if prev_props is None:
            return None
    else:  # OOD
        candidates = [
            (t, ood_prev_lookup.get((phase, t, rotation)))
            for t in OOD_PREV_TEMPS
            if ood_prev_lookup.get((phase, t, rotation)) is not None
        ]
        if not candidates:
            return None
        prev_temp, prev_props = random.choice(candidates)
    messages = build_property_prediction_prompt(
        text=sample['text'],
        image_path=sample['image_path'],
        curr_temp=curr_temp,
        prev_temp=prev_temp,
        prev_props=prev_props,
        model_name=model_name
    )
    try:
        llm_output = call_openrouter_llm(messages, model=model_name)
        print(f"Model: {model_name} | Phase: {phase} | Temp: {temp} | Rot: {rotation} | Output: {llm_output}")
        # Try to parse output, set missing values to None
        parsed = {'HOMO': None, 'LUMO': None, 'Eg': None, 'Ef': None, 'Et': None}
        if llm_output:
            import re
            import ast
            try:
                # Try to parse as dict
                if isinstance(llm_output, dict):
                    out_dict = llm_output
                else:
                    llm_output_clean = re.sub(r"^```[a-zA-Z]*|```$", "", str(llm_output)).strip()
                    out_dict = ast.literal_eval(llm_output_clean)
                for k in parsed:
                    parsed[k] = out_dict.get(k, None)
            except Exception:
                # Try regex fallback
                for k in parsed:
                    match = re.search(rf"{k}\\s*[:=]\\s*([-+]?\\d*\\.?\\d+)", str(llm_output))
                    if match:
                        try:
                            parsed[k] = float(match.group(1))
                        except Exception:
                            parsed[k] = None
        result = {
            'phase': phase,
            'temperature': temp,
            'rotation': rotation,
            'set': temp_set,
            'model': model_name,
            'llm_output': llm_output,
            'parsed': parsed
        }
        return result
    except Exception as e:
        print(f"Error for sample {sample.get('image_path', '')}: {e}")
        llm_output = None
        parsed = {'HOMO': None, 'LUMO': None, 'Eg': None, 'Ef': None, 'Et': None}
        result = {
            'phase': phase,
            'temperature': temp,
            'rotation': rotation,
            'set': temp_set,
            'model': model_name,
            'llm_output': llm_output,
            'parsed': parsed
        }
        return result

def main():
    dataset = LLMLoader(
        label_dir="data_revised",
        modalities=["text", "image"]
    )
    # Build a lookup for (phase, temperature, rotation) -> properties
    sample_lookup = {}
    for sample in dataset:
        key = (sample['phase'], sample['temperature'], sample['rotation'])
        sample_lookup[key] = {k: sample[k] for k in ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et']}
    # Group by phase and temperature
    phase_temp_samples = {}
    for idx, sample in enumerate(dataset):
        key = (sample['phase'], sample['temperature'])
        phase_temp_samples.setdefault(key, []).append((idx, sample))
    ood_prev_lookup = {}
    for sample in dataset:
        if sample['temperature'] in OOD_PREV_TEMPS:
            key = (sample['phase'], sample['temperature'], sample['rotation'])
            ood_prev_lookup[key] = {k: sample[k] for k in ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et']}
    for model_cfg in tqdm(models_to_test, desc="Models"):
        model_name = model_cfg['name']
        model_results = []
        for temp_set, temps, n_samples in [('ID', ID_TEMPS, ID_SAMPLES_PER_TEMP), ('OOD', OOD_TEMPS, OOD_SAMPLES_PER_TEMP)]:
            for phase in ['anatase', 'brookite', 'rutile']:
                for temp in temps:
                    key = (phase, temp)
                    samples = phase_temp_samples.get(key, [])
                    if not samples:
                        continue
                    chosen = random.sample(samples, min(n_samples, len(samples)))
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [
                            executor.submit(process_sample, sample, model_name, temp_set, phase, temp, sample_lookup, ood_prev_lookup)
                            for idx, sample in chosen
                        ]
                        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{model_name} {phase} {temp}K", leave=False):
                            result = future.result()
                            if result is not None:
                                # Calculate error for this sample
                                gt = {k: sample[k] for k in ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et']}
                                error = {}
                                for k in ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et']:
                                    try:
                                        if result['parsed'][k] is not None and gt[k] is not None:
                                            error[k] = abs(result['parsed'][k] - gt[k])
                                        else:
                                            error[k] = None
                                    except Exception:
                                        error[k] = None
                                result['error'] = error
                                result['ground_truth'] = gt
                                model_results.append(result)
        # Compute MAE for this model
        mae = {}
        for k in ['HOMO', 'LUMO', 'Eg', 'Ef', 'Et']:
            vals = [r['error'][k] for r in model_results if r['error'][k] is not None]
            mae[k] = sum(vals) / len(vals) if vals else None
        # Save as a dict with results and mae
        output = {'results': model_results, 'mae': mae}
        output_dir = os.path.join("results", "llm_regression", model_name.replace('/', '_'))
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "llm_property_prediction_outputs.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
    print("Saved per-sample outputs and MAE for each model.")

if __name__ == "__main__":
    main()