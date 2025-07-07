import base64
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
from PIL import Image
from tqdm import tqdm

from configs.llm_config import (
    BASE_DIR,
    FEW_SHOT_EXAMPLES,
    ID_SAMPLES_PER_TEMP,
    ID_TEMPS,
    MAX_RETRIES,
    MAX_ROTATIONS,
    MAX_TOKENS,
    MAX_WORKERS,
    MODALITIES,
    MODELS_TO_TEST,
    MULTIMODAL_MODELS,
    OOD_SAMPLES_PER_TEMP,
    OOD_TEMPS,
    OPENROUTER_API_KEY,
    TARGET_PROPERTIES,
    TEMPERATURE,
    TIMEOUT,
    TRAIN_TEMPS,
)
from dataloaders.llm_regression_dataloader import LLMLoader

# ========== CONFIG ==========
# All configuration is now imported from configs.llm_config


# ========== LLM CALL ==========
def call_openrouter_llm(messages, model, max_retries=MAX_RETRIES):
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
                timeout=TIMEOUT,
            )
            response.raise_for_status()
            response_json = response.json()
            if "error" in response_json:
                raise Exception(f"OpenRouter API Error: {response_json['error']}")
            return response_json["choices"][0]["message"]["content"]
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Failed to get response from OpenRouter after retries.")


# ========== PROMPT GENERATION ==========
def build_property_prediction_prompt(
    text, image_path, curr_temp, few_shot_examples, model_name
):
    target_props_str = ", ".join(TARGET_PROPERTIES)
    system_content = (
        "You are a materials science expert. "
        "Given a nanoparticle's summary, the current temperature, and examples of properties at different temperatures, "
        f"predict the following properties: {target_props_str} (in eV). "
        "\n\nProperty definitions:"
        "\n- HOMO: HOMO energy (E_H)"
        "\n- LUMO: LUMO energy (E_L)"
        "\n- Eg: band gap energy (E_g)"
        "\n- Ef: Fermi energy (E_f)"
        "\n- Et: total energy of the system (E_T)"
        "\n- Eta: total energy per atom (E_Ta)"
        "\n- disp: maximum atomic displacement (Δr_max)"
        "\n- vol: volumetric expansion (ΔV)"
        "\n- bond: Ti-O bond length change (Δd_Ti-O)"
        f"\n\nReturn your answer as a Python dict with keys: {target_props_str}. "
        "Use 4-digit precision for all values (e.g., 1.2340). "
        "Do not include any explanation or extra text."
    )

    # Build few-shot examples
    examples_text = ""
    for i, example in enumerate(few_shot_examples):
        example_props_str = ", ".join(
            [f"{k}: {example['properties'][k]:.4f}" for k in TARGET_PROPERTIES]
        )
        examples_text += f"Example {i + 1}:\n"
        examples_text += f"Temperature: {example['temperature']} K\n"
        examples_text += f"Summary: {example['text']}\n"
        examples_text += f"Properties: {example_props_str}\n\n"

    user_text = (
        f"Current temperature: {curr_temp} K\n"
        f"Summary:\n{text}\n\n"
        f"Here are some examples:\n{examples_text}"
        "Predict the properties and return as a Python dict."
    )

    if model_name in MULTIMODAL_MODELS:
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
    print("\n--- SYSTEM PROMPT ---\n", system_content)
    print("\n--- USER PROMPT ---\n", user_text)
    return messages


# ========== PROPERTY EXTRACTION ==========
def extract_properties_from_llm_output(output):
    """
    Parse the LLM output and return a dict with keys: HOMO, LUMO, Eg, Ef, Et (as floats).
    """
    try:
        props = eval(output, {"__builtins__": {}})
        return {k: float(props[k]) for k in TARGET_PROPERTIES}
    except Exception:
        # Fallback: try to parse numbers from text
        import re

        props = {}
        for k in TARGET_PROPERTIES:
            match = re.search(rf"{k}\s*[:=]\s*([-+]?\d*\.?\d+)", output)
            if match:
                props[k] = float(match.group(1))
        return props


# ========== ERROR COMPUTATION ==========
def compute_error(pred, true):
    """Compute absolute error for each property."""
    return {k: abs(pred.get(k, 0) - true[k]) for k in true}


# ========== MAIN ==========
def process_sample(
    sample, model_name, temp_set, phase, temp, sample_lookup, train_samples
):
    _ = {k: sample[k] for k in TARGET_PROPERTIES}
    curr_temp = sample["temperature"]
    rotation = sample["rotation"]

    # Get few-shot examples from training temperatures
    available_train_samples = [
        s
        for s in train_samples
        if s["phase"] == phase and s["temperature"] in TRAIN_TEMPS
    ]

    if len(available_train_samples) < FEW_SHOT_EXAMPLES:
        return None

    # Randomly select few-shot examples
    few_shot_examples = random.sample(available_train_samples, FEW_SHOT_EXAMPLES)

    # Format examples for the prompt and record indices
    formatted_examples = []
    few_shot_indices = []
    for example in few_shot_examples:
        formatted_examples.append(
            {
                "temperature": example["temperature"],
                "text": example["text"],
                "properties": {k: example[k] for k in TARGET_PROPERTIES},
            }
        )
        few_shot_indices.append(
            {"temperature": example["temperature"], "rotation": example["rotation"]}
        )

    messages = build_property_prediction_prompt(
        text=sample["text"],
        image_path=sample["image_path"],
        curr_temp=curr_temp,
        few_shot_examples=formatted_examples,
        model_name=model_name,
    )
    try:
        llm_output = call_openrouter_llm(messages, model=model_name)
        print(
            f"Model: {model_name} | Phase: {phase} | Temp: {temp} | Rot: {rotation} | Few-shot examples: {len(few_shot_examples)} | Output: {llm_output}"
        )
        # Try to parse output, set missing values to None
        parsed = {k: None for k in TARGET_PROPERTIES}
        if llm_output:
            import ast
            import re

            try:
                # Try to parse as dict
                if isinstance(llm_output, dict):
                    out_dict = llm_output
                else:
                    llm_output_clean = re.sub(
                        r"^```[a-zA-Z]*|```$", "", str(llm_output)
                    ).strip()
                    out_dict = ast.literal_eval(llm_output_clean)
                for k in parsed:
                    parsed[k] = out_dict.get(k, None)
            except Exception:
                # Try regex fallback
                for k in parsed:
                    match = re.search(
                        rf"{k}\\s*[:=]\\s*([-+]?\\d*\\.?\\d+)", str(llm_output)
                    )
                    if match:
                        try:
                            parsed[k] = float(match.group(1))
                        except Exception:
                            parsed[k] = None
        result = {
            "phase": phase,
            "temperature": temp,
            "rotation": rotation,
            "set": temp_set,
            "model": model_name,
            "llm_output": llm_output,
            "parsed": parsed,
            "few_shot_examples_used": len(few_shot_examples),
            "few_shot_indices": few_shot_indices,
        }
        return result
    except Exception:
        print(f"Error for sample {sample.get('image_path', '')}")
        llm_output = None
        parsed = {k: None for k in TARGET_PROPERTIES}
        result = {
            "phase": phase,
            "temperature": temp,
            "rotation": rotation,
            "set": temp_set,
            "model": model_name,
            "llm_output": llm_output,
            "parsed": parsed,
            "few_shot_examples_used": len(few_shot_examples),
            "few_shot_indices": few_shot_indices,
        }
        return result


def main():
    dataset = LLMLoader(
        label_dir=BASE_DIR, modalities=MODALITIES, max_rotations=MAX_ROTATIONS
    )
    # Build a lookup for (phase, temperature, rotation) -> properties
    sample_lookup = {}
    for sample in dataset:
        key = (sample["phase"], sample["temperature"], sample["rotation"])
        sample_lookup[key] = {k: sample[k] for k in TARGET_PROPERTIES}
    # Group by phase and temperature
    phase_temp_samples = {}
    for idx, sample in enumerate(dataset):
        key = (sample["phase"], sample["temperature"])
        phase_temp_samples.setdefault(key, []).append((idx, sample))

    # Prepare training samples for few-shot examples
    train_samples = []
    for sample in dataset:
        if sample["temperature"] in TRAIN_TEMPS:
            train_samples.append(sample)
    for model_cfg in tqdm(MODELS_TO_TEST, desc="Models"):
        model_name = model_cfg["name"]
        model_results = []
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
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [
                            executor.submit(
                                process_sample,
                                sample,
                                model_name,
                                temp_set,
                                phase,
                                temp,
                                sample_lookup,
                                train_samples,
                            )
                            for idx, sample in chosen
                        ]
                        for future in tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"{model_name} {phase} {temp}K",
                            leave=False,
                        ):
                            result = future.result()
                            if result is not None:
                                # Calculate error for this sample
                                gt = {k: sample[k] for k in TARGET_PROPERTIES}
                                error = {}
                                for k in TARGET_PROPERTIES:
                                    try:
                                        if (
                                            result["parsed"][k] is not None
                                            and gt[k] is not None
                                        ):
                                            error[k] = abs(result["parsed"][k] - gt[k])
                                        else:
                                            error[k] = None
                                    except Exception:
                                        error[k] = None
                                result["error"] = error
                                result["ground_truth"] = gt
                                model_results.append(result)
        # Compute MAE for this model
        mae = {}
        for k in TARGET_PROPERTIES:
            vals = [r["error"][k] for r in model_results if r["error"][k] is not None]
            mae[k] = sum(vals) / len(vals) if vals else None
        # Save as a dict with results and mae
        output = {"results": model_results, "mae": mae}
        output_dir = os.path.join(
            "results", "llm_regression2", model_name.replace("/", "_")
        )
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "llm_property_prediction_outputs.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
    print("Saved per-sample outputs and MAE for each model.")


if __name__ == "__main__":
    main()
