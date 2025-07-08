# CrysMTM Dataset Card

## Dataset Description

- **Repository:** [CrysMTM](https://github.com/KurbanIntelligenceLab/CrysMTM)
- **Paper:** CrysMTM: A Multiphase, Temperature-Resolved, Multimodal Dataset for Crystalline Materials
- **Authors:** Can Polat, Erchin Serpedin, Mustafa Kurban, and Hasan Kurban
- **Point of Contact:** [Can Polat](johnpolat.com)

### Dataset Summary

CrysMTM is a comprehensive multiphase, temperature-resolved, multimodal dataset for crystalline materials research, specifically focused on titanium dioxide (TiO₂) polymorphs. The dataset is designed primarily for regression tasks to predict 9 key material properties from multimodal inputs. It contains three crystalline phases of TiO₂ (anatase, brookite, and rutile) across a temperature range of 0-1000K, with multiple data modalities including atomic coordinates, visual representations, and textual descriptions.

### Supported Tasks and Leaderboards

The dataset primarily supports regression tasks for materials property prediction:

1. **Main Task - Regression**: Predict 9 material properties from multimodal inputs
   - HOMO energy, LUMO energy, band gap, Fermi energy, total energy, energy per atom, atomic displacement, volumetric expansion, and bond length changes
2. **Main Task - LLM Property Prediction**: Zero-shot and few-shot prediction of the 9 material properties using large language models
3. **Secondary Task - LLM Summary Generation**: Generate textual summaries of crystal structures and properties using large language models
4. **Tertiary Task - Classification**: Predict the crystalline phase (anatase, brookite, or rutile) from multimodal inputs

### Languages

The dataset contains English text descriptions of crystal structures and properties.

## Dataset Structure

### Data Instances

Each data instance represents a TiO₂ crystal structure at a specific temperature and rotation, containing:

- **Phase**: One of three TiO₂ polymorphs (anatase, brookite, rutile)
- **Temperature**: Temperature in Kelvin (0-1000K, in 50K increments)
- **Rotation**: Rotation index for the crystal structure
- **Modalities**: Multiple data representations of the same structure

### Data Fields

#### Core Metadata
- `phase` (string): Crystal phase - "anatase", "brookite", or "rutile"
- `temperature` (integer): Temperature in Kelvin (0, 50, 100, ..., 1000)
- `rotation` (integer): Rotation index for the crystal structure

#### Multimodal Data
- `image` (PIL.Image): Visual representation of the crystal structure (PNG format)
- `xyz` (torch.Tensor): Atomic coordinates in XYZ format (N×3 tensor)
- `text` (string): Textual description of the crystal structure and properties
- `element` (list): List of element symbols for each atom

#### Labels
**Primary Labels - Regression**:
- `regression_label` (torch.Tensor): 9-dimensional tensor containing the main prediction targets:
  - `HOMO` (float): HOMO energy (E_H) in eV
  - `LUMO` (float): LUMO energy (E_L) in eV
  - `Eg` (float): Band gap energy (E_g) in eV
  - `Ef` (float): Fermi energy (E_f) in eV
  - `Et` (float): Total energy of the system (E_T) in eV
  - `Eta` (float): Total energy per atom (E_Ta) in eV
  - `disp` (float): Maximum atomic displacement (Δr_max) in Å
  - `vol` (float): Volumetric expansion (ΔV) in Å³
  - `bond` (float): Ti-O bond length change (Δd_Ti-O) in Å

**Secondary Labels - Classification**:
- `label` (integer): Phase label (0=anatase, 1=brookite, 2=rutile)

**LLM Task Labels**:
- Individual property values for zero-shot/few-shot prediction
- Text summaries for generation tasks

### Data Splits

The dataset is organized by temperature ranges with varying rotation densities:

- **Training Set**: Temperatures 0-850K (excluding 250K, 450K, 650K, 750K, 800K)
- **In-Distribution (ID) Test**: Temperatures 250K, 450K, 650K, 750K, 800K
- **Out-of-Distribution (OOD) Test**: Temperatures 0K, 50K, 100K, 900K, 950K, 1000K

### Citation Information
```bibtex
SOON
```

## Usage Examples

### Option 1: Download and Use Locally

1. **Download the dataset** from [FigShare](https://doi.org/10.6084/m9.figshare.29497604.v1)
2. **Use the provided loading script**:

```python
# Download load_dataset.py from the repository and place it in your data directory
from load_dataset import load_dataset

# Load the dataset
dataset = load_dataset(".")

# Access splits
train_dataset = dataset["train"]      # 5,064 samples
test_id_dataset = dataset["test_id"]  # 1,380 samples
test_ood_dataset = dataset["test_ood"] # 6,588 samples

# Get a sample
sample = train_dataset[0]
print(f"Phase: {sample['phase']}")
print(f"Temperature: {sample['temperature']}K")
print(f"Image: {sample['image']}")
print(f"Regression labels: {sample['regression_labels']}")
```

### Option 2: Use with Original Dataloaders

```python
from dataloaders.regression_dataloader import RegressionLoader

# Load dataset for regression (main task)
dataset = RegressionLoader(
    label_dir="data",
    modalities=["image", "xyz", "text"],
    normalize_labels=True
)

# Get a sample
sample = dataset[0]
print(f"Target Properties: {sample['regression_label']}")
print(f"Temperature: {sample['temperature']}K")
print(f"Phase: {sample['phase']}")
print(f"Image shape: {sample['image'].size}")
print(f"XYZ coordinates shape: {sample['xyz'].shape}")
```

### Main Task - LLM Property Prediction
```python
from dataloaders.llm_regression_dataloader import LLMLoader

# Load dataset for LLM property prediction (main task)
dataset = LLMLoader(
    label_dir="data",
    modalities=["text", "image"]
)

# Get a sample for zero-shot/few-shot property prediction
sample = dataset[0]
print(f"HOMO: {sample['HOMO']}")
print(f"LUMO: {sample['LUMO']}")
print(f"Band gap: {sample['Eg']}")
print(f"Temperature: {sample['temperature']}K")
print(f"Phase: {sample['phase']}")
```

### Secondary Task - LLM Summary Generation
```python
from dataloaders.llm_regression_dataloader import LLMLoader

# Load dataset for LLM summary generation (secondary task)
dataset = LLMLoader(
    label_dir="data",
    modalities=["text", "image"]
)

# Get a sample for summary generation
sample = dataset[0]
print(f"Input text: {sample['text'][:200]}...")
print(f"Target properties: {sample['HOMO']}, {sample['LUMO']}, {sample['Eg']}")
```

### Tertiary Task - Classification
```python
from dataloaders.classification_dataloader import ClassificationLoader

# Load dataset for classification (tertiary task)
dataset = ClassificationLoader(
    base_dir="data",
    modalities=["image", "xyz", "text"],
    max_rotations=10
)

# Get a sample
sample = dataset[0]
print(f"Phase: {sample['label']}")
print(f"Image shape: {sample['image'].size}")
print(f"XYZ coordinates shape: {sample['xyz'].shape}")
print(f"Text: {sample['text'][:100]}...")
```

### PyTorch Geometric Integration
```python
# For graph neural networks
dataset = ClassificationLoader(
    base_dir="data",
    modalities=["xyz", "element"],
    as_pyg_data=True
)

# Returns PyG Data objects
sample = dataset[0]
print(f"Node features: {sample.z}")
print(f"Positions: {sample.pos}")
print(f"Label: {sample.y}")
```

## Technical Details

### Dataset Statistics
- **Total samples**: 13,032
- **Total files**: 39,096 (3 files per sample: image, XYZ, text)
- **Phases**: 3 (anatase, brookite, rutile)
- **Temperature range**: 0-1000K (21 temperatures total)
- **Rotation densities**: Variable (59, 92, or 366 rotations per temperature)

### File Structure
```
data/
├── anatase/
│   ├── 0K/          # 366 rotations (high resolution)
│   │   ├── images/
│   │   │   ├── rot_0.png
│   │   │   ├── rot_1.png
│   │   │   └── ... (366 files)
│   │   ├── xyz/
│   │   │   ├── rot_0.xyz
│   │   │   ├── rot_1.xyz
│   │   │   └── ... (366 files)
│   │   └── text/
│   │       ├── rot_0.txt
│   │       ├── rot_1.txt
│   │       └── ... (366 files)
│   ├── 150K/        # 59 rotations (standard resolution)
│   │   ├── images/  # 59 files
│   │   ├── xyz/     # 59 files
│   │   └── text/    # 59 files
│   └── ...
├── brookite/
├── rutile/
└── labels.csv
```

### Data Formats

#### XYZ Files
Standard XYZ format with atomic coordinates:
```
[number of atoms]
[comment line]
[element] [x] [y] [z]
[element] [x] [y] [z]
...
```

#### Images
PNG format visualizations of crystal structures.

#### Text Files
Natural language descriptions of crystal structures and properties.

#### Labels CSV
Contains material properties for each phase-temperature combination:
```csv
Polymorph,Temperature,Parameter,Value
anatase,0K,HOMO,-7.2340
anatase,0K,LUMO,-4.1234
...
```

### Supported Models

The dataset is compatible with various model architectures:

- **Vision Models**: ResNet, ViT
- **Graph Neural Networks**: SchNet, DimeNet, EGNN, FAENet, GoTenNet
- **Language Models**: LLMs for zero-shot/few-shot learning
- **Multimodal Models**: CLIP, Pure2DopeNet, ViSNet 

### Performance Metrics

#### Primary Task - Regression
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² score
- Per-property evaluation metrics

#### Primary Task - LLM Property Prediction
- Property prediction accuracy
- Zero-shot vs few-shot performance comparison
- Out-of-distribution generalization
- Per-property evaluation metrics

#### Secondary Task - LLM Summary Generation
The nanoparticle summary task requires domain-specific evaluation beyond traditional string-based metrics like ROUGE or BLEU, which do not penalize incorrect numerical values. A more meaningful strategy is to extract structured key–value pairs-such as particle size, center of mass, coordination numbers, or bond angles-and compare them to ground truth using:
- Information-level F₁ score that accepts only values within defined tolerances (e.g., 0.1 Å or 1 degree)
- MAPE over all numeric entries
- Factual consistency score like BERTScore or QA-based faithfulness after masking numeric values
- Optional assessments of readability and clarity using expert judgment or coherence-based metrics (e.g., Coh-LM)

#### Tertiary Task - Classification
A three-class classification task to distinguish among the TiO₂ polymorphs. While overall accuracy provides a general overview, it is important to also report:
- Class-wise precision, recall, and their harmonic mean (F₁ score), followed by macro-averaging to account for class imbalance
- Full 3×3 confusion matrix to identify systematic misclassifications between phase pairs
- Matthews correlation coefficient (MCC) and Cohen's κ statistic for chance-adjusted evaluations
- Cross-entropy loss and macro-averaged area under the ROC curve (AUROC) when class probabilities are available