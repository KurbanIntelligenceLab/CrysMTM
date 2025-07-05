BASE_DIR = "CrysMTM"
MAX_ROTATIONS = None

# Temperature splits
TRAIN_RANGE = range(0, 851, 50)
TRAIN_TEMPS = [T for T in TRAIN_RANGE if T not in [250, 450, 650, 750, 800]]
ID_TEMPS = [250, 450, 650, 750, 800]
OOD_TEMPS = [0, 50, 100, 900, 950, 1000]

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_EPOCHS = 150
LEARNING_RATE = 1e-3
SEEDS = [10, 20, 30]
TARGET_PROPERTIES = ["HOMO", "LUMO", "Eg", "Ef", "Et", "Eta", "disp", "vol", "bond"]

# Label normalization parameters
NORMALIZE_LABELS = True
NORMALIZATION_METHOD = "standard"  # "standard" for z-score, "minmax" for min-max scaling

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4


class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop
