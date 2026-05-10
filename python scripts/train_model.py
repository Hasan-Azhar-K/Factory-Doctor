"""
train_model.py  –  Factory Doctor FYP
Trains a two-layer LSTM model on the multi-machine dataset.
Run AFTER generate_dataset.py.

Data split (done BEFORE any training):
  First 80% of each machine's rows  →  training + validation
  Last  20% of each machine's rows  →  saved as simulation_data.csv
                                        (completely unseen during training)

This ensures simulate_and_monitor.py runs on data the model has never seen,
which is the correct way to demonstrate a real predictive maintenance system.

Outputs:
  factory_doctor_lstm.keras  - trained Keras model
  feature_scaler.pkl         - fitted StandardScaler (must match simulation)
  simulation_data.csv        - held-out rows for the simulator
  training_curves.png        - loss and AUC plots (use in your FYP report)
"""

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf

# ─── Config ───────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 20          # number of past readings fed to the LSTM
FEATURES = [
    'Air_Temperature_K',
    'Process_Temperature_K',
    'Rotational_Speed_RPM',
    'Torque_Nm',
    'Tool_Wear_Min',
]
BATCH_SIZE  = 64
MAX_EPOCHS  = 60
MODEL_PATH  = 'factory_doctor_lstm.keras'
SCALER_PATH = 'feature_scaler.pkl'

# ─── 1. Load & split BEFORE anything else ────────────────────────────────────
# The split is done here, on raw unscaled data, so the scaler is ONLY fitted
# on training rows. Fitting the scaler on all data (including simulation rows)
# would be a form of data leakage.
print("[1/5] Loading data and splitting off simulation set …")
df = pd.read_csv('multi_machine_data.csv')

train_parts = []
sim_parts   = []

for machine_id, group in df.groupby('Machine_ID'):
    group = group.reset_index(drop=True)
    split_idx = int(len(group) * 0.80)          # first 80% → training
    train_parts.append(group.iloc[:split_idx])   # last  20% → simulation
    sim_parts.append(group.iloc[split_idx:])

train_df = pd.concat(train_parts).reset_index(drop=True)
sim_df   = pd.concat(sim_parts).reset_index(drop=True)

# Save simulation data BEFORE scaling (simulator scales its own rows at runtime)
sim_df.to_csv('simulation_data.csv', index=False)

print(f"      Training rows  : {len(train_df):,}  (80% of each machine)")
print(f"      Simulation rows: {len(sim_df):,}  (last 20% -- never seen during training)")
print(f"      Simulation data saved → simulation_data.csv")

# ─── Fit scaler ONLY on training data ────────────────────────────────────────
scaler = StandardScaler()
train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
joblib.dump(scaler, SCALER_PATH)
print(f"      Scaler fitted on training data only → {SCALER_PATH}")

# ─── 2. Build sequences per machine ───────────────────────────────────────────
print("[2/5] Creating LSTM sequences …")

def make_sequences(data: np.ndarray, labels: np.ndarray, seq_len: int):
    """Sliding-window: each sample is seq_len consecutive readings."""
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len: i])
        y.append(labels[i])
    return np.array(X), np.array(y)

all_X, all_y = [], []
for machine_id, group in train_df.groupby('Machine_ID'):
    group = group.reset_index(drop=True)
    Xm, ym = make_sequences(
        group[FEATURES].values,
        group['Machine_Failure'].values,
        SEQUENCE_LENGTH,
    )
    all_X.append(Xm)
    all_y.append(ym)

X = np.vstack(all_X)
y = np.concatenate(all_y)
print(f"      Total sequences: {len(X):,}   Shape: {X.shape}")
print(f"      Failure rate   : {y.mean():.2%}")

# ─── 3. Train / test split ────────────────────────────────────────────────────
print("[3/5] Splitting data …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Handle class imbalance with class weights
neg, pos = np.bincount(y_train.astype(int))
class_weight = {0: 1.0, 1: round(neg / pos, 2)}
print(f"      Class weight for failure class: {class_weight[1]:.1f}×")

# ─── 4. Build LSTM model ──────────────────────────────────────────────────────
print("[4/5] Building model …")
model = tf.keras.Sequential([
    # First LSTM layer – learns temporal patterns over the sequence
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(FEATURES))),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    # Second LSTM layer – compresses to a single context vector
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    # Classification head
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid'),   # outputs P(failure)
], name='FactoryDoctor_LSTM')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ],
)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=8,
                                     restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_auc', mode='max',
                                       save_best_only=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4,
                                        min_lr=1e-6, verbose=1),
]

print("[4/5] Training …")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# ─── 5. Evaluate ──────────────────────────────────────────────────────────────
print("\n[5/5] Evaluation …")
y_proba = model.predict(X_test, verbose=0).flatten()
y_pred  = (y_proba >= 0.5).astype(int)

print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Failure']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print(f"\n── Confusion Matrix ───────────────────────────────────")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

# ── Training curve plot ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss'); axes[0].legend()

axes[1].plot(history.history['auc'],     label='Train AUC')
axes[1].plot(history.history['val_auc'], label='Val AUC')
axes[1].set_title('AUC'); axes[1].legend()

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150)
print("\nTraining curves saved → training_curves.png")
print(f"Model saved          → {MODEL_PATH}")
