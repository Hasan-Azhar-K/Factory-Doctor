"""
simulate_and_monitor.py  –  Factory Doctor FYP
Simulates live sensor streams for MULTIPLE machines concurrently.
Each machine runs in its own thread and uses a rolling LSTM buffer.

Prerequisites:
  1. generate_dataset.py  →  multi_machine_data.csv
  2. train_model.py       →  factory_doctor_lstm.keras + feature_scaler.pkl
  3. setup_database.sql   →  FactoryDoctorDB created in SQL Server

Run:
  python simulate_and_monitor.py
"""

import pandas as pd
import numpy as np
import time
import joblib
import pyodbc
import smtplib
import warnings
import threading
import queue
from collections import deque
from email.message import EmailMessage
from datetime import datetime, timedelta

import tensorflow as tf

warnings.filterwarnings("ignore", category=UserWarning)

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_PATH  = 'factory_doctor_lstm.keras'
SCALER_PATH = 'feature_scaler.pkl'
DATA_PATH   = 'simulation_data.csv'   # last 20% of each machine -- never seen during training
SEQUENCE_LENGTH = 20   # must match what was used during training

FEATURES = [
    'Air_Temperature_K',
    'Process_Temperature_K',
    'Rotational_Speed_RPM',
    'Torque_Nm',
    'Tool_Wear_Min',
]

# Email / alert settings
EMAIL_ADDRESS  = "voidextremexio@gmail.com"
EMAIL_PASSWORD = "yiae hmhd vrkv vmru"
MANAGER_EMAIL  = "hassan.kh290@gmail.com"
ALERT_COOLDOWN_MIN = 15   # minimum minutes between alerts per machine

# Probability thresholds
THRESHOLD_CRITICAL = 0.70
THRESHOLD_WARNING  = 0.40

# Seconds between simulated sensor reads per machine
READ_INTERVAL = 2

# ─── Load shared resources ────────────────────────────────────────────────────
print("Loading model and scaler …")
model  = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Loading dataset …")
df = pd.read_csv(DATA_PATH)

# ─── DB helper ────────────────────────────────────────────────────────────────
def get_connection():
    """Create a fresh SQL Server connection (one per thread)."""
    return pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=FactoryDoctorDB;'
        'Trusted_Connection=yes;'
    )

# ─── Email alert helper ───────────────────────────────────────────────────────
# Per-machine cooldown tracked in a dict  {machine_id: last_alert_datetime}
_alert_lock = threading.Lock()
_last_alert: dict[str, datetime] = {}

def send_alert(machine_id: str, sensor_data: dict, status: str, failure_prob: float):
    """Send an email alert if the per-machine cooldown has expired."""
    with _alert_lock:
        last = _last_alert.get(machine_id, datetime.min)
        if datetime.now() - last < timedelta(minutes=ALERT_COOLDOWN_MIN):
            return   # still in cooldown
        _last_alert[machine_id] = datetime.now()

    sensor_str = "\n".join(f"  {k}: {v}" for k, v in sensor_data.items())
    msg = EmailMessage()
    msg['Subject'] = f'⚠ Machine {machine_id} – {status} Alert | Factory Doctor'
    msg['From']    = EMAIL_ADDRESS
    msg['To']      = MANAGER_EMAIL
    msg.set_content(
        f"MACHINE HEALTH ALERT\n"
        f"{'='*40}\n"
        f"Machine ID       : {machine_id}\n"
        f"Status           : {status}\n"
        f"Failure Prob.    : {failure_prob:.1%}\n"
        f"Timestamp        : {datetime.now():%Y-%m-%d %H:%M:%S}\n\n"
        f"Live Sensor Readings:\n{sensor_str}\n"
        f"{'='*40}\n"
        f"Please inspect this machine immediately.\n"
        f"– Factory Doctor Monitoring System"
    )

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"  [EMAIL] Alert sent for {machine_id} ({status})")
    except Exception as e:
        print(f"  [EMAIL] Failed for {machine_id}: {e}")


# ─── Shared print lock (avoids garbled console output) ───────────────────────
_print_lock = threading.Lock()

def tprint(msg: str):
    with _print_lock:
        print(msg)


# ─── Per-machine simulation thread ───────────────────────────────────────────
def machine_worker(machine_id: str, machine_df: pd.DataFrame):
    """
    Reads sensor rows one at a time for a single machine,
    maintains a rolling LSTM input buffer, predicts, and logs to DB.
    """
    conn   = get_connection()
    cursor = conn.cursor()

    # Rolling buffer of the last SEQUENCE_LENGTH scaled feature vectors
    buffer: deque = deque(maxlen=SEQUENCE_LENGTH)

    for idx, row in machine_df.iterrows():
        raw_features = {f: row[f] for f in FEATURES}

        # Scale the reading
        scaled = scaler.transform([[raw_features[f] for f in FEATURES]])[0]
        buffer.append(scaled)

        # We need a full sequence before we can predict
        if len(buffer) < SEQUENCE_LENGTH:
            tprint(f"  [{machine_id}] Buffering … ({len(buffer)}/{SEQUENCE_LENGTH})")
            time.sleep(READ_INTERVAL)
            continue

        # Build (1, seq_len, n_features) input tensor
        seq = np.array(buffer)[np.newaxis, :, :]
        failure_prob = float(model.predict(seq, verbose=0)[0][0])

        # 3-tier health classification
        if failure_prob >= THRESHOLD_CRITICAL:
            status = "Critical"
        elif failure_prob >= THRESHOLD_WARNING:
            status = "Warning"
        else:
            status = "Good"

        # ── Write to database ────────────────────────────────────────────
        cursor.execute(
            """
            INSERT INTO MachineData (
                MachineID, AirTemperature, ProcessTemperature,
                RotationalSpeed, Torque, ToolWear,
                FailureProbability, HealthStatus, RecordedAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                machine_id,
                raw_features['Air_Temperature_K'],
                raw_features['Process_Temperature_K'],
                raw_features['Rotational_Speed_RPM'],
                raw_features['Torque_Nm'],
                raw_features['Tool_Wear_Min'],
                failure_prob,
                status,
                datetime.now(),
            )
        )
        conn.commit()

        tprint(
            f"  [{machine_id}] Row {idx:>5} | "
            f"Temp={raw_features['Air_Temperature_K']:.1f}K  "
            f"RPM={raw_features['Rotational_Speed_RPM']:.0f}  "
            f"Wear={raw_features['Tool_Wear_Min']:.1f}min | "
            f"Prob={failure_prob:.2f} → {status}"
        )

        # ── Trigger alert if Critical ────────────────────────────────────
        if status == "Critical":
            threading.Thread(
                target=send_alert,
                args=(machine_id, raw_features, status, failure_prob),
                daemon=True,
            ).start()

        time.sleep(READ_INTERVAL)

    cursor.close()
    conn.close()
    tprint(f"  [{machine_id}] ✓ Simulation complete.")


# ─── Main: launch one thread per machine ──────────────────────────────────────
if __name__ == '__main__':
    machine_ids = df['Machine_ID'].unique()
    print(f"\n{'='*60}")
    print(f"  Factory Doctor – Multi-Machine Monitor")
    print(f"  Machines : {', '.join(machine_ids)}")
    print(f"  Model    : LSTM  |  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Thresholds: Warning≥{THRESHOLD_WARNING:.0%}  Critical≥{THRESHOLD_CRITICAL:.0%}")
    print(f"{'='*60}\n")

    threads = []
    for mid in machine_ids:
        machine_subset = df[df['Machine_ID'] == mid].reset_index(drop=True)
        t = threading.Thread(
            target=machine_worker,
            args=(mid, machine_subset),
            name=f"Thread-{mid}",
            daemon=False,
        )
        threads.append(t)

    # Stagger starts slightly so DB inserts don't all hit at the same millisecond
    for i, t in enumerate(threads):
        time.sleep(0.3 * i)
        t.start()
        print(f"  Started thread for {machine_ids[i]}")

    for t in threads:
        t.join()

    print("\nAll machine simulations finished.")
