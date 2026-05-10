"""
generate_dataset.py  –  Factory Doctor FYP
Generates a realistic multi-machine sensor dataset with degradation patterns.
Run this ONCE before training the model.
Output: multi_machine_data.csv  (~10,000 rows, 5 machines)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ─── Configuration ───────────────────────────────────────────────────────────
MACHINES = {
    'M001': {'type': 'L', 'base_rpm': 1400, 'base_torque': 42},
    'M002': {'type': 'M', 'base_rpm': 1600, 'base_torque': 38},
    'M003': {'type': 'H', 'base_rpm': 1800, 'base_torque': 55},
    'M004': {'type': 'M', 'base_rpm': 1500, 'base_torque': 45},
    'M005': {'type': 'L', 'base_rpm': 1350, 'base_torque': 40},
}
READINGS_PER_MACHINE = 2000   # 2,000 × 5 machines = 10,000 total rows
MAINTENANCE_THRESHOLD = 220   # Tool wear (min) before maintenance reset

# ─── Generation ──────────────────────────────────────────────────────────────
rows = []
start_time = datetime(2024, 1, 1, 8, 0, 0)

for machine_id, cfg in MACHINES.items():
    # Each machine starts with slightly different baseline conditions
    base_air   = np.random.uniform(297, 303)
    base_proc  = base_air + np.random.uniform(9, 11)
    base_rpm   = cfg['base_rpm']
    base_torque = cfg['base_torque']
    tool_wear  = np.random.uniform(0, 30)      # random starting wear
    timestamp  = start_time

    # Slow sinusoidal drift to simulate shift-level temperature changes
    for i in range(READINGS_PER_MACHINE):

        # ── Tool wear progression ──────────────────────────────────────────
        tool_wear += np.random.uniform(0.05, 0.45)
        if tool_wear >= MAINTENANCE_THRESHOLD:
            tool_wear = np.random.uniform(0, 10)   # maintenance resets it

        wear_ratio = tool_wear / MAINTENANCE_THRESHOLD   # 0.0 → 1.0

        # ── Ambient drift (sinusoidal shift pattern) ───────────────────────
        shift_drift = 1.5 * np.sin(2 * np.pi * i / 480)   # ~8-hour cycle

        # ── Sensor values degrade with wear ───────────────────────────────
        air_temp     = base_air   + shift_drift + wear_ratio * 6  + np.random.normal(0, 0.4)
        process_temp = base_proc  + shift_drift + wear_ratio * 9  + np.random.normal(0, 0.3)
        rpm          = base_rpm               - wear_ratio * 180  + np.random.normal(0, 12)
        torque       = base_torque            + wear_ratio * 22   + np.random.normal(0, 1.8)

        # ── Realistic multi-cause failure model ───────────────────────────
        # Each failure mode contributes independently (mirrors real ISO 13849)
        twf = 1.0 if tool_wear >= MAINTENANCE_THRESHOLD * 0.95 else 0.0   # Tool Wear Failure
        hdf = 1.0 if (process_temp - air_temp) < 8.5 else 0.0             # Heat Dissipation Failure
        pwf = 1.0 if (rpm * torque / 9550) < 3.5 or (rpm * torque / 9550) > 11 else 0.0  # Power Failure
        osf = 1.0 if torque * rpm > 65 * 2500 else 0.0                    # Overstrain Failure

        # Probabilistic base failure scaled by wear
        prob_failure = (
            0.04 * wear_ratio**2 +
            0.03 * max(0, (torque - 60) / 20) +
            0.02 * max(0, (air_temp - 308) / 6) +
            0.01 * (twf + hdf + pwf + osf)
        )
        prob_failure = min(prob_failure, 0.95)
        machine_failure = int(np.random.random() < prob_failure)

        rows.append({
            'Machine_ID'            : machine_id,
            'Machine_Type'          : cfg['type'],
            'Timestamp'             : timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'Air_Temperature_K'     : round(air_temp, 2),
            'Process_Temperature_K' : round(process_temp, 2),
            'Rotational_Speed_RPM'  : round(max(rpm, 800), 2),
            'Torque_Nm'             : round(max(torque, 10), 2),
            'Tool_Wear_Min'         : round(tool_wear, 2),
            'TWF'                   : int(twf),
            'HDF'                   : int(hdf),
            'PWF'                   : int(pwf),
            'OSF'                   : int(osf),
            'Machine_Failure'       : machine_failure,
        })
        timestamp += timedelta(minutes=2)

# ─── Save ─────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df.to_csv('multi_machine_data.csv', index=False)

print("=" * 50)
print(f"  Dataset saved → multi_machine_data.csv")
print(f"  Total rows    : {len(df):,}")
print(f"  Machines      : {df['Machine_ID'].nunique()}")
print(f"  Failure rate  : {df['Machine_Failure'].mean():.2%}")
print("=" * 50)
print(df.groupby('Machine_ID')['Machine_Failure'].agg(['count','sum','mean']).rename(
    columns={'count':'Readings','sum':'Failures','mean':'Failure Rate'}))
