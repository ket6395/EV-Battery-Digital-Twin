# Dataset

This folder contains the processed battery-cycle dataset used by the EV Battery Digital Twin project.

## File

`ml_dataset.csv`

The dataset contains cycle-level battery measurements and engineered features used for SOH estimation.

### Main columns

- `sequence_id` – battery sequence identifier
- `cycle` – battery cycle number
- `voltage_mean`
- `voltage_std`
- `voltage_min`
- `voltage_max`
- `voltage_range`
- `current_mean`
- `current_std`
- `current_min`
- `current_max`
- `temperature_mean`
- `temperature_std`
- `temperature_max`
- `temperature_rise`
- `duration_s`
- `energy_Wh`
- `SOH`
- `local_cycle`

The trained Extra Trees model uses the voltage, current, and temperature features listed in the project notebook.

## Note

This repository contains the processed dataset used for the project. Raw battery files are not included.