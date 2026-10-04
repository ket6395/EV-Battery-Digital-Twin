# 🔋 EV Battery Digital Twin

## Data-Driven Digital Twin for EV Battery State of Health Monitoring, Degradation Tracking and Remaining Useful Life Prediction

A data-driven, sequential Digital Twin prototype for monitoring the health of an Electric Vehicle (EV) battery using recorded battery-cycle data.

The project combines cycle-level battery feature engineering, machine-learning-based State of Health (SOH) estimation, sequential Digital Twin state updating, degradation-rate estimation, End-of-Life (EOL) prediction, Remaining Useful Life (RUL) estimation, and an interactive Streamlit dashboard.

> **Current scope:** This is a sequential/offline Digital Twin prototype. Recorded battery-cycle observations are replayed chronologically to simulate new observations. The current system is not connected to a live EV battery, physical BMS, or CAN network.

---

# Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Motivation](#3-motivation)
4. [Project Objectives](#4-project-objectives)
5. [Background](#5-background)
6. [Scope of the Current Project](#6-scope-of-the-current-project)
7. [Overall System Architecture](#7-overall-system-architecture)
8. [Detailed Architecture](#8-detailed-architecture)
9. [End-to-End Data Flow](#9-end-to-end-data-flow)
10. [Dataset](#10-dataset)
11. [Feature Engineering](#11-feature-engineering)
12. [Machine Learning Model](#12-machine-learning-model)
13. [Digital Twin Formulation](#13-digital-twin-formulation)
14. [Detailed Sequential Digital Twin Workflow](#14-detailed-sequential-digital-twin-workflow)
15. [Mathematical Formulation](#15-mathematical-formulation)
16. [Pseudocode](#16-pseudocode)
17. [Sequence 5 Experiment](#17-sequence-5-experiment)
18. [Validation Methodology](#18-validation-methodology)
19. [Results](#19-results)
20. [Result Interpretation](#20-result-interpretation)
21. [Important Evaluation Caveat](#21-important-evaluation-caveat)
22. [Streamlit Dashboard](#22-streamlit-dashboard)
23. [Repository Structure](#23-repository-structure)
24. [File-by-File Explanation](#24-file-by-file-explanation)
25. [Technologies Used](#25-technologies-used)
26. [Installation](#26-installation)
27. [Running the Dashboard](#27-running-the-dashboard)
28. [Running the Notebook](#28-running-the-notebook)
29. [Reproducing the Workflow](#29-reproducing-the-workflow)
30. [Understanding the Results](#30-understanding-the-results)
31. [Limitations](#31-limitations)
32. [Current Architecture vs Future Architecture](#32-current-architecture-vs-future-architecture)
33. [Future Real-Time BMS Architecture](#33-future-real-time-bms-architecture)
34. [Future Work](#34-future-work)
35. [Automotive and EV Relevance](#35-automotive-and-ev-relevance)
36. [Technical Skills Demonstrated](#36-technical-skills-demonstrated)
37. [Project Status](#37-project-status)
38. [Conclusion](#38-conclusion)
39. [Author](#39-author)

---

# 1. Project Overview

Battery health is one of the most important factors affecting the reliability, usable lifetime, performance, and maintenance requirements of an Electric Vehicle.

A battery does not remain electrically identical throughout its life. As it experiences repeated operating cycles, its measurable behavior changes. Voltage characteristics, current behavior, and temperature response can provide information about the condition of the battery.

A Battery Management System (BMS) can acquire these measurements, but a monitoring system that only reports instantaneous measurements does not provide a complete picture of how the battery is degrading.

This project develops a **data-driven Digital Twin prototype for EV battery health monitoring**.

The system uses recorded battery-cycle data and follows the battery sequentially. For each cycle, the system:

1. Extracts the required battery features.
2. Estimates SOH using a trained Extra Trees regression model.
3. Adds the new estimated SOH to the Digital Twin history.
4. Uses recent SOH history to estimate the degradation rate.
5. Estimates the future EOL cycle using a selected SOH threshold.
6. Calculates the remaining useful life in cycles.
7. Assigns a battery health status.
8. Makes the updated state available to the monitoring dashboard.

The central idea is therefore not simply:

```text
Battery data → Machine Learning → SOH
```

but:

```text
Battery cycle data
        ↓
Feature extraction
        ↓
SOH estimation
        ↓
Digital Twin state update
        ↓
Historical SOH trajectory
        ↓
Degradation estimation
        ↓
EOL prediction
        ↓
RUL estimation
        ↓
Battery health monitoring
```

The machine-learning model is the **SOH estimation component**. The Digital Twin is the **sequential state/history and prognostics layer** built around that estimator.

---

# 2. Problem Statement

The problem addressed by this project is the development of a data-driven framework capable of monitoring the changing health of an EV battery from battery-cycle measurements.

A single-cycle SOH estimator can answer:

> "What is the estimated SOH for this battery observation?"

However, a useful battery-health monitoring system should also answer:

> "How is the battery health changing?"

and:

> "Based on its recent degradation, how many cycles may remain before the selected EOL condition is reached?"

These questions require a sequential representation of the battery.

Therefore, the project addresses four connected tasks:

### Task 1 — SOH Estimation

Estimate battery State of Health from measured electrical and thermal characteristics.

### Task 2 — Sequential State Tracking

Maintain a history of the estimated battery condition as new cycles are processed.

### Task 3 — Degradation Estimation

Estimate the recent rate at which SOH is changing.

### Task 4 — Prognostics

Use the estimated degradation trend to estimate EOL and RUL.

The final system therefore combines **estimation, state tracking, degradation analysis, and prognostics**.

---

# 3. Motivation

A machine-learning model can provide predictions for individual observations, but battery health is fundamentally a time-dependent quantity.

Consider a sequence:

```text
Cycle 453
Cycle 454
Cycle 455
...
Cycle 714
```

The battery condition at one cycle should be interpreted in the context of previous observations.

For example, an SOH estimate of 92% means something different if the recent history is:

```text
94% → 93% → 92%
```

compared with:

```text
92% → 92.1% → 92%
```

The first case indicates a stronger downward trend, while the second represents relatively stable behavior over that local window.

The Digital Twin layer therefore provides a mechanism for retaining and analyzing recent battery history rather than treating every battery-cycle observation as an isolated prediction.

---

# 4. Project Objectives

The project was developed with the following objectives.

## 4.1 Process Battery-Cycle Data

Convert battery measurements into a structured cycle-level dataset suitable for machine learning and sequential analysis.

## 4.2 Develop Battery Features

Represent each battery cycle using statistical voltage, current, and temperature characteristics.

## 4.3 Estimate Battery SOH

Use an Extra Trees regression model to estimate SOH from the selected features.

## 4.4 Build a Sequential Digital Twin

Maintain a history containing battery-cycle information and estimated health.

## 4.5 Estimate Degradation

Use recent SOH estimates to obtain a local degradation rate.

## 4.6 Estimate EOL

Predict the cycle at which SOH reaches the project-defined 80% threshold.

## 4.7 Estimate RUL

Calculate the number of cycles remaining between the current cycle and predicted EOL.

## 4.8 Provide Visualization

Develop an interactive Streamlit interface through which the sequential battery state can be explored.

---

# 5. Background

## 5.1 EV Battery Health Monitoring

An EV battery is subjected to repeated charging and discharging operations. Its behavior can change as it ages.

Measurements such as:

- voltage,
- current,
- temperature,
- cycle number,
- energy,
- capacity/SOH

can be used to characterize the battery.

In a practical EV system, these measurements would typically be acquired by sensors and processed by the BMS.

In the present project, the same type of battery-cycle information is represented using recorded historical data.

## 5.2 State of Charge

State of Charge (SOC) describes the current charge level of the battery.

Conceptually:

```text
SOC → How much charge is currently available?
```

SOC and SOH are different concepts.

## 5.3 State of Health

State of Health (SOH) represents the health or condition of the battery relative to a reference condition.

A common conceptual definition is:

```text
SOH (%) =
Available Capacity
----------------- × 100
Rated Capacity
```

In this project, SOH is used as a continuous regression target.

The machine-learning model estimates:

```text
Estimated SOH (%)
```

for each battery-cycle observation.

## 5.4 Remaining Useful Life

Remaining Useful Life (RUL) represents the remaining useful operating life before a selected end-of-life criterion is reached.

In this project:

```text
RUL → Remaining battery cycles
```

The project uses:

```text
80% SOH
```

as its EOL criterion.

This is a project-defined threshold and is not claimed to be universal for every battery chemistry or application.

## 5.5 Battery Degradation

Battery degradation can manifest as changes in:

- available capacity,
- voltage response,
- current behavior,
- thermal behavior,
- internal resistance,
- charge/discharge characteristics.

The current project does not explicitly model individual electrochemical mechanisms. Instead, it uses measured cycle-level behavior to estimate SOH and then analyzes the resulting SOH trajectory.

## 5.6 Digital Twin

A Digital Twin is a computational representation of a physical system that evolves according to information associated with that system.

For this project, the conceptual physical object is:

```text
EV Battery
```

and the virtual representation contains:

```text
Cycle information
Estimated SOH
SOH history
Degradation rate
Predicted EOL
RUL
Health status
```

The current implementation does not receive live measurements directly from a physical battery. Recorded observations are replayed sequentially.

Therefore, the technically accurate description is:

> **Data-driven, sequential/offline Digital Twin prototype for EV battery health monitoring.**

---

# 6. Scope of the Current Project

The present project includes:

- processed battery-cycle data,
- cycle-level feature engineering,
- Extra Trees SOH estimation,
- sequential Digital Twin updating,
- local degradation-rate estimation,
- EOL prediction,
- RUL estimation,
- battery-health classification,
- Streamlit visualization.

The present project does **not** include:

- direct physical EV battery connection,
- live BMS communication,
- CAN acquisition,
- real-time embedded deployment,
- hardware-in-the-loop validation,
- cloud-connected Digital Twin,
- production battery prognostics.

These are future extension areas.

---

# 7. Overall System Architecture

```text
                         EV BATTERY DATA
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Battery-Cycle Dataset   │
                  │                         │
                  │ Voltage                 │
                  │ Current                 │
                  │ Temperature             │
                  │ Cycle                   │
                  │ SOH                     │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Feature Engineering     │
                  │                         │
                  │ Voltage Statistics      │
                  │ Current Statistics      │
                  │ Temperature Statistics  │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │ Extra Trees Regressor   │
                  │                         │
                  │ SOH Estimation          │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │       DIGITAL TWIN      │
                  │                         │
                  │ Current Cycle           │
                  │ Estimated SOH           │
                  │ SOH History             │
                  │ Degradation Rate        │
                  │ Predicted EOL           │
                  │ RUL                     │
                  │ Health Status           │
                  └────────────┬────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌────────────────────┐      ┌────────────────────┐
       │ Degradation Model  │      │ Health Monitoring  │
       │ Linear Regression  │      │                    │
       └──────────┬─────────┘      └────────────────────┘
                  │
                  ▼
       ┌────────────────────────┐
       │ EOL / RUL Estimation   │
       │ EOL threshold = 80%    │
       └────────────┬───────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │ Streamlit Dashboard    │
       │ SOH / RUL / EOL        │
       │ Battery Signals        │
       └────────────────────────┘
```

---

# 8. Detailed Architecture

## 8.1 Data Layer

The data layer contains the processed battery-cycle observations.

The main processed dataset is:

```text
data/ml_dataset.csv
```

Each row represents a cycle-level observation.

The data layer contains:

- sequence identifier,
- cycle number,
- voltage statistics,
- current statistics,
- temperature statistics,
- duration,
- energy,
- SOH,
- local cycle information.

---

## 8.2 Feature Engineering Layer

The final inference model uses 13 features:

```text
1.  voltage_mean
2.  voltage_std
3.  voltage_min
4.  voltage_max
5.  voltage_range
6.  current_mean
7.  current_std
8.  current_min
9.  current_max
10. temperature_mean
11. temperature_std
12. temperature_max
13. temperature_rise
```

The purpose of this layer is to convert the cycle-level electrical and thermal behavior into a compact numerical representation suitable for the trained model.

---

## 8.3 SOH Estimation Layer

The SOH estimation stage is:

```text
13-feature vector
       ↓
Extra Trees Regressor
       ↓
Estimated SOH
```

The prediction is clipped to:

```text
0 ≤ SOH ≤ 100
```

before being incorporated into the Digital Twin.

---

## 8.4 Digital Twin Layer

The Digital Twin maintains the evolving battery state.

The state contains:

```text
cycle
estimated_SOH
degradation_rate
predicted_EOL_cycle
RUL
health_status
history
```

The history is important because degradation is estimated from previous SOH estimates.

The Digital Twin therefore adds temporal context to the machine-learning estimator.

---

## 8.5 Degradation Layer

The framework selects recent estimated SOH observations.

The main sequential update uses:

```text
trend_window = 20
```

A Linear Regression model is fitted:

```text
SOH = a × cycle + b
```

where:

```text
a = local degradation rate
b = intercept
```

For a decreasing SOH trajectory:

```text
a < 0
```

---

## 8.6 EOL and RUL Layer

The project uses:

```text
EOL = 80% SOH
```

When the current SOH is above 80% and the local degradation rate is negative, the future EOL cycle is estimated.

The RUL is:

```text
RUL =
Predicted EOL Cycle - Current Cycle
```

If the required conditions are not satisfied, EOL/RUL can remain unavailable rather than producing an unsupported prediction.

---

## 8.7 Visualization Layer

The Streamlit dashboard provides:

- cycle navigation,
- current cycle,
- estimated SOH,
- battery state values,
- degradation rate,
- EOL,
- RUL,
- health status,
- SOH plots,
- temperature plots,
- voltage/current plots.

The dashboard is the visualization and interaction layer; it is not itself the Digital Twin.

---

# 9. End-to-End Data Flow

```text
Battery-cycle data
        ↓
Select sequence
        ↓
Sort by cycle
        ↓
Create initial history
        ↓
Select next cycle
        ↓
Extract 13 features
        ↓
Extra Trees SOH prediction
        ↓
Append estimated SOH
        ↓
Select recent history
        ↓
Linear degradation fit
        ↓
Estimate degradation rate
        ↓
Estimate EOL
        ↓
Estimate RUL
        ↓
Assign health status
        ↓
Store updated Digital Twin state
        ↓
Move to next cycle
        ↓
Repeat
```

The important point is that the updated history from one cycle is used when processing the next cycle.

---

# 10. Dataset

## 10.1 Dataset Size

The processed dataset contains:

```text
408 rows
19 columns
5 sequences
```

Sequence lengths:

| Sequence | Samples |
|---|---:|
| 1 | 138 |
| 2 | 69 |
| 3 | 69 |
| 4 | 25 |
| 5 | 107 |

Sequence 5 contains:

```text
107 observations
Cycle range: 453–714
```

## 10.2 Dataset Columns

```text
sequence_id
cycle

voltage_mean
voltage_std
voltage_min
voltage_max
voltage_range

current_mean
current_std
current_min
current_max

temperature_mean
temperature_std
temperature_max
temperature_rise

duration_s
energy_Wh

SOH
local_cycle
```

The complete dataset therefore contains more columns than the final model feature vector.

## 10.3 Battery Sequences

The five sequences provide separate groups of battery-cycle observations.

Sequence 5 is used for the main sequential Digital Twin experiment because it provides 107 ordered observations, allowing an initial history and a subsequent sequential prediction region.

---

# 11. Feature Engineering

## 11.1 Voltage Features

### `voltage_mean`

Average voltage during the cycle.

### `voltage_std`

Standard deviation of voltage during the cycle.

### `voltage_min`

Minimum observed voltage.

### `voltage_max`

Maximum observed voltage.

### `voltage_range`

```text
voltage_range =
voltage_max - voltage_min
```

These five features summarize the level and variation of voltage behavior.

---

## 11.2 Current Features

The current features are:

```text
current_mean
current_std
current_min
current_max
```

They describe average current, current variation, and the observed current limits.

---

## 11.3 Temperature Features

The thermal features are:

```text
temperature_mean
temperature_std
temperature_max
temperature_rise
```

They describe the average temperature, temperature variation, peak temperature, and temperature increase.

---

## 11.4 Additional Features

The processed dataset also contains:

```text
duration_s
energy_Wh
```

These remain available in the dataset but are not part of the final 13-feature Extra Trees inference vector.

---

## 11.5 Final Model Feature Set

The exact model input is:

```text
[
    voltage_mean,
    voltage_std,
    voltage_min,
    voltage_max,
    voltage_range,
    current_mean,
    current_std,
    current_min,
    current_max,
    temperature_mean,
    temperature_std,
    temperature_max,
    temperature_rise
]
```

The ordering is stored in:

```text
models/feature_columns.pkl
```

This is important because a trained model expects a consistent feature representation.

---

# 12. Machine Learning Model

## 12.1 Why SOH Is a Regression Problem

SOH is continuous.

For example:

```text
97.3%
94.6%
91.2%
87.8%
```

are continuous health values.

Therefore, the problem is treated as regression rather than classification.

---

## 12.2 Extra Trees Regressor

Extra Trees is an ensemble of randomized decision trees.

For this project, it is used to model the nonlinear mapping between:

```text
Voltage
Current
Temperature
```

and:

```text
SOH
```

The ensemble combines predictions from multiple trees.

The trained model is stored as:

```text
models/extra_trees_model.pkl
```

---

## 12.3 Model Input

For cycle `k`:

```text
x_k ∈ R^13
```

contains the 13 features listed above.

---

## 12.4 Model Output

The estimator produces:

```text
SOH_hat(k) = f_ET(x_k)
```

and the implementation applies:

```text
SOH_hat(k) =
clip(SOH_hat(k), 0, 100)
```

---

## 12.5 Model Artifacts

### `extra_trees_model.pkl`

Contains the trained Extra Trees model.

### `feature_columns.pkl`

Contains the exact feature list/order used during inference.

These two files allow the dashboard to use the trained model without retraining it.

---

# 13. Digital Twin Formulation

## 13.1 Digital Twin State

At cycle `k`, the Digital Twin can be represented as:

```text
DT_k = {
    cycle,
    estimated_SOH,
    degradation_rate,
    predicted_EOL_cycle,
    RUL,
    health_status,
    history
}
```

The history contains the observations processed so far.

---

## 13.2 SOH Estimation

For a new cycle:

```text
x_k
 ↓
f_ET
 ↓
SOH_hat(k)
```

The prediction is constrained to 0–100%.

---

## 13.3 History Update

The current row is copied and an `estimated_SOH` field is added.

Then:

```text
H_new =
H_old + current observation
```

This updated history becomes the state used in subsequent degradation calculations.

---

## 13.4 Degradation Estimation

The latest observations are selected.

For the current update configuration:

```text
N = 20
```

A linear model is fitted:

```text
SOH_hat = a × cycle + b
```

The coefficient `a` is interpreted as:

```text
degradation_rate
```

with units of approximately:

```text
% SOH per cycle
```

---

## 13.5 EOL Estimation

Let:

```text
SOH_EOL = 80
```

and:

```text
a < 0
```

Then:

```text
80 =
SOH_current + a × Δk
```

Therefore:

```text
Δk =
(SOH_current - 80) / |a|
```

and:

```text
k_EOL =
k_current + Δk
```

---

## 13.6 RUL Estimation

The remaining useful life is:

```text
RUL =
max(0, k_EOL - k_current)
```

---

## 13.7 Health Status

The current thresholds are:

```text
SOH ≥ 90%       → Healthy
80% ≤ SOH < 90% → Degrading
SOH < 80%       → Severely degraded
```

---

# 14. Detailed Sequential Digital Twin Workflow

## 14.1 Selecting a Battery Sequence

Sequence 5 is selected and sorted by cycle number.

```text
sequence_id = 5
cycles = 453–714
samples = 107
```

---

## 14.2 Initial History

The first:

```text
40 observations
```

are used as the initial Digital Twin history.

The Extra Trees model estimates SOH for these initial observations.

This creates the starting history required for subsequent degradation analysis.

---

## 14.3 Processing a New Cycle

The first observation after initialization is processed.

The system:

1. selects the current row,
2. extracts the model features,
3. predicts SOH,
4. adds estimated SOH to the row,
5. appends the row to the history.

---

## 14.4 Updating the Twin

The updated history is used to calculate:

```text
degradation_rate
predicted_EOL_cycle
RUL
health_status
```

The function returns the updated history.

---

## 14.5 Moving to the Next Cycle

The updated history becomes the previous history for the next iteration.

The process continues:

```text
Cycle k
   ↓
Update
   ↓
Cycle k+1
   ↓
Update
   ↓
Cycle k+2
   ↓
...
```

For Sequence 5:

```text
107 total observations
40 initial observations
67 sequentially processed observations
```

---

# 15. Mathematical Formulation

Let the 13-dimensional feature vector for cycle `k` be:

```text
x_k =
[
v_mean,
v_std,
v_min,
v_max,
v_range,
i_mean,
i_std,
i_min,
i_max,
T_mean,
T_std,
T_max,
T_rise
]
```

The SOH model is:

```text
SOH_hat_k = f_ET(x_k)
```

where `f_ET` is the trained Extra Trees regressor.

The bounded prediction is:

```text
SOH_hat_k = min(100, max(0, SOH_hat_k))
```

For the recent Digital Twin history:

```text
SOH_hat_i ≈ a c_i + b
```

where `c_i` is cycle number.

The degradation rate is:

```text
dSOH/dcycle ≈ a
```

For the EOL threshold:

```text
SOH_EOL = 80
```

the estimated number of cycles to EOL is:

```text
Δc =
(SOH_hat_k - 80) / |a|
```

and:

```text
c_EOL = c_k + Δc
```

Finally:

```text
RUL_k = max(0, c_EOL - c_k)
```

The equations represent a local linear extrapolation of the recent SOH trend.

---

# 16. Pseudocode

```text
INPUT:
    sequence
    initial_history
    Extra Trees model
    feature_columns

FOR every future cycle:

    1. Extract 13 model features.

    2. Predict SOH using Extra Trees.

    3. Clip SOH to [0,100].

    4. Add estimated SOH to current observation.

    5. Append observation to Digital Twin history.

    6. Select the most recent trend_window observations.

    7. Fit:
           SOH = a × cycle + b

    8. Store:
           degradation_rate = a

    9. If:
           a < 0
           and SOH > 80

       calculate:
           EOL = cycle + (SOH - 80)/abs(a)

    10. Calculate:
           RUL = max(0, EOL - cycle)

    11. Determine:
           Healthy / Degrading / Severely degraded

    12. Save the current Digital Twin state.

    13. Use the updated history for the next cycle.
```

---

# 17. Sequence 5 Experiment

The primary sequential experiment uses Sequence 5.

```text
Sequence length = 107
Cycle range = 453–714
```

The first 40 observations are used to establish the initial Digital Twin history.

The remaining 67 observations are processed sequentially.

For each processed observation, the resulting record contains:

```text
cycle
estimated_SOH
actual_SOH
degradation_rate
predicted_EOL_cycle
RUL
health_status
```

This creates a sequential Digital Twin result table rather than a collection of independent predictions.

---

# 18. Validation Methodology

## 18.1 Why Validation Strategy Matters

Battery-cycle observations can be strongly correlated.

Adjacent observations from the same battery are likely to have similar operating and aging characteristics.

Therefore:

```text
Random train/test split
```

can be easier than:

```text
Future-cycle prediction
```

or:

```text
Completely unseen battery prediction
```

For this reason, the project considers several validation strategies.

---

## 18.2 Standard Evaluation

The standard Extra Trees evaluation produced:

| Metric | Result |
|---|---:|
| MAE | 2.730% |
| RMSE | 3.493% |
| R² | 0.954 |

This indicates strong performance under that evaluation setup.

However, it should not be treated as proof of generalization to completely unseen battery sequences.

---

## 18.3 Leave-One-Sequence-Out Validation

The LOSO results are:

| Test Sequence | MAE (%) | RMSE (%) | R² |
|---|---:|---:|---:|
| Sequence 1 | 16.918 | 18.921 | -0.686 |
| Sequence 2 | 10.293 | 10.724 | -2.251 |
| Sequence 3 | 4.462 | 5.370 | 0.303 |
| Sequence 4 | 7.586 | 7.628 | -1.171 |
| Sequence 5 | 8.867 | 11.073 | -1.930 |
| **Average** | **9.625** | **10.743** | **-1.147** |

This demonstrates that the model's performance varies significantly across sequences.

---

## 18.4 Temporal Validation

Sequence 5 was also evaluated chronologically:

```text
Training:
453–594

Testing:
595–714
```

Results:

| Metric | Result |
|---|---:|
| MAE | 11.640% |
| RMSE | 13.028% |
| R² | -3.048 |

This is a substantially more difficult prediction problem than standard random evaluation.

---

## 18.5 Sequential Digital Twin Validation

The Digital Twin experiment uses:

```text
Initial history = 40 observations
Sequential future observations = 67
```

The reported sequential evaluation was:

| Metric | Result |
|---|---:|
| MAE | 0.552% |
| RMSE | 0.758% |
| R² | 0.990 |

This result describes the evaluated Sequence 5 sequential Digital Twin setup.

---

# 19. Results

## Standard Extra Trees Evaluation

```text
MAE  = 2.730%
RMSE = 3.493%
R²   = 0.954
```

## Leave-One-Sequence-Out

```text
Average MAE  = 9.625%
Average RMSE = 10.743%
Average R²   = -1.147
```

## Temporal Sequence 5 Validation

```text
MAE  = 11.640%
RMSE = 13.028%
R²   = -3.048
```

## Sequential Digital Twin Evaluation

```text
MAE  = 0.552%
RMSE = 0.758%
R²   = 0.990
```

---

# 20. Result Interpretation

The different validation results demonstrate that battery-health model performance is highly dependent on how the data is divided.

The standard evaluation produces a low MAE, but LOSO and temporal validation are substantially more difficult.

This matters because a real EV battery-health system must eventually handle:

- future observations,
- different degradation trajectories,
- different batteries,
- changing operating conditions.

Therefore, the project does not interpret the standard 2.730% MAE as the complete picture of deployment performance.

The results instead show two things:

1. The current feature/model pipeline can estimate SOH effectively under some evaluation setups.
2. Generalization to unseen sequences and future data remains a major engineering challenge.

This is an important result rather than something to hide from the project documentation.

---

# 21. Important Evaluation Caveat

The final Digital Twin evaluation reports:

```text
MAE = 0.552%
RMSE = 0.758%
R² = 0.990
```

These values should be described specifically as results from the current Sequential Digital Twin evaluation on the selected Sequence 5 setup.

They should **not** be presented as proof of unbiased generalization to completely unseen batteries.

The project contains stronger stress tests, including LOSO and temporal validation, and those show substantially weaker performance.

A technically defensible project statement is:

> The sequential Digital Twin evaluation achieved MAE of 0.552%, RMSE of 0.758%, and R² of 0.990 on the evaluated Sequence 5 future-observation setup. Separate leave-one-sequence-out and temporal validation experiments indicate that generalization to unseen sequences and future data remains a significant challenge.

---

# 22. Streamlit Dashboard

## 22.1 Dashboard Purpose

The Streamlit dashboard provides an interactive interface for exploring the battery Digital Twin.

It allows the user to:

- select a sequence,
- move through battery cycles,
- inspect current battery values,
- observe estimated SOH,
- inspect degradation,
- inspect EOL/RUL,
- visualize historical behavior.

The dashboard is intended as the monitoring layer around the Digital Twin computation.

---

## 22.2 Dashboard Loading

The dashboard loads the repository's processed dataset and model artifacts.

The important files are:

```text
data/ml_dataset.csv
models/extra_trees_model.pkl
models/feature_columns.pkl
```

The current dashboard uses repository-relative paths rather than the original Google Colab `/content/...` directory structure.

---

## 22.3 Sequence Selection

The user can select a battery sequence.

The selected sequence is used to build the observed history shown by the dashboard.

---

## 22.4 Sequential Navigation

The dashboard includes navigation controls for moving through the selected battery-cycle sequence.

Conceptually:

```text
Previous Cycle ← Current Cycle → Next Cycle
```

A reset operation returns the interface to the beginning of the selected sequence.

---

## 22.5 Battery Health KPIs

The dashboard provides key indicators including:

### Current Cycle

The currently selected cycle.

### Estimated SOH

The current model-based SOH estimate.

### Estimated RUL

The current remaining-cycle estimate when available.

### Health Status

The current qualitative battery state.

---

## 22.6 Battery State Information

The dashboard provides cycle-level information such as:

- mean voltage,
- mean current,
- temperature,
- energy.

These values provide context for the health estimate.

---

## 22.7 Degradation and RUL

The dashboard also presents:

- degradation rate,
- predicted EOL cycle,
- RUL,
- health state.

This connects the current SOH estimate to the prognostics layer.

---

## 22.8 Graphs

The dashboard provides historical visualizations for:

- SOH,
- temperature,
- voltage,
- current.

The objective is to allow the user to understand both the current state and its evolution.

---

# 23. Repository Structure

```text
EV-Battery-Digital-Twin/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── ml_dataset.csv
│   └── README.md
│
├── images/
│   └── dashboard.png
│
├── models/
│   ├── extra_trees_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── EV_Battery_Digital_Twin.ipynb
│
└── results/
    ├── digital_twin_history.csv
    └── rul_summary.csv
```

---

# 24. File-by-File Explanation

## `README.md`

Main project documentation.

It explains the architecture, methodology, mathematical formulation, results, installation, limitations, and future work.

## `requirements.txt`

Contains the Python packages required to run the project:

```text
numpy
pandas
scikit-learn
streamlit
joblib
```

## `.gitignore`

Prevents temporary files, virtual environments, notebook checkpoints, secrets, raw archives, and other unwanted files from being committed.

Examples:

```text
__pycache__/
.venv/
.env
kaggle.json
*.zip
```

## `dashboard/app.py`

Contains the Streamlit monitoring interface.

It loads the dataset and trained model artifacts and provides sequential battery-cycle visualization.

## `data/ml_dataset.csv`

Processed cycle-level battery dataset.

## `data/README.md`

Documentation for the processed data directory.

## `models/extra_trees_model.pkl`

Serialized Extra Trees SOH estimator.

## `models/feature_columns.pkl`

Exact feature names/order required for model inference.

## `notebooks/EV_Battery_Digital_Twin.ipynb`

Main development and execution notebook.

It contains the Digital Twin update function, initialization, sequential simulation, RUL calculation, evaluation, and result generation.

## `results/digital_twin_history.csv`

Sequential Digital Twin results.

Important fields include:

```text
cycle
estimated_SOH
actual_SOH
degradation_rate
predicted_EOL_cycle
RUL
health_status
```

## `results/rul_summary.csv`

RUL-related results generated during the project.

## `images/dashboard.png`

Dashboard screenshot used in the project documentation.

---

# 25. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main implementation |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Scikit-learn | ML and regression |
| Extra Trees | SOH estimation |
| Linear Regression | Degradation trend |
| Joblib | Model serialization |
| Streamlit | Dashboard |
| Jupyter/Google Colab | Development |
| Git | Version control |
| GitHub | Repository |

---

# 26. Installation

## Clone

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd EV-Battery-Digital-Twin
```

## Create Environment

Windows:

```bash
python -m venv .venv
```

Activate:

```bash
.venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 27. Running the Dashboard

From the repository root:

```bash
streamlit run dashboard/app.py
```

The dashboard expects the repository structure to remain:

```text
dashboard/app.py
data/ml_dataset.csv
models/extra_trees_model.pkl
models/feature_columns.pkl
```

The model does not need to be retrained to run the dashboard.

---

# 28. Running the Notebook

Open:

```text
notebooks/EV_Battery_Digital_Twin.ipynb
```

using Jupyter Notebook, JupyterLab, or a compatible notebook environment.

The notebook provides the development and execution workflow for the Digital Twin.

---

# 29. Reproducing the Workflow

A clean reproduction follows:

```text
1. Load ml_dataset.csv
2. Load extra_trees_model.pkl
3. Load feature_columns.pkl
4. Verify model features
5. Select Sequence 5
6. Sort by cycle
7. Create initial 40-observation history
8. Estimate initial SOH
9. Start sequential processing
10. Predict SOH for next cycle
11. Append estimate to history
12. Estimate degradation rate
13. Estimate EOL
14. Estimate RUL
15. Determine health status
16. Save Digital Twin state
17. Repeat until final cycle
18. Evaluate SOH predictions
19. Save results
```

---

# 30. Understanding the Results

## Estimated SOH

The machine-learning model's estimate for the current cycle.

## Actual SOH

The reference SOH present in the dataset and used for evaluation.

## Degradation Rate

The slope obtained from the recent SOH-versus-cycle linear regression.

## Predicted EOL

The estimated cycle where the projected SOH reaches 80%.

## RUL

The estimated number of cycles from the current cycle to predicted EOL.

## Health Status

A simple qualitative classification based on SOH thresholds.

---

# 31. Limitations

## 31.1 Offline Input

The project currently replays recorded data rather than receiving live battery measurements.

## 31.2 No Physical Battery

There is no direct physical battery connection.

## 31.3 No BMS Integration

No live BMS interface is currently implemented.

## 31.4 No CAN

CAN communication is a future extension.

## 31.5 Limited Dataset

The processed dataset contains 408 observations across five sequences.

## 31.6 Generalization

LOSO and temporal validation show that generalization remains difficult.

## 31.7 Simplified Degradation Model

A local linear trend is used. Actual battery degradation can be nonlinear.

## 31.8 Point RUL Estimate

The current implementation does not provide probabilistic uncertainty for RUL.

## 31.9 EOL Definition

80% SOH is a project-defined threshold, not a universal battery EOL criterion.

---

# 32. Current Architecture vs Future Architecture

| Component | Current | Future |
|---|---|---|
| Battery input | Recorded cycle data | Live BMS |
| Acquisition | Offline replay | Online acquisition |
| Communication | Dataset | CAN/UART |
| SOH | Extra Trees | Improved/hybrid estimator |
| Digital Twin | Sequential/offline | Online |
| Degradation | Linear local trend | Advanced model |
| EOL | 80% SOH | Application-specific |
| RUL | Trend extrapolation | Uncertainty-aware |
| Hardware | None | HIL/embedded |
| Dashboard | Streamlit | Real-time monitoring |

---

# 33. Future Real-Time BMS Architecture

A future implementation can replace recorded data with a physical BMS.

```text
                  PHYSICAL EV BATTERY
                          │
                          ▼
                  ┌───────────────┐
                  │ Sensors / BMS │
                  │               │
                  │ Voltage       │
                  │ Current       │
                  │ Temperature   │
                  └───────┬───────┘
                          │
                          ▼
                     CAN / UART
                          │
                          ▼
                ┌──────────────────┐
                │ Data Acquisition │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Feature          │
                │ Extraction       │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ SOH Estimator    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Digital Twin     │
                │ State Update     │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Degradation      │
                │ + EOL + RUL      │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Monitoring UI    │
                └──────────────────┘
```

This is a **future architecture**, not a claim about the current repository.

---

# 34. Future Work

## 34.1 Live BMS Integration

Use real BMS signals instead of recorded cycle data.

Potential inputs:

- cell voltage,
- pack voltage,
- current,
- temperature,
- SOC,
- cycle count.

## 34.2 CAN Integration

Receive battery information over automotive CAN.

## 34.3 Online Digital Twin

Update the virtual battery state whenever a new measurement arrives.

## 34.4 Improved Degradation Modeling

Investigate nonlinear and condition-dependent degradation.

## 34.5 Improved RUL

Develop uncertainty-aware and potentially probabilistic RUL estimation.

## 34.6 Multi-Battery Validation

Evaluate generalization across independent batteries rather than only within sequences.

## 34.7 Hardware Validation

Validate the framework on BMS hardware, battery emulators, HIL systems, or a suitable physical battery test platform.

## 34.8 Embedded Deployment

Investigate deployment of the SOH estimator and feature-processing pipeline on embedded controllers.

---

# 35. Automotive and EV Relevance

The project is relevant to several automotive areas.

### EV Battery Monitoring

The system directly targets battery health estimation.

### Battery Management Systems

The features correspond to typical battery electrical and thermal measurements.

### Predictive Maintenance

RUL estimation can eventually support maintenance decisions.

### Automotive Communication

Future CAN integration can connect the Digital Twin with vehicle electronics.

### Embedded Systems

The estimation pipeline can potentially be optimized for embedded deployment.

### Cyber-Physical Systems

The project represents a physical battery computationally and updates its virtual state using measured data.

---

# 36. Technical Skills Demonstrated

## Programming

- Python
- NumPy
- Pandas

## Machine Learning

- Regression
- Extra Trees
- Feature engineering
- Model evaluation
- LOSO validation
- Temporal validation

## Battery Systems

- SOH
- RUL
- EOL
- Degradation tracking
- Voltage/current/temperature analysis

## Digital Twin

- Sequential state update
- Historical state management
- Degradation monitoring
- Prognostics

## Software Engineering

- Git
- GitHub
- Repository organization
- Model artifact management
- Streamlit application development

---

# 37. Project Status

## Completed

- [x] Battery-cycle dataset processing
- [x] Feature engineering
- [x] SOH regression
- [x] Extra Trees model
- [x] Model serialization
- [x] Feature configuration
- [x] Digital Twin update function
- [x] Initial history
- [x] Sequential simulation
- [x] Degradation estimation
- [x] EOL estimation
- [x] RUL estimation
- [x] Health classification
- [x] Streamlit dashboard
- [x] Results files
- [x] GitHub repository structure
- [x] Documentation

## Planned

- [ ] Live BMS
- [ ] CAN communication
- [ ] Online Digital Twin
- [ ] Hardware validation
- [ ] Multi-battery validation
- [ ] Improved RUL
- [ ] Uncertainty estimation
- [ ] Embedded deployment

---

# 38. Conclusion

This project implements an end-to-end software prototype for a data-driven EV battery Digital Twin.

The system begins with recorded battery-cycle measurements and converts them into a structured set of voltage, current, and temperature features.

An Extra Trees regression model estimates battery SOH.

The SOH estimate is then incorporated into a sequential Digital Twin history. As additional cycles are processed, the Digital Twin maintains an evolving representation of battery health.

Recent SOH history is analyzed using Linear Regression to estimate a local degradation rate. That rate is used to estimate the cycle at which SOH reaches the selected 80% EOL threshold, after which RUL is calculated.

The overall workflow is:

```text
Battery-cycle data
        ↓
Feature engineering
        ↓
Extra Trees SOH estimation
        ↓
Digital Twin history update
        ↓
Degradation estimation
        ↓
EOL prediction
        ↓
RUL estimation
        ↓
Health classification
        ↓
Streamlit monitoring
```

The project also demonstrates why validation methodology matters. Standard evaluation produces strong results, while leave-one-sequence-out and chronological validation expose significant generalization challenges.

This is particularly relevant for battery prognostics because a practical system must eventually work on future observations and independent batteries rather than only correlated observations from the same data distribution.

The current project therefore establishes a working foundation for:

- data-driven SOH estimation,
- sequential Digital Twin state tracking,
- degradation monitoring,
- EOL estimation,
- RUL prediction,
- interactive battery-health visualization.

The natural next stage is to replace the offline sequential data source with live BMS measurements and progress toward CAN integration, online Digital Twin operation, hardware validation, improved degradation modeling, and uncertainty-aware RUL prediction.

---

# 39. Author

## Ketan Bathla

**M.Tech — Cyber-Physical Systems**  
**Indian Institute of Technology Jodhpur**

---

> **EV Battery Digital Twin — From Battery-Cycle Data to SOH Estimation, Degradation Tracking and RUL Prediction.**
