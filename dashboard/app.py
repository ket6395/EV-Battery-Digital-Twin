
import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression


# ============================================================
# PROJECT PATHS
# ============================================================

# Repository structure:
# EV-Battery-Digital-Twin/
# ├── data/
# ├── models/
# └── dashboard/
#     └── app.py

DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(DASHBOARD_DIR, ".."))

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "ml_dataset.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "extra_trees_model.pkl"
)

FEATURE_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "feature_columns.pkl"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EV Battery Digital Twin",
    page_icon="🔋",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    features = joblib.load(
        FEATURE_PATH
    )

    return model, features


df = load_dataset()

model, feature_columns = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🔋 EV Battery Digital Twin")

st.write(
    "Sequential simulation of an EV battery "
    "Digital Twin using recorded battery-cycle data."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Digital Twin Controls"
)

sequences = sorted(
    df["sequence_id"].unique()
)

selected_sequence = st.sidebar.selectbox(
    "Select Battery Sequence",
    sequences
)


# ============================================================
# SELECT SEQUENCE
# ============================================================

seq_df = (
    df[
        df["sequence_id"] == selected_sequence
    ]
    .sort_values("cycle")
    .reset_index(drop=True)
)

total_cycles = len(seq_df)


# ============================================================
# SESSION STATE
# ============================================================

state_key = (
    f"current_cycle_{selected_sequence}"
)

if state_key not in st.session_state:

    st.session_state[state_key] = 0


current_index = (
    st.session_state[state_key]
)


# ============================================================
# CONTROL BUTTONS
# ============================================================

col1, col2, col3 = st.sidebar.columns(3)


with col1:

    if st.button("◀"):

        current_index = max(
            0,
            current_index - 1
        )

        st.session_state[
            state_key
        ] = current_index


with col2:

    if st.button("Next ▶"):

        current_index = min(
            total_cycles - 1,
            current_index + 1
        )

        st.session_state[
            state_key
        ] = current_index


with col3:

    if st.button("Reset"):

        current_index = 0

        st.session_state[
            state_key
        ] = current_index


# ============================================================
# CURRENT BATTERY CYCLE
# ============================================================

current_row = seq_df.iloc[
    current_index
]

current_cycle = int(
    current_row["cycle"]
)


# ============================================================
# SOH ESTIMATION
# ============================================================

X_current = current_row[
    feature_columns
].to_frame().T

estimated_soh = float(
    model.predict(X_current)[0]
)

estimated_soh = float(
    np.clip(
        estimated_soh,
        0,
        100
    )
)


# ============================================================
# OBSERVED HISTORY
# ============================================================

observed_df = seq_df.iloc[
    :current_index + 1
].copy()


# Estimate SOH for observed history

X_history = observed_df[
    feature_columns
]

observed_df[
    "estimated_SOH"
] = np.clip(
    model.predict(X_history),
    0,
    100
)


# ============================================================
# DEGRADATION TREND
# ============================================================

trend_window = min(
    30,
    len(observed_df)
)

recent = observed_df.tail(
    trend_window
).copy()


recent[
    "SOH_smooth"
] = (
    recent["estimated_SOH"]
    .rolling(
        window=5,
        min_periods=1
    )
    .mean()
)


if len(recent) >= 5:

    trend_model = LinearRegression()

    trend_model.fit(
        recent[["cycle"]],
        recent["SOH_smooth"]
    )

    degradation_rate = float(
        trend_model.coef_[0]
    )

else:

    degradation_rate = np.nan


# ============================================================
# RUL CALCULATION
# ============================================================

EOL_THRESHOLD = 80.0

MIN_RATE = 0.05


if estimated_soh <= EOL_THRESHOLD:

    predicted_eol = current_cycle

    rul = 0.0


elif (
    np.isfinite(degradation_rate)
    and degradation_rate < 0
):

    effective_rate = min(
        degradation_rate,
        -MIN_RATE
    )

    predicted_eol = (
        current_cycle
        +
        (
            estimated_soh
            - EOL_THRESHOLD
        )
        /
        abs(effective_rate)
    )

    rul = max(
        0.0,
        predicted_eol - current_cycle
    )


else:

    predicted_eol = np.nan

    rul = np.nan


# ============================================================
# HEALTH STATUS
# ============================================================

if estimated_soh >= 90:

    health_status = "Healthy"

elif estimated_soh >= 80:

    health_status = "Degrading"

else:

    health_status = "EOL Reached"


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.subheader(
    f"Battery Sequence {selected_sequence}"
)


# ============================================================
# KPI CARDS
# ============================================================

k1, k2, k3, k4 = st.columns(4)


with k1:

    st.metric(
        "Current Cycle",
        current_cycle
    )


with k2:

    st.metric(
        "Estimated SOH",
        f"{estimated_soh:.2f}%"
    )


with k3:

    if np.isfinite(rul):

        st.metric(
            "Estimated RUL",
            f"{rul:.1f} cycles"
        )

    else:

        st.metric(
            "Estimated RUL",
            "N/A"
        )


with k4:

    st.metric(
        "Health Status",
        health_status
    )


# ============================================================
# CURRENT BATTERY STATE
# ============================================================

st.subheader(
    "Current Battery State"
)

b1, b2, b3, b4 = st.columns(4)


with b1:

    st.metric(
        "Mean Voltage",
        f"{current_row['voltage_mean']:.3f} V"
    )


with b2:

    st.metric(
        "Mean Current",
        f"{current_row['current_mean']:.3f} A"
    )


with b3:

    st.metric(
        "Temperature",
        f"{current_row['temperature_mean']:.2f} °C"
    )


with b4:

    st.metric(
        "Energy",
        f"{current_row['energy_Wh']:.3f} Wh"
    )


# ============================================================
# DIGITAL TWIN STATUS
# ============================================================

st.subheader(
    "Digital Twin Status"
)

d1, d2, d3 = st.columns(3)


with d1:

    if np.isfinite(degradation_rate):

        st.metric(
            "Degradation Rate",
            f"{degradation_rate:.4f} %/cycle"
        )

    else:

        st.metric(
            "Degradation Rate",
            "N/A"
        )


with d2:

    if np.isfinite(predicted_eol):

        st.metric(
            "Predicted EOL",
            f"{predicted_eol:.1f}"
        )

    else:

        st.metric(
            "Predicted EOL",
            "N/A"
        )


with d3:

    st.metric(
        "Observed Cycles",
        current_index + 1
    )


# ============================================================
# SOH GRAPH
# ============================================================

st.subheader(
    "SOH Tracking"
)

soh_plot = observed_df[
    [
        "cycle",
        "SOH",
        "estimated_SOH"
    ]
].set_index("cycle")


st.line_chart(
    soh_plot
)


# ============================================================
# TEMPERATURE GRAPH
# ============================================================

st.subheader(
    "Battery Temperature"
)

temperature_plot = observed_df[
    [
        "cycle",
        "temperature_mean"
    ]
].set_index("cycle")


st.line_chart(
    temperature_plot
)


# ============================================================
# VOLTAGE / CURRENT GRAPH
# ============================================================

st.subheader(
    "Electrical Measurements"
)

electrical_plot = observed_df[
    [
        "cycle",
        "voltage_mean",
        "current_mean"
    ]
].set_index("cycle")


st.line_chart(
    electrical_plot
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EV Battery Digital Twin | "
    "Extra Trees SOH estimation + "
    "degradation-based RUL simulation"
)
