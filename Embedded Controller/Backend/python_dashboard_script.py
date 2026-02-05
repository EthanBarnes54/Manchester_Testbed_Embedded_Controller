"""
#--------------- ESP32 Control and Monitoring Dashboard -------------------#

# A Dash-based web interface for real-time control and data visualization
# of the ESP32-based embedded control system. Communicates with the backend
# and  ML system to display live metrics of both the ion beam testbed and 
# RNN remote controller.

# Run locally in a new terminal and open http://127.0.0.1:8050 in a browser.
# ------------------------------------------------------------------------#
"""
from dash import Dash, dcc, html, Input, Output, State, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pandas as pd
import pkgutil, importlib.util
import numpy as np
import logging
import time
import os

log = logging.getLogger("Dashboard")

try:
    from python_RNN_Controller import propose_control_vector
except Exception as fault:
    log.warning(f"WARNING: Unable to contact the RNN - {fault}!")
    propose_control_vector = None

from python_Backend import (
    get_ml_metrics,
    Back_End_Controller,
    start_training_sweep,
    get_sweep_status,
    stop_training_sweep,
    get_model_info,
    set_window_update_time,
    set_online_learning_rate,
    save_model_parameters,
    compute_feature_importance,
)

# Workaround for Python 3.14 (pkgutil.find_loader has been removed)
if not hasattr(pkgutil, "find_loader"):
    pkgutil.find_loader = lambda name: importlib.util.find_spec(name)

# -------------------------------------------------------------------------
#                                Logging setup
# -------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",)

try:
    MODEL_INFO_BOOTSTRAP = get_model_info()
except Exception as fault:
    log.warning(f"WARNING: Unable to obtain model history - {fault}!")
    MODEL_INFO_BOOTSTRAP = {}

# -------------------------------------------------------------------------
#                                 Initialize app
# -------------------------------------------------------------------------

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
app = Dash(__name__, title="Manchester Ion Beam Testbed: Control Dashboard", assets_folder=ASSETS_DIR)
server = app.server

LAST_AUTO_TS = 0.0
# -------------------------------------------------------------------------
#                                     Layout
# -------------------------------------------------------------------------

def _control_tab():
    """Builds the main control dashboard layout."""
    
    return html.Div(
        children=[
            html.H2("ESP12-F Control Dashboard", style={"textAlign": "center"}),
            dcc.Graph(id="live-graph", style={"height": "60vh"}),

            html.Div(
                style={"marginTop": "1em"},
                children=[
                    html.H4("Pin Controls: 5 Pin Channels + Switch Timing"),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "1em"},
                        children=[
                            html.Div(
                                [
                                    html.Label("Squeeze Plate (V)"),
                                    dcc.Input(
                                        id="pwm1",
                                        type="text",
                                        min=0.0,
                                        max=2000.0,
                                        step=0.01,
                                        value="0.00",
                                        debounce=False,
                                        className="digit-nudge",
                                        style={"width": "140px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Ion Source (V)"),
                                    dcc.Input(
                                        id="pwm2",
                                        type="text",
                                        min=0.0,
                                        max=2000.0,
                                        step=0.01,
                                        value="0.00",
                                        debounce=False,
                                        className="digit-nudge",
                                        style={"width": "140px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Wein Filter (V)"),
                                    dcc.Input(
                                        id="pwm3",
                                        type="text",
                                        min=0.0,
                                        max=2000.0,
                                        step=0.01,
                                        value="0.00",
                                        debounce=False,
                                        className="digit-nudge",
                                        style={"width": "140px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Upper Cone (V)"),
                                    dcc.Input(
                                        id="pwm4",
                                        type="text",
                                        min=0.0,
                                        max=2000.0,
                                        step=0.01,
                                        value="0.00",
                                        debounce=False,
                                        className="digit-nudge",
                                        style={"width": "140px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Lower Cone (V)"),
                                    dcc.Input(
                                        id="pwm5",
                                        type="text",
                                        min=0.0,
                                        max=2000.0,
                                        step=0.01,
                                        value="0.00",
                                        debounce=False,
                                        className="digit-nudge",
                                        style={"width": "140px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Label("Switch Time (us)"),
                                    dcc.Input(
                                        id="switch-time-us",
                                        type="number",
                                        min=1,
                                        max=20,
                                        step=1,
                                        value=5,
                                        debounce=False,
                                        style={"width": "120px"},
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.Div(id="pins-ack", style={"marginTop": "0.5em", "fontWeight": "bold"}),
                ],
            ),

            html.Div(
                id="metrics-bar",
                style={
                    "display": "flex",
                    "justifyContent": "space-around",
                    "marginTop": "2em",
                    "padding": "1em",
                    "borderTop": "1px solid #ddd",
                    "color": "#333",
                    "gap": "1em",
                    "flexWrap": "wrap",
                },

                children=[
                    html.Div(
                        id="latest-voltage",
                        children="Live Voltage: -- V",
                        style={
                            "border": "1px solid #000",
                            "borderRadius": "6px",
                            "padding": "0.4em 0.8em",
                            "minWidth": "180px",
                            "textAlign": "center",
                            "fontWeight": "bold",
                        },
                    ),

                    html.Div(
                        id="num-points",
                        children="Sample Count: --",
                        style={
                            "border": "1px solid #000",
                            "borderRadius": "6px",
                            "padding": "0.4em 0.8em",
                            "minWidth": "180px",
                            "textAlign": "center",
                            "fontWeight": "bold",
                        },
                    ),

                    html.Div(
                        id="status",
                        children="OFFLINE",
                        style={"color": "red", "fontWeight": "bold"},
                    ),
                ],
            ),

            html.Div(id="pin-status", style={"display": "flex", "flexWrap": "wrap", "gap": "0.5em 1em", "marginTop": "0.5em"}),
            html.Div(
                style={"marginTop": "1.5em", "display": "flex", "gap": "1em", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={
                            "flex": "1 1 360px",
                            "padding": "1em",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "8px",
                            "background": "#fafafa",
                        },

                        children=[
                            html.Div(
                                children=[
                                    html.H4("RNN Training Control Panel", style={"margin": 0}),
                                    html.P(
                                        "Generate a new training dataset by sweeping over output pin voltages, then intiate a training cycle of the RNN controller.",
                                        style={"margin": "0.25em 0 0.75em 0", "color": "#555"},
                                    ),
                                ]
                            ),

                            html.Div(
                                style={
                                    "display": "grid",
                                    "gridTemplateColumns": "repeat(5, 1fr)",
                                    "gap": "0.75em",
                                    "alignItems": "end",
                                },

                                children=[
                                    html.Div(
                                        children=[
                                            html.Small("Min V"),
                                            dcc.Input(
                                                id="sweep-min",
                                                type="number",
                                                value=0.0,
                                                min=0.0,
                                                max=3.3,
                                                step=0.01,
                                                placeholder="0.0",
                                                style={"width": "100%"},
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Small("Max V"),
                                            dcc.Input(
                                                id="sweep-max",
                                                type="number",
                                                value=3.3,
                                                min=0.0,
                                                max=3.3,
                                                step=0.01,
                                                placeholder="3.3",
                                                style={"width": "100%"},
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Small("Voltage Step Size (V)"),
                                            dcc.Input(
                                                id="sweep-step",
                                                type="number",
                                                value=0.05,
                                                min=0.001,
                                                max=1.0,
                                                step=0.001,
                                                placeholder="0.05",
                                                style={"width": "100%"},
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Small("Time Between Steps (s)"),
                                            dcc.Input(
                                                id="sweep-dwell",
                                                type="number",
                                                value=0.05,
                                                min=0.0,
                                                step=0.01,
                                                placeholder="0.05",
                                                style={"width": "100%"},
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Small("Training Epochs (number of passes over sweep data)"),
                                            dcc.Input(
                                                id="sweep-epochs",
                                                type="number",
                                                value=10,
                                                min=1,
                                                step=1,
                                                placeholder="10",
                                                style={"width": "100%"},
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Small("Baseline Voltages (grid points in the sweep range)"),
                                            dcc.Input(
                                                id="sweep-baselines",
                                                type="number",
                                                min=1,
                                                step=1,
                                                value=3,
                                                placeholder="3",
                                                style={"width": "100%"},
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Small("Factorial Depth (combinations per pin)"),
                                            dcc.Input(
                                                id="sweep-factorials",
                                                type="number",
                                                min=1,
                                                step=1,
                                                value=3,
                                                placeholder="3",
                                                style={"width": "100%"},
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Small("Random Samples (additional/random data variations)"),
                                            dcc.Input(
                                                id="sweep-random-samples",
                                                type="number",
                                                min=1,
                                                step=1,
                                                value=60,
                                                placeholder="auto",
                                                style={"width": "100%"},
                                            ),
                                        ]
                                    ),
                                ],
                            ),

                            html.Div(
                                style={"display": "flex", "alignItems": "center", "gap": "1em", "marginTop": "0.5em"},
                                children=[
                                    html.Button(
                                        "Start Training Sweep",
                                        id="sweep-btn",
                                        n_clicks=0,
                                        style={"padding": "0.5em 1em"},
                                    ),

                                    html.Button(
                                        "Stop",
                                        id="sweep-stop-btn",
                                        n_clicks=0,
                                        style={"padding": "0.5em 1em"},
                                    ),

                                    html.Span(
                                        id="sweep-status",
                                        children="Sweep: idle",
                                        style={"fontWeight": "bold"},
                                    ),

                                    html.Span(id="sweep-eta", children=""),
                                ],
                            ),

                            html.Div(
                                id="sweep-progress",
                                style={
                                    "width": "100%",
                                    "height": "12px",
                                    "background": "#eee",
                                    "borderRadius": "6px",
                                    "overflow": "hidden",
                                    "marginTop": "0.5em",
                                },

                                children=[
                                    html.Div(
                                        id="sweep-progress-inner",
                                        style={
                                            "height": "100%",
                                            "width": "0%",
                                            "background": "#ccc",
                                            "transition": "width 0.2s ease",
                                        },
                                    )
                                ],
                            ),
                        ],
                    ),

                    html.Div(
                        style={
                            "flex": "1 1 280px",
                            "padding": "1em",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "8px",
                            "background": "#fafafa",
                        },

                        children=[
                            html.Div(children=[html.H4("RNN Feedback Control Panel", style={"margin": 0})]),
                            
                            html.P("Enable auto control to let the RNN propose DAC voltages, manually save the model, or set auto-update timing.",style={"margin": "0.25em 0 0.75em 0", "color": "#555"},),
                            
                            html.Div(
                                style={
                                    "display": "flex",
                                    "gap": "1em",
                                    "alignItems": "center",
                                    "flexWrap": "wrap",
                                    "marginTop": "0.5em",
                                },

                                children=[
                                    html.Div(
                                        children=[
                                            html.Button(
                                                "Auto Control: OFF",
                                                id="auto-mode-button",
                                                n_clicks=0,
                                                style={
                                                    "minWidth": "130px",
                                                    "padding": "0.6em 1.2em",
                                                    "fontWeight": "bold",
                                                    "borderRadius": "6px",
                                                    "border": "none",
                                                    "background": "#e74c3c",
                                                    "color": "#ffffff",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Button(
                                                "Save Model Parameters",
                                                id="save-model-button",
                                                n_clicks=0,
                                                style={
                                                    "minWidth": "150px",
                                                    "padding": "0.6em 1.2em",
                                                    "fontWeight": "bold",
                                                    "borderRadius": "6px",
                                                    "border": "none",
                                                    "background": "#3498db",
                                                    "color": "#ffffff",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Button(
                                                "Save After Sweep: OFF",
                                                id="save-dataset-button",
                                                n_clicks=0,
                                                style={
                                                    "minWidth": "150px",
                                                    "padding": "0.6em 1.2em",
                                                    "fontWeight": "bold",
                                                    "borderRadius": "6px",
                                                    "border": "none",
                                                    "background": "#e74c3c",
                                                    "color": "#ffffff",
                                                    "cursor": "pointer",
                                                },
                                            ),
                                        ]
                                    ),

                                    html.Div(
                                        children=[
                                            html.Label("Auto Control Update Period (ms)"),
                                            dcc.Input(
                                                id="auto-rate-ms",
                                                type="number",
                                                value=500,
                                                step=25,
                                                style={"width": "120px"},
                                            ),
                                        ]
                                    ),

                                    html.Span(id="save-dataset-ack", style={"minWidth": "160px"}),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _ml_tab():
    """Builds the ML metrics tab layout."""

    window_default = MODEL_INFO_BOOTSTRAP.get("online_window_seconds")

    if window_default is None:
        window_default = 30.0

    lr_default = MODEL_INFO_BOOTSTRAP.get("learning_rate")

    if lr_default is None:
        lr_default = 1e-3

    momentum_default = MODEL_INFO_BOOTSTRAP.get("momentum")

    if momentum_default is None:
        momentum_default = 0.9

    return html.Div(
        style={"padding": "1em"},
        children=[
            html.H3("ML Metrics"),

            html.Div(
                style={"display": "flex", "gap": "1em", "flexWrap": "wrap"},
                children=[
                    html.Div(id="ml-beam-mean", style={"minWidth": "220px"}),
                    html.Div(id="ml-beam-var", style={"minWidth": "220px"}),
                    html.Div(id="ml-effort", style={"minWidth": "220px"}),
                    html.Div(id="ml-saturation_series", style={"minWidth": "220px"}),
                    html.Div(id="ml-model-status", style={"minWidth": "220px"}),
                ],
            ),

            html.Div(
                style={
                    "display": "flex",
                    "gap": "1em",
                    "flexWrap": "wrap",
                    "marginTop": "1em",
                    "alignItems": "flex-end",
                },

                children=[
                    html.Div(
                        [
                            html.Label("Live Retraining Window (s)", style={"fontWeight": "bold"}),
                            dcc.Input(
                                id="online-window-seconds",
                                type="number",
                                min=5,
                                max=600,
                                step=5,
                                debounce=True,
                                value=float(window_default),
                                style={"width": "150px"},
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            html.Label("Learning Rate", style={"fontWeight": "bold"}),
                            dcc.Input(
                                id="online-learning-rate",
                                type="text",
                                debounce=True,
                                value=float(lr_default),
                                style={"width": "150px"},
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            html.Label("Learning Momentum", style={"fontWeight": "bold"}),
                            dcc.Input(
                                id="online-momentum",
                                type="text",
                                debounce=True,
                                value=float(momentum_default),
                                style={"width": "150px"},
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            html.Label("SHAP Permutations (Sampling Depth)", style={"fontWeight": "bold"}),
                            dcc.Input(
                                id="shap-permutations",
                                type="number",
                                min=1,
                                step=1,
                                value=20,
                                debounce=True,
                                style={"width": "150px"},
                            ),
                        ]
                    ),

                    html.Div(
                        children=[
                            html.Button(
                                "Compute SHAP Values",
                                id="compute-shap-button",
                                n_clicks=0,
                                style={
                                    "minWidth": "150px",
                                    "padding": "0.6em 1.2em",
                                    "fontWeight": "bold",
                                    "borderRadius": "6px",
                                    "border": "none",
                                    "background": "#3498db",
                                    "color": "#ffffff",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Span(id="shap-status", style={"marginLeft": "0.5em", "fontWeight": "bold"}),
                        ],
                        style={"display": "flex", "alignItems": "center", "gap": "0.5em"},
                    ),

                    html.Div(
                        id="online-update-config-status",
                        style={
                            "minWidth": "260px",
                            "fontWeight": "bold",
                            "color": "#34495e",
                        },
                    ),
                ],
            ),

            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1em", "marginTop": "1em"},
                children=[
                    dcc.Graph(id="ml-beam-graph", style={"height": "38vh"}),
                    dcc.Graph(id="ml-effort-graph", style={"height": "38vh"}),
                ],
            ),

            html.Div(
                style={"marginTop": "1em", "display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1em"},
                children=[
                    dcc.Graph(id="ml-attrib-bar", style={"height": "36vh"}),
                    dcc.Graph(id="ml-shap-bar", style={"height": "36vh"}),
                ],
            ),
        ],
    )


def _rig_tab():
    """Builds the placeholder rig-control tab layout."""
    return html.Div(
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "70vh",
            "color": "#7f8c8d",
            "fontSize": "1.2em",
        },
        children="Rig controls coming to a testbed near you... (soon!)",
    )


app.layout = html.Div(
    style={"fontFamily": "Segoe UI, sans-serif", "padding": "2em"},
    children=[
        dcc.Store(id="auto-mode", data=False),
        dcc.Tabs(
            id="tabs",
            value="control",
            children=[
                dcc.Tab(label="Board Controls", value="control", children=[_control_tab()]),
                dcc.Tab(label="NN Auto Controller", value="ml", children=[_ml_tab()]),
                dcc.Tab(label="Rig Controls", value="rig", children=[_rig_tab()]),
            ],
        ),
        dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
    ],
)


@app.callback(
    Output("online-update-config-status", "children"),
    Input("online-window-seconds", "value"),
    Input("online-learning-rate", "value"),
    Input("online-momentum", "value"),
)

def _configure_online_updates(window_seconds, learning_rate, momentum_value):
    """Applies online-update settings and return the relevant status message."""

    errors = []

    if window_seconds is not None and callable(set_window_update_time):
        try:
            applied_window = set_window_update_time(window_seconds)

        except Exception as fault:
            log.error(f"ERROR: Unable to set retraining window time - {fault}!")
            errors.append(f"ERROR: Unable to set retraining window time - {fault}!")

    if learning_rate is not None and callable(set_online_learning_rate):
        try:
            applied_lr = set_online_learning_rate(learning_rate)

        except Exception as fault:
            log.error(f"ERROR: Unable to set the model's learning rate - {fault}!") 
            errors.append(f"ERROR: Unable to set the model's learning rate - {fault}!")
            
    if momentum_value is not None and hasattr(Back_End_Controller, "set_online_momentum"):
        try:
            Back_End_Controller.set_online_momentum(momentum_value)

        except Exception as fault:
            log.error(f"ERROR: Unbale to set the model's learning momentum  - {fault}!")
            errors.append(f"ERROR: Unbale to set the model's learning momentum  - {fault}!")

    if not errors:
        return html.Span("")
    return html.Span(" | ".join(errors), style={"color": "#e74c3c"})


@app.callback(
    Output("save-dataset-ack", "children"),
    Input("save-model-button", "n_clicks"),
    prevent_initial_call=True,
)

def _manual_save_model(user_input):
    """Triggers a manual machine learning model parameter save and returns a status message."""
    if not user_input:
        raise PreventUpdate
    
    try:
        path = save_model_parameters()
        return f"SUCCESS: Model saved to {path} succesfully..."
    except Exception as fault:
        return f"ERROR: Unable to save model parameters - {fault}!"

# -------------------------------------------------------------------------
#                                Graph Updates
# -------------------------------------------------------------------------

@app.callback(
    Output("live-graph", "figure"),
    Output("latest-voltage", "children"),
    Output("num-points", "children"),
    Input("update-interval", "n_intervals"),
    State("auto-mode", "data"),
    State("auto-rate-ms", "value"),
)


def update_graph(_, auto_mode_enabled, auto_rate_ms):
    """Updates the live plot and optionally issue auto-control updates."""
    
    data_frame = Back_End_Controller.get_data()

    if data_frame.empty:
        return go.Figure(), "Voltage: -- V", "Samples: 0"

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=pd.to_datetime(data_frame["timestamp"], unit="s"),
            y=data_frame["voltage"],
            mode="lines",
            line=dict(color="royalblue"),
        )
    )

    figure.update_layout(
        xaxis_title="Time",
        yaxis_title="Voltage (V)",
        margin=dict(l=50, r=20, t=40, b=40),
        template="plotly_white",
    )

    recent_voltage_measurement = f"Voltage: {data_frame['voltage'].iloc[-1]:.3f} V"
    number_of_points = f"Samples: {len(data_frame)}"

    try:
        auto_on = bool(auto_mode_enabled)
        rate_ms = 500 if auto_rate_ms is None else max(100, int(auto_rate_ms))

    except Exception as fault:
        log.error(f"ERROR: Recieving invalid auto control parameters - {fault}!")
        auto_on, rate_ms = False, 500
        
    try:
        st = get_sweep_status()
        if isinstance(st, dict) and str(st.get("state", "")).lower() == "running":
            auto_on = False

    except Exception as fault:
        log.error(f"ERROR: Unable to determine sweep status - {fault}!")
        pass

    if auto_on and propose_control_vector is not None:
        try:
            global LAST_AUTO_TS
            now = time.time()
            if now - LAST_AUTO_TS >= (rate_ms / 1000.0):
                targets = propose_control_vector(data_frame)
                Back_End_Controller.set_pin_voltages(targets)
                LAST_AUTO_TS = now

        except Exception as fault:
            log.error(f"ERROR: Auto control update failed - {fault}!")

    return figure, recent_voltage_measurement, number_of_points


@app.callback(
    Output("auto-mode", "data"),
    Output("auto-mode-button", "children"),
    Output("auto-mode-button", "style"),
    Input("auto-mode-button", "n_clicks"),
)

def _toggle_auto_mode(User_Input):
    """Toggle auto-control mode and updates the toggle (button) state."""

    base_style = {
        "minWidth": "130px",
        "padding": "0.6em 1.2em",
        "fontWeight": "bold",
        "borderRadius": "6px",
        "border": "none",
        "color": "#ffffff",
        "cursor": "pointer",
    }

    try:
        Input = 0 if User_Input is None else int(User_Input)

    except Exception as fault:
        log.error(f"ERROR: Invalid user toggle input - {fault}!")
        Input = 0

    if Input % 2 == 1:
        style = {**base_style, "background": "#2ecc71", "boxShadow": "0 0 4px rgba(46,204,113,0.6)"}
        return True, "Auto Control: ON", style

    style = {**base_style, "background": "#e74c3c", "boxShadow": "0 0 4px rgba(231,76,60,0.6)"}
    return False, "Auto Control: OFF", style


@app.callback(
    Output("save-dataset-button", "children"),
    Output("save-dataset-button", "style"),
    Input("save-dataset-button", "n_clicks"),
)

def _toggle_save_dataset(User_Input):
    """Toggles automatic dataset saving and update the toggle (button) state."""

    base_style = {
        "minWidth": "150px",
        "padding": "0.6em 1.2em",
        "fontWeight": "bold",
        "borderRadius": "6px",
        "border": "none",
        "color": "#ffffff",
        "cursor": "pointer",
    }

    try:
        Input = 0 if User_Input is None else int(User_Input)

    except Exception as fault:
        log.error(f"ERROR: Invalid user toggle input - {fault}!")
        Input = 0

    user_enabled = (Input % 2) == 1

    try:
        getattr(Back_End_Controller, "set_save_dataset_enabled", lambda _: None)(user_enabled)

    except Exception as fault:
        log.error(f"ERROR: Unable to set save dataset flag - {fault}!")
        pass

    if user_enabled:
        style = {**base_style, "background": "#2ecc71", "boxShadow": "0 0 4px rgba(46,204,113,0.6)"}
    else:
        style = {**base_style, "background": "#e74c3c", "boxShadow": "0 0 4px rgba(231,76,60,0.6)"}

    label = f"Save After Sweep: {'ON' if user_enabled else 'OFF'}"
    return label, style


@app.callback(
    Output("shap-status", "children"),
    Output("ml-shap-bar", "figure"),
    Input("compute-shap-button", "n_clicks"),
    State("shap-permutations", "value"),
    prevent_initial_call=True,
)

def _compute_shap_on_demand(User_Input, shap_permutations):
    """Computes Shapley feature importance on user demand and returns a basic bar chart figure."""

    if not User_Input:
        log.warning("WARNING: SHAP computation requested without user input!")
        raise PreventUpdate
    
    try:
        perms = 20
        try:
            perms = max(1, int(shap_permutations)) if shap_permutations is not None else 20

        except Exception as fault:
            log.error(f"ERROR: Invalid SHAP permutations input - {fault}! (Using 20 as default...)")
            perms = 20

        shap_values = compute_feature_importance(max_samples=200, num_permutations=perms)
        feauture_names = shap_values.get("feature_names", [])
        importance_values = shap_values.get("importances", [])
        
        fig = go.Figure(
            data=[
                go.Bar(
                    x = importance_values,
                    y = feauture_names,
                    orientation="h",
                    marker=dict(color="#1f77b4"),
                )
            ]
        )

        fig.update_layout(
            template="plotly_white",
            xaxis_title="SHAP values (Accuracy dictated by User)",
            yaxis_title="",
            margin=dict(l=80, r=20, t=40, b=40),
        )

        fig.update_xaxes(showgrid=True, gridcolor="#e6e6e6", zeroline=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)

        return f"SHAP computed ({len(importance_values)} features)", fig
    
    except Exception as fault:
        return log.error(f"ERROR: Unable to compute SHAP values - {fault}!"), go.Figure()

# -------------------------------------------------------------------------
#                     Input validation (UI Interface)
# -------------------------------------------------------------------------

@app.callback(
    Output("switch-time-us", "style"),
    Input("switch-time-us", "value"),
)

def _validate_switch_time_input(input_value):
    """Validates the switch timing input and return a style for display."""

    base_style = {"width": "120px"}

    if input_value is None:
        return base_style

    try:
        fval = float(input_value)

    except Exception  as fault:
        log.error(f"ERROR: Invalid switch time - {fault}! (Must be in 1-20 us range...)")
        return {**base_style, "border": "2px solid red", "boxShadow": "0 0 4px rgba(255,0,0,0.6)"}

    if 1.0 <= fval <= 20.0:
        return base_style

    return {**base_style, "border": "2px solid red", "boxShadow": "0 0 4px rgba(255,0,0,0.6)"}

# -------------------------------------------------------------------------
#                             Status updater
# -------------------------------------------------------------------------

@app.callback(
    Output("status", "children"),
    Output("status", "style"),
    Input("update-interval", "n_intervals"),
)

def update_status(_):
    """Updates the connection status label and returns a style for display."""
    try:
        status_text = getattr(Back_End_Controller, "get_status", lambda: "Connecting...")()

    except Exception as fault:
        log.error(f"ERROR: Unable to retrieve controller status - {fault}!")
        status_text = "Connecting..."

    if str(status_text).startswith("Simulated"):
        return "OFFLINE", {"color": "red", "fontWeight": "bold"}
    
    elif "Connected" in str(status_text):
        return "CONNECTED", {"color": "green", "fontWeight": "bold"}
    
    else:
        return "OFFLINE", {"color": "red", "fontWeight": "bold"}

# -------------------------------------------------------------------------
#                             Pin control updates
# -------------------------------------------------------------------------

@app.callback(
    Output("pins-ack", "children"),
    Output("pins-ack", "style"),
    Input("pwm1", "value"),
    Input("pwm2", "value"),
    Input("pwm3", "value"),
    Input("pwm4", "value"),
    Input("pwm5", "value"),
    Input("switch-time-us", "value"),

    prevent_initial_call=True,
)

def update_pins(pin_voltage_1, pin_voltage_2, pin_voltage_3, pin_voltage_4, pin_voltage_5, switch_time_us):
    """Send voltage targets and switch timing to the backend."""

    pin_voltages = [pin_voltage_1, pin_voltage_2, pin_voltage_3, pin_voltage_4, pin_voltage_5]

    try:
        def _coerce_float(val):
            if val is None:
                return None
            if isinstance(val, str) and not val.strip():
                return None
            try:
                return float(val)
            except Exception:
                return "INVALID"

        targets = []

        for i, voltage in enumerate(pin_voltages, start=1):
            float_voltages = _coerce_float(voltage)
            if float_voltages is None:
                return "Enter all five target voltages (0-2000).", {"color": "red", "fontWeight": "bold"}
            if float_voltages == "INVALID":
                return f"ERROR: Invalid target for pin {i}", {"color": "red", "fontWeight": "bold"}
            if not (0.0 <= float_voltages <= 2000.0):
                return f"ERROR: Pin {i} target out of range (0-2000)!", {"color": "red", "fontWeight": "bold"}
            
            targets.append(float_voltages)

        Back_End_Controller.set_pin_voltages(targets)

        switch_timing_messgae = "not set"

        if switch_time_us is not None and not (isinstance(switch_time_us, str) and not switch_time_us.strip()):
            try:
                float_switch_timing = float(switch_time_us)

            except Exception as fault:
                return f"ERROR: Invalid switch time - {fault}!", {"color": "red", "fontWeight": "bold"}

            if not (1.0 <= float_switch_timing <= 20.0):
                return "ERROR: Switch time out of range (1-20 us)!", {"color": "red", "fontWeight": "bold"}

            try:
                getattr(Back_End_Controller, "set_switch_timing_us", lambda *_: None)(float_switch_timing)

            except Exception as fault:
                log.error(f"ERROR: Unable to set switch timing - {fault}!")
                pass

            switch_timing_messgae = f"{float_switch_timing:.1f} us"

        status_message = (
            f"Targets updated (V): squeeze_plate={targets[0]:.3f}, "
            f"ion_source={targets[1]:.3f}, "
            f"wein_filter={targets[2]:.3f}, "
            f"cone_1={targets[3]:.3f}, "
            f"cone_2={targets[4]:.3f}, "
            f"switch_time={switch_timing_messgae}"
        )

        return "", {"display": "none"}

    except Exception as fault:
        return f"ERROR: Pin update failed - {fault}!", {"color": "red", "fontWeight": "bold", "display": "block"}

# -------------------------------------------------------------------------
#                               Pin statuses
# -------------------------------------------------------------------------

@app.callback(
    Output("pin-status", "children"),
    Input("update-interval", "n_intervals"),
)

def update_pin_status(_):
    """Updates the pin-status chips based on the latest backend snapshot."""

    try:
        connection_status = getattr(Back_End_Controller, "get_status", lambda: "Connecting...")()

    except Exception as fault:
        log.error(f"ERROR: Unable to retrieve controller status - {fault}!")
        connection_status = "Connecting..."

    pin_snapshot = getattr(
        Back_End_Controller,
        "get_pins",
        lambda: {"names": [], "values": [], "timestamp": 0.0},
    )()

    pin_names = pin_snapshot.get("names", [])
    pin_values = pin_snapshot.get("values", [])

    if "Connected" in str(connection_status):
        try:
            Back_End_Controller.send_command("PINS")

        except Exception as fault:
            log.error(f"ERROR: Unable to request pin update from board - {fault}!")
            pass

    def chip(pin_label, pin_value, connection_state):
        """Builds a display chip (small UI label) for a single pin."""

        label_map = {
            "squeeze_plate": "Squeeze Plate",
            "ion_source": "Ion Source",
            "wein_filter": "Wein Filter",
            "cone_1": "Cone 1",
            "cone_2": "Cone 2",
            "switch_logic": "Switch Logic",
        }

        pretty_label = label_map.get(pin_label, pin_label.replace("_", " ").title())
        color = {
            "CONNECTED": "green",
            "DISCONNECTED": "red",
        }.get(connection_state, "#333")

        try:
            numeric_value = float(pin_value)

        except Exception as fault:
            log.warning(f"WARNING: Non-numeric pin value for {pin_label} - {fault}! (value set to None...)")
            numeric_value = None

        display = str(pin_value)

        if pin_label != "switch_logic" and numeric_value is not None:
            volts = (numeric_value / 1023.0) * 3.3
            display = f"{int(numeric_value)} ({volts:.3f} V)"

        return html.Div(
            style={
                "border": f"1px solid {color}",
                "borderRadius": "6px",
                "padding": "0.25em 0.5em",
                "minWidth": "180px",
            },

            children=[
                html.Span(f"{pretty_label}: ", style={"fontWeight": "bold"}),
                html.Span(display),
                html.Span(f"  ({connection_state})", style={"marginLeft": "0.4em", "color": color}),
            ],
        )

    if not pin_names or not pin_values:
        return [chip("Pins", "--", "DISCONNECTED")]

    if str(connection_status).startswith("Simulated"):
        connection_state = "DISCONNECTED"

    elif "Connected" in str(connection_status):
        connection_state = "CONNECTED"

    else:
        connection_state = "DISCONNECTED"

    value_display = list(pin_values)

    if len(value_display) >= 6:
        value_display[5] = "ON" if int(value_display[5]) else "OFF"

    pin_status_updates = [
        chip(pin_names, value_display[i] if i < len(value_display) else "--", connection_state)
        for i, pin_names in enumerate(pin_names)
    ]

    return pin_status_updates


# -------------------------------------------------------------------------
#                           Training sweep control
# -------------------------------------------------------------------------

@app.callback(
        
    Output("sweep-status", "children"),
    Output("sweep-btn", "disabled"),
    Output("sweep-progress-inner", "style"),

    Input("update-interval", "n_intervals"),
    Input("sweep-btn", "n_clicks"),
    Input("sweep-stop-btn", "n_clicks"),

    State("sweep-min", "value"),
    State("sweep-max", "value"),
    State("sweep-step", "value"),
    State("sweep-dwell", "value"),
    State("sweep-epochs", "value"),
    State("sweep-baselines", "value"),
    State("sweep-factorials", "value"),
    State("sweep-random-samples", "value"),

    prevent_initial_call=False,
)

def handle_training_sweep(
    _,
    input_counter,
    stop_click_count,
    min_voltage,
    max_voltage,
    step,
    dwell_s,
    number_of_epochs,
    baseline_levels_value,
    factorial_levels_value,
    random_samples_value,
):
    """Handles sweep start/stop and updatse the sweep progress indicator for the UI."""
    
    trigger_id = None
    
    try:
        if ctx and ctx.triggered:
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

    except Exception as fault:
        log.error(f"ERROR: Unable to determine sweep trigger from user - {fault}!")
        trigger_id = None

    if trigger_id == "sweep-btn" and input_counter and int(input_counter) > 0:

        try:
            min_sweep_voltage = 0.0 if min_voltage is None else float(min_voltage)
            max_sweep_voltage = 3.3 if max_voltage is None else float(max_voltage)

            sweep_step = 0.05 if step is None or step <= 0 else float(step)
            dwell_seconds = 0.05 if dwell_s is None or dwell_s < 0 else float(dwell_s)

            try:
                epochs_value = int(number_of_epochs) if number_of_epochs is not None else 10

            except Exception as fault:
                log.error(f"ERROR: Invalid epochs input - {fault}! (Using 10 as default...)")
                epochs_value = 10

            if epochs_value <= 0:
                log.warning("WARNING: Negative epochs input! (Using 10 as default...)")
                epochs_value = 10

            def _clean_positive_int(messy_data_point):
                """Ensures user inputs are positive integers."""

                if messy_data_point is None:
                    return None
                try:
                    clean_data_point = int(messy_data_point)

                except Exception as fault:
                    log.error(f"ERROR: Invalid integer input - {fault}!")
                    return None
                return clean_data_point if clean_data_point > 0 else None

            baseline_count = _clean_positive_int(baseline_levels_value)
            factorial_count = _clean_positive_int(factorial_levels_value)
            random_sample_override = _clean_positive_int(random_samples_value)

            started = start_training_sweep(
                min_v=min_sweep_voltage,
                max_v=max_sweep_voltage,
                step=sweep_step,
                dwell_s=dwell_seconds,
                epochs=epochs_value,
                baseline_levels=baseline_count,
                factorial_levels=factorial_count,
                random_samples=random_sample_override,
            )

            if started:
                log.info("Training sweep initiated by user...")

        except Exception as fault:
            log.error(f"ERROR: Failed to start training sweep - {fault}!")

    elif trigger_id == "sweep-stop-btn" and stop_click_count and int(stop_click_count) > 0:
        try:
            stop_training_sweep()
            log.info("Training sweep cancelled by user...")

        except Exception as fault:
            log.error(f"ERROR: Failed to stop training sweep - {fault}!")

    try:
        sweep_status = get_sweep_status()

    except Exception as fault:
        log.error(f"ERROR: Unable to retrieve sweep status - {fault}!")
        sweep_status = {"state": "unknown", "progress": 0.0, "message": ""}

    sweep_state = str(sweep_status.get("state", "idle"))
    sweep_progress = float(sweep_status.get("progress", 0.0))
    sweep_message = str(sweep_status.get("message", ""))

    percent_completion = int(max(0.0, min(1.0, sweep_progress)) * 100)
    text = f"Sweep: {sweep_state} ({percent_completion}%)" + (" - " + sweep_message if sweep_message else "")
    disabled = sweep_state == "running"

    if sweep_state == "running":
        color = "#3498db"  

    elif sweep_state == "completed":
        color = "#2ecc71"  

    elif sweep_state == "failed":
        color = "#e74c3c"  

    else:
        color = "#ccc"

    bar_style = {"height": "100%", "width": f"{percent_completion}%", "background": color, "transition": "width 0.2s ease",}

    return text, disabled, bar_style

# -------------------------------------------------------------------------
#                              ML Tab updates
# -------------------------------------------------------------------------

@app.callback(
    Output("ml-beam-graph", "figure"),
    Output("ml-effort-graph", "figure"),
    Output("ml-attrib-bar", "figure"),
    Output("ml-beam-mean", "children"),
    Output("ml-beam-var", "children"),
    Output("ml-effort", "children"),
    Output("ml-saturation_series", "children"),
    Output("ml-model-status", "children"),
    Input("update-interval", "n_intervals"),
)

def update_ml_tab(_):
    """Updates ML plots and summary cards from the latest metrics snapshot."""

    try:
        metrics_snapshot = get_ml_metrics()

    except Exception as fault:
        log.error(f"ERROR: Unable to retrieve ML metrics - {fault}!")
        metrics_snapshot = {}

    sample_times_series = metrics_snapshot.get("Sample_Times_series", [])
    voltage_time_series = metrics_snapshot.get("Input_Voltage_series", [])
    control_effort_series = metrics_snapshot.get("Pin_Control_Changes_series", [])
    saturation_series = metrics_snapshot.get("Saturation_Indicators_series", [])
    training_times_series = metrics_snapshot.get("Training_Times", [])
    training_loss_series = metrics_snapshot.get("Training_Loss", [])
    training_r2_series = metrics_snapshot.get("Training_R2", [])
    training_source_series = metrics_snapshot.get("Training_Source", [])

    if sample_times_series and voltage_time_series:
        beam_figure = go.Figure(data=[go.Scatter(x=pd.to_datetime(sample_times_series, unit="s"), y=voltage_time_series, mode="lines", line=dict(color="#1f77b4", width=3))])
        beam_figure.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30), xaxis_title="Time", yaxis_title="Output voltage (V)")
        beam_figure.update_xaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        beam_figure.update_yaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
    
    else:
       beam_figure = go.Figure()

    if sample_times_series and control_effort_series:
        effort_figure = go.Figure(data=[go.Scatter(x=pd.to_datetime(sample_times_series, unit="s"), y=control_effort_series, mode="lines", line=dict(color="#2e7d32", width=3))])
        effort_figure.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30), xaxis_title="Time", yaxis_title="Control effort")
    
    else:
        effort_figure = go.Figure()

    try:
        effort_figure.update_xaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        effort_figure.update_yaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        effort_figure.update_layout(font=dict(family="Segoe UI, sans-serif", size=12, color="#111"), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", yaxis_title="Control effort")

    except Exception as fault:
        log.error(f"ERROR: Unable to construct appropriate figure axes - {fault}!")
        pass

    if sample_times_series and saturation_series:
        try:
            effort_figure.add_trace(
                go.Scatter(
                    x=pd.to_datetime(sample_times_series, unit="s"),
                    y=saturation_series,
                    mode="lines",
                    line=dict(color="#e74c3c", width=2),
                    name="Saturation",
                    yaxis="y2",
                )
            )

            effort_figure.update_layout(
                yaxis2=dict(
                    title="Saturation",
                    overlaying="y",
                    side="right",
                    range=[-0.05, 1.05],
                    showgrid=False,
                    zeroline=False,
                    tickmode="array",
                    tickvals=[0, 1],
                    ticktext=["0", "1"],
                    linecolor="#444",
                )
            )

            try:
                for tr in list(effort_figure.data):
                    if getattr(tr, "mode", "") == "lines":
                        tr.showlegend = False

            except Exception as fault:
                log.error(f"ERROR: Unable to update figure legends - {fault}!")
                pass
            
            effort_figure.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(symbol="square", size=12, color="#2e7d32", line=dict(color="#000", width=1)),
                    name="Correction effort",
                    showlegend=True,
                )
            )

            effort_figure.add_trace(
                go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(symbol="square", size=12, color="#e74c3c", line=dict(color="#000", width=1)),
                    name="Saturation",
                    showlegend=True,
                )
            )

            effort_figure.update_layout(
                legend=dict(
                    orientation="v",
                    y=1.0, x=1.0,
                    yanchor="top", xanchor="right",
                    bgcolor="#ffffff",
                    bordercolor="#000000", borderwidth=0.5,
                    font=dict(size=11, color="#111"),
                )
            )

        except Exception as fault:
            log.error(f"ERROR: Unable to plot saturations - {fault}!")
            pass

    if training_times_series and (training_loss_series or training_r2_series):
        try:
            training_time_stamp = [i for i, j in enumerate(training_times_series) if j is not None]

        except Exception as fault:
            log.error(f"ERROR: Unable to understand training time stamps - {fault}!")
            training_time_stamp = list(range(len(training_times_series)))

        times = np.array([training_times_series[i] for i in training_time_stamp], dtype=float)
        losses = np.array(
            [training_loss_series[i] for i in training_time_stamp if i < len(training_loss_series)],
            dtype=float,
        )

        regression_score = np.array(
            [training_r2_series[i] for i in training_time_stamp if i < len(training_r2_series)],
            dtype=float,
        )

        time_index = pd.to_datetime(times, unit="s") if times.size else []

        training_figure = go.Figure()

        if times.size and losses.size:
            training_figure.add_trace(
                go.Scatter(
                    x=time_index,
                    y=losses,
                    mode="lines+markers",
                    line=dict(color="#1f77b4"),
                    name="Loss (MSE)",
                )
            )

            try:
                window = int(min(5, max(2, losses.size)))
            except Exception as fault:
                log.error(f"ERROR: Unable to determine smoothing window size - {fault}! (Using 2 as default...)")
                window = 2

            if losses.size >= window:
                kernel = np.ones(window, dtype=float) / float(window)
                smooth_loss = np.convolve(losses, kernel, mode="valid")
                smooth_times = time_index[window - 1 :]
                training_figure.add_trace(
                    go.Scatter(
                        x=smooth_times,
                        y=smooth_loss,
                        mode="lines",
                        line=dict(color="#3498db", dash="dash"),
                        name="Loss (rolling avg)",
                    )
                )

        if times.size and regression_score.size:
            training_figure.add_trace(
                go.Scatter(
                    x=time_index,
                    y=regression_score,
                    mode="lines+markers",
                    line=dict(color="#e67e22"),
                    name="R²",
                    yaxis="y2",
                )
            )
        try:
            for i, training_source in enumerate(training_source_series or []):
                if str(training_source).lower() == "sweep" and i < len(time_index):
                    training_figure.add_vline(
                        x=time_index[i],
                        line=dict(color="rgba(231,76,60,0.6)", width=1, dash="dot"),
                    )

        except Exception as fault:
            log.error(f"ERROR: Unable to add sweep data to figures - {fault}!")
            pass

        training_figure.update_layout(
            template="plotly_white",
            margin=dict(l=40, r=10, t=30, b=30),
            xaxis_title="Time",
            yaxis_title="Training loss (MSE)",
            yaxis2=dict(
                title="R²",
                overlaying="y",
                side="right",
                range=[-0.1, 1.05],
                showgrid=False,
                zeroline=False,
                tickmode="array",
                tickvals=[0.0, 0.5, 1.0],
            ),
        )

    else:
        training_figure = go.Figure()
        training_figure.add_annotation(
            text="Training history not available yet.",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        training_figure.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30))

    try:
        training_figure.update_xaxes(showgrid=False, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        training_figure.update_yaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        training_figure.update_layout(font=dict(family="Segoe UI, sans-serif", size=12, color="#111"), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
   
    except Exception as fault:
        log.error(f"ERROR: Unable to construct appropriate figure axes - {fault}!")
        pass

    mean_beam_voltage = metrics_snapshot.get("Input_Voltage_mean")
    beam_variance = metrics_snapshot.get("Input_Voltage_var")
    mean_control_effort = metrics_snapshot.get("Pin_Control_Changes_mean")
    saturation_indicators = metrics_snapshot.get("Saturation_Indicators_total", 0)

    beam_mean_text = f"{mean_beam_voltage:.3f} V" if mean_beam_voltage is not None else "--"
    beam_var_text = f"{beam_variance:.5f}" if beam_variance is not None else "--"
    effort_text = f"{mean_control_effort:.2f}" if mean_control_effort is not None else "--"
    saturation_text = f"{int(saturation_indicators)}"

    def _format_age(seconds: float) -> str:
        """Formats Machine learning model age into seconds."""
        if seconds is None:
            return ""
        
        if seconds < 60:
            return f"{seconds:.0f}s"
        
        if seconds < 3600:
            return f"{seconds/60:.1f} min"
        
        return f"{seconds/3600:.1f} h"

    try:
        model_info = get_model_info()

    except Exception as fault:
        log.error(f"ERROR: Unable to retrieve model info - {fault}!")
        model_info = {}

    last_train_sec = model_info.get("last_train_ago_sec")
    sweep_state = str(model_info.get("sweep_state", "")).lower()
    window_sec = model_info.get("online_window_seconds")

    if window_sec is None:
        log.warning("WARNING: Online window size not available from model info! (Using 30s as default...)")
        window_sec = 30.0

    online_enabled = model_info.get("online_updates_enabled", True)
    learning_rate_value = model_info.get("learning_rate")

    if sweep_state == "running":
        model_state_label = "training"
        state_color = "#2ecc71"

    elif last_train_sec is None:
        model_state_label = "not trained"
        state_color = "#e74c3c"

    else:
        stale_threshold = max(120.0, 2.0 * float(window_sec))

        if float(last_train_sec) <= stale_threshold:
            model_state_label = "trained"
            state_color = "#2ecc71"
        else:
            model_state_label = "stale"
            state_color = "#f1c40f"

    card_style = {
        "border": "1px solid #ddd",
        "borderRadius": "6px",
        "padding": "0.5em 0.75em",
        "minWidth": "220px",
        "background": "#fff",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
        "color": "#111",
    }

    label_style = {"fontWeight": "bold", "marginRight": "0.35em", "color": "#111"}
    value_style = {"fontWeight": "bold", "color": "#111"}

    mean_beam_outputs = html.Div([
        html.Span("Mean Beam Voltage:", style=label_style), html.Span(beam_mean_text, style=value_style)
    ], style=card_style)

    mean_variance_outputs = html.Div([
        html.Span("Variance Beam Voltage:", style=label_style), html.Span(beam_var_text, style=value_style)
    ], style=card_style)

    mean_control_effort_outputs = html.Div([
        html.Span("Control Effort:", style=label_style), html.Span(effort_text, style=value_style)
    ], style=card_style)

    saturation_indicator_outputs = html.Div([
        html.Span("Saturations:", style=label_style), html.Span(saturation_text, style=value_style)
    ], style=card_style)

    age_text = _format_age(last_train_sec) if last_train_sec is not None else ""

    config_bits = []

    if age_text:
        config_bits.append(f"Time since last retrain: {age_text}")

    if window_sec is not None:
        try:
            config_bits.append(f"Window {float(window_sec):.0f}s")

        except Exception as fault:
            log.error(f"ERROR: Unable to format window size - {fault}!")
            pass

    if learning_rate_value:
        try:
            config_bits.append(f"LR {float(learning_rate_value):.3e}")

        except Exception as fault:
            log.error(f"ERROR: Unable to format learning rate - {fault}!")
            pass

    model_status_outputs = html.Div(
        [
            html.Span("Model:", style=label_style),
            html.Span(model_state_label.capitalize(), style={"fontWeight": "bold", "color": state_color}),
            html.Span(
                "" if online_enabled else " | Online updates paused",
                style={"marginLeft": "0.4em", "color": "#e67e22"},
            ),
            html.Div(
                " | ".join(config_bits) if config_bits else "",
                style={"marginTop": "0.25em", "color": "#555"},
            ),
        ],
        style={**card_style, "border": f"1px solid {state_color}"},
    )

    return (
        beam_figure,
        effort_figure,
        training_figure,
        mean_beam_outputs,
        mean_variance_outputs,
        mean_control_effort_outputs,
        saturation_indicator_outputs,
        model_status_outputs,
    )
# -------------------------------------------------------------------------
#                               Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":

    log.info("Launching ESP-12F Control Dashboard...")
    app.run(debug=False, host="0.0.0.0", port=8050)
