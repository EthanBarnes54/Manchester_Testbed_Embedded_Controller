#--------------- ESP32 Control and Monitoring Dashboard -------------------#

# A Dash-based web interface for real-time control and data visualization
# of the ESP32-based embedded control system. Communicates with the backend
# and  ML system to display live metrics of both the ion beam testbed and 
# RNN remote controller.

# Run locally in a new terminal and open http://127.0.0.1:8050 in a browser.
# ------------------------------------------------------------------------#

from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pandas as pd
import pkgutil, importlib.util
import numpy as np
import logging
import time
import dash
try:
    from python_RNN_Controller import propose_dac
except Exception:
    propose_dac = None
from python_Backend import (get_ml_metrics, Back_End_Controller, start_training_sweep, get_sweep_status, stop_training_sweep, get_model_info)

# Workaround for Python 3.14 (pkgutil.find_loader has been removed)
if not hasattr(pkgutil, "find_loader"):
    pkgutil.find_loader = lambda name: importlib.util.find_spec(name)

# -------------------------------------------------------------------------
#                                Logging setup
# -------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S",)
log = logging.getLogger("Dashboard")

# -------------------------------------------------------------------------
#                                 Initialize app
# -------------------------------------------------------------------------

app = Dash(__name__, title="Manchester Ion Beam Testbed: Control Dashboard")
server = app.server

LAST_AUTO_TS = 0.0
# -------------------------------------------------------------------------
#                                     Layout
# -------------------------------------------------------------------------

def _control_tab():
    return html.Div(
        children=[
            html.H2("ESP12-F Control Dashboard", style={"textAlign": "center"}),
            dcc.Graph(id="live-graph", style={"height": "60vh"}),

            html.Div(style={"marginTop": "2em", "display": "flex", "alignItems": "center", "gap": "1em"},
                    children=[
                    html.Label("Set DAC Voltage (V):", style={"fontWeight": "bold"}),
                    dcc.Input(id="voltage-input", type="number", min=0.0, max=3.3, step=0.000001, value=1.0, debounce=True, style={"width": "140px"}),
                    html.Button("Send Command", id="send-btn", n_clicks=0, style={"padding": "0.5em 1em"}),
                    html.Span(id="ack", style={"marginLeft": "1em", "fontWeight": "bold"}),
                ],
            ),

            html.Div(style={"marginTop": "1em"},
                     
                children=[
                    html.H4("Pin Controls: 5 PWM + 1 Logic Switch"),
                    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "1em"},
                        children=[
                            html.Div([html.Label("Squeeze Plate"), dcc.Input(id="pwm1", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Ion Source"), dcc.Input(id="pwm2", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Wein Filter"), dcc.Input(id="pwm3", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Upper Cone (Initial/Entry Cone)"), dcc.Input(id="pwm4", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Lower Cone (Final/Exit Cone)"), dcc.Input(id="pwm5", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Switch Logic"), dcc.Checklist(id="sw6", options=[{"label": "ON", "value": "on"}], value=[], inputStyle={"marginRight": "0.5em"})]),
                        ],
                    ),
                    html.Div(id="pins-ack", style={"marginTop": "0.5em", "fontWeight": "bold"}),
                ],
            ),

            html.Div(id="metrics-bar",
                style={"display": "flex", "justifyContent": "space-around", "marginTop": "2em", "padding": "1em", "borderTop": "1px solid #ddd", "color": "#333"},
                children=[
                    html.Div(id="latest-voltage", children="Voltage: -- V"),
                    html.Div(id="num-points", children="Samples: --"),
                    html.Div(id="status", children="OFFLINE", style={"color": "red", "fontWeight": "bold"}),
                ],
            ),

            html.Div(id="pin-status", style={"display": "flex", "flexWrap": "wrap", "gap": "0.5em 1em", "marginTop": "0.5em"}),
            html.Div(style={"marginTop": "1.5em", "padding": "1em", "border": "1px solid #e0e0e0", "borderRadius": "8px", "maxWidth": "740px", "background": "#fafafa"},
                children=[
                    html.Div(children=[html.H4("Training Sweep", style={"margin": 0}), html.P("Generate the training dataset by sweeping over all ouput DAC values, then train the RNN controller.", style={"margin": "0.25em 0 0.75em 0", "color": "#555"})]),

                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "0.75em", "alignItems": "end"},
                        children=[
                            html.Div(children=[html.Small("Min V"), dcc.Input(id="sweep-min", type="number", value=0.0, min=0.0, max=3.3, step=0.01, placeholder="0.0", style={"width": "100%"})]),
                            html.Div(children=[html.Small("Max V"), dcc.Input(id="sweep-max", type="number", value=3.3, min=0.0, max=3.3, step=0.01, placeholder="3.3", style={"width": "100%"})]),
                            html.Div(children=[html.Small("Step"), dcc.Input(id="sweep-step", type="number", value=0.05, min=0.001, max=1.0, step=0.001, placeholder="0.05", style={"width": "100%"})]),
                            html.Div(children=[html.Small("Dwell (s)"), dcc.Input(id="sweep-dwell", type="number", value=0.05, min=0.0, step=0.01, placeholder="0.05", style={"width": "100%"})]),
                            html.Div(children=[html.Small("Epochs"), dcc.Input(id="sweep-epochs", type="number", value=10, min=1, step=1, placeholder="10", style={"width": "100%"})]),
                        ],
                    ),

                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "1em", "marginTop": "0.5em"},
                        children=[
                            html.Button("Start Training Sweep", id="sweep-btn", n_clicks=0, style={"padding": "0.5em 1em"}),
                            html.Button("Stop", id="sweep-stop-btn", n_clicks=0, style={"padding": "0.5em 1em"}),
                            html.Span(id="sweep-status", children="Sweep: idle", style={"fontWeight": "bold"}),
                            html.Span(id="sweep-eta", children=""),
                        ],
                    ),
                    
                    html.Div(id="sweep-progress", style={"width": "100%", "height": "12px", "background": "#eee", "borderRadius": "6px", "overflow": "hidden", "marginTop": "0.5em"}, children=[html.Div(id="sweep-progress-inner", style={"height": "100%", "width": "0%", "background": "#ccc", "transition": "width 0.2s ease"})]),

                    html.Div(
                        style={"marginTop": "1em", "padding": "1em", "border": "1px solid #e0e0e0", "borderRadius": "8px", "maxWidth": "740px", "background": "#fafafa"},
                        children=[
                            html.Div(children=[html.H4("Automation & Data", style={"margin": 0})]),
                            html.Div(
                                style={"display": "flex", "gap": "1em", "alignItems": "center", "flexWrap": "wrap", "marginTop": "0.5em"},
                                children=[
                                    html.Div(children=[html.Label("Auto Control"), dcc.Checklist(id="auto-mode", options=[{"label": "ON", "value": "on"}], value=[], inputStyle={"marginRight": "0.5em"})]),
                                    html.Div(children=[html.Label("Auto Rate (ms)"), dcc.Input(id="auto-rate-ms", type="number", value=500, min=100, step=50, style={"width": "100px"})]),
                                    html.Div(children=[html.Label("Save dataset after sweep"), dcc.Checklist(id="save-dataset-toggle", options=[{"label": "Enable", "value": "on"}], value=[], inputStyle={"marginRight": "0.5em"})]),
                                    html.Span(id="save-dataset-ack", style={"fontWeight": "bold"}),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def _ml_tab():
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
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "1em", "marginTop": "1em"},
                children=[
                    dcc.Graph(id="ml-beam-graph", style={"height": "38vh"}),
                    dcc.Graph(id="ml-effort-graph", style={"height": "38vh"}),
                ],
            ),
            html.Div(style={"marginTop": "1em"}, children=[dcc.Graph(id="ml-attrib-bar", style={"height": "36vh"})]),
        ],
    )


app.layout = html.Div(style={"fontFamily": "Segoe UI, sans-serif", "padding": "2em"},
                      
    children=[

        dcc.Tabs(id="tabs", value="control", 
            children=[
            dcc.Tab(label="Control", value="control", children=[_control_tab()]),
            dcc.Tab(label="ML", value="ml", children=[_ml_tab()]),
        ]),

        dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
    ],
)

# -------------------------------------------------------------------------
#                                Graph Updates
# -------------------------------------------------------------------------

@app.callback(
    Output("live-graph", "figure"),
    Output("latest-voltage", "children"),
    Output("num-points", "children"),
    Input("update-interval", "n_intervals"),
    State("auto-mode", "value"),
    State("auto-rate-ms", "value"),
)


def update_graph(_, auto_mode_value, auto_rate_ms):
    
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

        # Auto-control (guarded, rate-limited)
    try:
        auto_on = isinstance(auto_mode_value, (list, tuple)) and ("on" in auto_mode_value)
        rate_ms = 500 if auto_rate_ms is None else max(100, int(auto_rate_ms))
    except Exception:
        auto_on, rate_ms = False, 500

    # Disable auto during sweep runs
    try:
        st = get_sweep_status()
        if isinstance(st, dict) and str(st.get("state", "")).lower() == "running":
            auto_on = False
    except Exception:
        pass

    if auto_on and propose_dac is not None:
        try:
            global LAST_AUTO_TS
            now = time.time()
            if now - LAST_AUTO_TS >= (rate_ms / 1000.0):
                grid = np.linspace(0.0, 3.3, 34, dtype=float)
                best_dac, _ = propose_dac(data_frame, candidate_dacs=grid)
                if 0.0 <= best_dac <= 3.3:
                    Back_End_Controller.send_command(f"SET {best_dac:.6f}")
                    LAST_AUTO_TS = now
        except Exception:
            pass
    return figure, recent_voltage_measurement, number_of_points


@app.callback(
    Output("ack", "children"),
    Output("ack", "style"),
    Input("send-btn", "n_clicks"),
    State("voltage-input", "value"),
    prevent_initial_call=True,
)

def send_voltage_command(number_of_clicks, input_value):

    if not number_of_clicks:
        raise PreventUpdate
    try:
        if input_value is None:
            return "Enter a value...", {"color": "#333"}
        
        input_voltage = float(input_value)

        if not (0.0 <= input_voltage <= 3.3):
            return "ERROR: Input out of range (0.0-3.3 V)!", {"color": "red", "fontWeight": "bold"}
        
        Back_End_Controller.send_command(f"SET {input_voltage:.6f}")

        return f"Sent: SET {input_voltage:.6f}", {"color": "green", "fontWeight": "bold"}
    except Exception as fault:
        return f"ERROR: Command not set - {fault}!", {"color": "red", "fontWeight": "bold"}
@app.callback(
    Output("save-dataset-ack", "children"),
    Input("save-dataset-toggle", "value"),
)
def _toggle_save_dataset(value):
    enabled = isinstance(value, (list, tuple)) and ("on" in value)
    try:
        getattr(Back_End_Controller, "set_save_dataset_enabled", lambda _: None)(enabled)
    except Exception:
        pass
    return f"Save after sweep: {'ON' if enabled else 'OFF'}"

# -------------------------------------------------------------------------
#                     Input validation (UI Interface)
# -------------------------------------------------------------------------

@app.callback(
    Output("voltage-input", "style"),
    Input("voltage-input", "value"),
)

def _validate_voltage_input(input_value):
    base_style = {"width": "140px"}

    if input_value is None:
        return base_style
    try:
        fval = float(input_value)
    except Exception:
        return {** base_style, "border": "2px solid red", "boxShadow": "0 0 4px rgba(255,0,0,0.6)"}
    
    if 0.0 <= fval <= 3.3:
        return  base_style
    return {** base_style, "border": "2px solid red", "boxShadow": "0 0 4px rgba(255,0,0,0.6)"}

# -------------------------------------------------------------------------
#                             Status updater
# -------------------------------------------------------------------------

@app.callback(
    Output("status", "children"),
    Output("status", "style"),
    Input("update-interval", "n_intervals"),
)

def update_status(_):
    try:
        status_text = getattr(Back_End_Controller, "get_status", lambda: "Connecting...")()
    except Exception:
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
    Input("sw6", "value"),

    prevent_initial_call=True,
)

def update_pins(pin_voltage_1, pin_voltage_2, pin_voltage_3, pin_voltage_4, pin_voltage_5, logic_pin):

    pin_voltages = [pin_voltage_1, pin_voltage_2, pin_voltage_3, pin_voltage_4, pin_voltage_5]

    try:
        for channel_index, value_index in enumerate(pin_voltages, start=1):

            if value_index is None:
                continue

        try:
            Back_End_Controller.set_pwm(channel_index, int(value_index))
        except (TypeError, ValueError):
            pass

        switch_status = isinstance(logic_pin, (list, tuple)) and ("on" in logic_pin)
        Back_End_Controller.set_switch(bool(switch_status))

        status_message = (
            f"Pins updated: squeeze_plate={pin_voltages[0]}, "
            f"ion_source={pin_voltages[1]}, "
            f"wein_filter={pin_voltages[2]}, "
            f"cone_1={pin_voltages[3]}, "
            f"cone_2={pin_voltages[4]}, "
            f"switch_logic={'ON' if switch_status else 'OFF'}"
            )        
        
        return status_message, {"color": "green", "fontWeight": "bold"}
    
    except Exception as fault:
        return f"ERROR: Pin update failed - {fault}!", {"color": "red", "fontWeight": "bold"}

# -------------------------------------------------------------------------
#                               Pin statuses
# -------------------------------------------------------------------------

@app.callback(
    Output("pin-status", "children"),
    Input("update-interval", "n_intervals"),
)

def update_pin_status(_):

    try:
        connection_status = getattr(Back_End_Controller, "get_status", lambda: "Connecting...")()
    except Exception:
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
        except Exception:
            pass

    def chip(pin_label, pin_value, connection_state):
        color = {
            "CONNECTED": "green",
            "DISCONNECTED": "red",
        }.get(connection_state, "#333")
        return html.Div(
            style={
                "border": f"1px solid {color}",
                "borderRadius": "6px",
                "padding": "0.25em 0.5em",
                "minWidth": "180px",
            },
            children=[
                html.Span(f"{pin_label}: ", style={"fontWeight": "bold"}),
                html.Span(str(pin_value)),
                html.Span(f"  ({connection_state})", style={"marginLeft": "0.4em", "color": color}),
            ],
        )

    if not pin_names or not pin_values:
        return [chip("Pins", "--", "DISCONNECTED")]

    # Determine connection state from status text
    if str(connection_status).startswith("Simulated"):
        connection_state = "DISCONNECTED"
    elif "Connected" in str(connection_status):
        connection_state = "CONNECTED"
    else:
        connection_state = "DISCONNECTED"

    value_display = list(pin_values)
    if len(value_display) >= 6:
        value_display[5] = "ON" if int(value_display[5]) else "OFF"

    pin_status_chips = [
        chip(name, value_display[i] if i < len(value_display) else "--", connection_state)
        for i, name in enumerate(pin_names)
    ]
    return pin_status_chips


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

    prevent_initial_call=False,
)

def handle_training_sweep(_, start_click_count, stop_click_count, min_voltage, max_voltage, step, dwell_s,  number_of_epochs):
    
    trigger_id = None
    
    try:
        callback_context = dash.callback_context

        if callback_context and callback_context.triggered:
            trigger_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    except Exception:
        trigger_id = None

    if trigger_id == "sweep-btn" and start_click_count and int(start_click_count) > 0:

        try:
            min_sweep_voltage = 0.0 if min_voltage is None else float(min_voltage)
            max_sweep_voltage = 3.3 if max_voltage is None else float(max_voltage)

            sweep_step = 0.05 if step is None or step <= 0 else float(step)
            dwell_seconds = 0.05 if dwell_s is None or dwell_s < 0 else float(dwell_s)
            number_of_epochs = 10 if  number_of_epochs is None or int( number_of_epochs) <= 0 else int( number_of_epochs)

            started = start_training_sweep(min_v=min_sweep_voltage, max_v=max_sweep_voltage, step=sweep_step, dwell_s=dwell_seconds, epochs=int(number_of_epochs) if number_of_epochs else 10)

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
    except Exception:
        sweep_status = {"state": "unknown", "progress": 0.0, "message": ""}

    sweep_state = str(sweep_status.get("state", "idle"))
    sweep_progress = float(sweep_status.get("progress", 0.0))
    sweep_message = str(sweep_status.get("message", ""))

    percent_completion = int(max(0.0, min(1.0, sweep_progress)) * 100)
    text = f"Sweep: {sweep_state} ({percent_completion}%)" + (f" – {sweep_message}" if sweep_message else "")
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

    try:
        metrics_snapshot = get_ml_metrics()
    except Exception:
        metrics_snapshot = {}

    sample_times_series = metrics_snapshot.get("Sample_Times_series", [])
    voltage_time_series = metrics_snapshot.get("Input_Voltage_series", [])
    control_effort_series = metrics_snapshot.get("Pin_Control_Changes_series", [])
    saturation_series = metrics_snapshot.get("Saturation_Indicators_series", [])

    if sample_times_series and voltage_time_series:
        beam_figure = go.Figure(data=[go.Scatter(x=pd.to_datetime(sample_times_series, unit="s"), y=voltage_time_series, mode="lines", line=dict(color="#1f77b4", width=3))])
        beam_figure.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30), xaxis_title="Time", yaxis_title="Diode voltage (V)")
        beam_figure.update_xaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        beam_figure.update_yaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
    
    else:
       beam_figure = go.Figure()

    if sample_times_series and control_effort_series:
        effort_figure = go.Figure(data=[go.Scatter(x=pd.to_datetime(sample_times_series, unit="s"), y=control_effort_series, mode="lines", line=dict(color="#2e7d32", width=3))])
        effort_figure.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30), xaxis_title="Time", yaxis_title="Control effort (||Î”u||^2)")
    
    else:
        effort_figure = go.Figure()

    try:
        effort_figure.update_xaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        effort_figure.update_yaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        effort_figure.update_layout(font=dict(family="Segoe UI, sans-serif", size=12, color="#111"), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", yaxis_title="Control effort (||du||^2)")
    except Exception:
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
            except Exception:
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
        except Exception:
            pass

    feature_names = metrics_snapshot.get("Feature_Names", [])
    feature_saliency = metrics_snapshot.get("Feature_Saliency", [])
    
    if feature_names and feature_saliency and len(feature_names) == len(feature_saliency):
        attribution_figure = go.Figure(data=[go.Bar(x=feature_names, y=np.abs(np.array(feature_saliency)).tolist(), marker_color="#2c3e50")])
        attribution_figure.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30), yaxis_title="|Attribution|")
    
    else:
        attribution_figure = go.Figure()
        attribution_figure.add_annotation(text="Attribution unavailable (attach pipeline+controller or install SHAP)", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        attribution_figure.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30))

    try:
        attribution_figure.update_xaxes(showgrid=False, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        attribution_figure.update_yaxes(showgrid=True, gridcolor="#e6e6e6", gridwidth=1, zeroline=False, linecolor="#444", mirror=True, ticks="outside")
        attribution_figure.update_layout(font=dict(family="Segoe UI, sans-serif", size=12, color="#111"), plot_bgcolor="#ffffff", paper_bgcolor="#ffffff")
    except Exception:
        pass

    mean_beam_voltage = metrics_snapshot.get("Input_Voltage_mean")
    beam_variance = metrics_snapshot.get("Input_Voltage_var")
    mean_control_effort = metrics_snapshot.get("Pin_Control_Changes_mean")
    saturation_indicators = metrics_snapshot.get("Saturation_Indicators_total", 0)

    beam_mean_text = f"{mean_beam_voltage:.3f} V" if mean_beam_voltage is not None else "--"
    beam_var_text = f"{beam_variance:.5f}" if beam_variance is not None else "--"
    effort_text = f"{mean_control_effort:.2f}" if mean_control_effort is not None else "--"
    saturation_text = f"{int(saturation_indicators)}"
    # Model status: compute trained state and age text
    model_status_text = "Model: not trained"
    trained = False
    age_text = ""
    try:
        mi = get_model_info()
        sec = mi.get("last_train_ago_sec")
        if sec is not None:
            sec = float(sec)
            trained = True
            if sec < 60:
                age_text = f"updated {sec:.0f}s ago"
            elif sec < 3600:
                age_text = f"updated {sec/60:.1f} min ago"
            else:
                age_text = f"updated {sec/3600:.1f} h ago"
    except Exception:
        pass
    state_color = "green" if trained else "red"

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
        html.Span("Beam mean:", style=label_style), html.Span(beam_mean_text, style=value_style)
    ], style=card_style)

    mean_variance_outputs = html.Div([
        html.Span("Beam var:", style=label_style), html.Span(beam_var_text, style=value_style)
    ], style=card_style)

    mean_control_effort_outputs = html.Div([
        html.Span("Effort (avg ||du||^2):", style=label_style), html.Span(effort_text, style=value_style)
    ], style=card_style)

    saturation_indicator_outputs = html.Div([
        html.Span("Saturations (window):", style=label_style), html.Span(saturation_text, style=value_style)
    ], style=card_style)

    model_status_outputs = html.Div(
        [
            html.Span("Model:", style=label_style),
            html.Span("trained" if trained else "not trained", style={"fontWeight": "bold", "color": state_color}),
            html.Span(f"  ({age_text})" if trained else "", style={"marginLeft": "0.4em", "color": "#333"}),
        ],
        style={**card_style, "border": f"1px solid {state_color}"},
    )

    return (
        beam_figure,
        effort_figure,
        attribution_figure,
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








