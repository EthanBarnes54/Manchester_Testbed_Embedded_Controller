
    # ----------     ESP32 Control & Monitoring Dashboard     ---------- # 

    # A Dash based web interface for real-time control and data visualization
    # of the ESP32F-based embedded control system. Communicates with hardware
    # via python_backend and recieves inputs from the RNN system for optimised 
    # outputs to the board.

    # http://127.0.0.1:8050  <- past into browser for dashboard access

    #----------------------------------------------------------------------#

from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pandas as pd
import pkgutil, importlib.util
import numpy as np
import logging
from python_backend import (
    Back_End_Controller,
    start_training_sweep,
    get_sweep_status,
    stop_training_sweep,
    get_ml_metrics,
)
import dash

# Temporary shim for Python 3.14 (pkgutil.find_loader has been removed)

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

# -------------------------------------------------------------------------
#                                     Layout
# -------------------------------------------------------------------------

def _control_tab():
    return html.Div(
        children=[
            html.H2("ESP12-F Control Dashboard", style={"textAlign": "center"}),
            dcc.Graph(id="live-graph", style={"height": "60vh"}),
            html.Div(
                style={"marginTop": "2em", "display": "flex", "alignItems": "center", "gap": "1em"},
                children=[
                    html.Label("Set DAC Voltage (V):", style={"fontWeight": "bold"}),
                    dcc.Input(id="voltage-input", type="number", min=0.0, max=3.3, step=0.000001, value=1.0, debounce=True, style={"width": "140px"}),
                    html.Button("Send Command", id="send-btn", n_clicks=0, style={"padding": "0.5em 1em"}),
                    html.Span(id="ack", style={"marginLeft": "1em", "fontWeight": "bold"}),
                ],
            ),
            html.Div(
                style={"marginTop": "1em"},
                children=[
                    html.H4("Pin Controls: 5 PWM + 1 Logic Switch"),
                    html.Div(
                        style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "1em"},
                        children=[
                            html.Div([html.Label("Squeeze Plate (0-1023)"), dcc.Input(id="pwm1", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Ion Source (0-1023)"), dcc.Input(id="pwm2", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Wein Filter (0-1023)"), dcc.Input(id="pwm3", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Upper Cone (Initial/Entry Cone) (0-1023)"), dcc.Input(id="pwm4", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Lower Cone (Final/Exit Cone) (0-1023)"), dcc.Input(id="pwm5", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"})]),
                            html.Div([html.Label("Switch Logic"), dcc.Checklist(id="sw6", options=[{"label": "ON", "value": "on"}], value=[], inputStyle={"marginRight": "0.5em"})]),
                        ],
                    ),
                    html.Div(id="pins-ack", style={"marginTop": "0.5em", "fontWeight": "bold"}),
                ],
            ),
            html.Div(
                id="metrics-bar",
                style={"display": "flex", "justifyContent": "space-around", "marginTop": "2em", "padding": "1em", "borderTop": "1px solid #ddd", "color": "#333"},
                children=[
                    html.Div(id="latest-voltage", children="Voltage: -- V"),
                    html.Div(id="num-points", children="Samples: --"),
                    html.Div(id="status", children="OFFLINE", style={"color": "red", "fontWeight": "bold"}),
                ],
            ),
            html.Div(id="pin-status", style={"display": "flex", "flexWrap": "wrap", "gap": "0.5em 1em", "marginTop": "0.5em"}),
            html.Div(
                style={"marginTop": "1.5em", "padding": "1em", "border": "1px solid #e0e0e0", "borderRadius": "8px", "maxWidth": "740px", "background": "#fafafa"},
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
                    html.Div(id="ml-sats", style={"minWidth": "220px"}),
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


app.layout = html.Div(
    style={"fontFamily": "Segoe UI, sans-serif", "padding": "2em"},
    children=[
        dcc.Tabs(id="tabs", value="control", children=[
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
)

def update_graph(_):
    
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

    return figure, recent_voltage_measurement, number_of_points


@app.callback(
    Output("ack", "children"),
    Output("ack", "style"),
    Input("send-btn", "n_clicks"),
    State("voltage-input", "value"),
    prevent_initial_call=True,
)

def send_voltage_command(n_clicks, value):
    if not n_clicks:
        raise PreventUpdate
    try:
        if value is None:
            return "Enter a value", {"color": "#333"}
        v = float(value)
        if not (0.0 <= v <= 3.3):
            return "Out of range (0.0–3.3 V)", {"color": "red", "fontWeight": "bold"}
        Back_End_Controller.send_command(f"SET {v:.6f}")
        return f"Sent: SET {v:.6f}", {"color": "green", "fontWeight": "bold"}
    except Exception as fault:
        return f"Error: {fault}", {"color": "red", "fontWeight": "bold"}

# -------------------------------------------------------------------------
#                     Input validation (UI Interface)
# -------------------------------------------------------------------------

@app.callback(
    Output("voltage-input", "style"),
    Input("voltage-input", "value"),
)

def _validate_voltage_input(val):
    base = {"width": "140px"}
    if val is None:
        return base
    try:
        fval = float(val)
    except Exception:
        return {**base, "border": "2px solid red", "boxShadow": "0 0 4px rgba(255,0,0,0.6)"}
    if 0.0 <= fval <= 3.3:
        return base
    return {**base, "border": "2px solid red", "boxShadow": "0 0 4px rgba(255,0,0,0.6)"}

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
def update_pins(v1, v2, v3, v4, v5, sw6):
    try:
        if v1 is not None:
            Back_End_Controller.set_pwm(1, int(v1))
        if v2 is not None:
            Back_End_Controller.set_pwm(2, int(v2))
        if v3 is not None:
            Back_End_Controller.set_pwm(3, int(v3))
        if v4 is not None:
            Back_End_Controller.set_pwm(4, int(v4))
        if v5 is not None:
            Back_End_Controller.set_pwm(5, int(v5))

        is_on = isinstance(sw6, (list, tuple)) and ("on" in sw6)
        Back_End_Controller.set_switch(bool(is_on))

        msg = (
            f"Pins updated: squeeze_plate={v1}, ion_source={v2}, wein_filter={v3}, "
            f"cone_1={v4}, cone_2={v5}, switch_logic={'ON' if is_on else 'OFF'}"
        )
        return msg, {"color": "green", "fontWeight": "bold"}
    except Exception as fault:
        return f"Error updating pins: {fault}", {"color": "red", "fontWeight": "bold"}

# -------------------------------------------------------------------------
#                               Pin statuses
# -------------------------------------------------------------------------

@app.callback(
    Output("pin-status", "children"),
    Input("update-interval", "n_intervals"),
)

def update_pin_status(_):
    try:
        status_text = getattr(Back_End_Controller, "get_status", lambda: "Connecting...")()
    except Exception:
        status_text = "Connecting..."

    pins = getattr(Back_End_Controller, "get_pins", lambda: {"names": [], "values": [], "timestamp": 0.0})()
    names = pins.get("names", [])
    values = pins.get("values", [])

    if "Connected" in str(status_text):
        try:
            Back_End_Controller.send_command("PINS")
        except Exception:
            pass

    def chip(label, val, state):
        color = {
            "CONNECTED": "green",
            "DISCONNECTED": "red",
        }.get(state, "#333")
        return html.Div(
            style={
                "border": f"1px solid {color}",
                "borderRadius": "6px",
                "padding": "0.25em 0.5em",
                "minWidth": "180px",
            },
            children=[
                html.Span(f"{label}: ", style={"fontWeight": "bold"}),
                html.Span(str(val)),
                html.Span(f"  ({state})", style={"marginLeft": "0.4em", "color": color}),
            ],
        )

    if not names or not values:
        return [chip("Pins", "--", "DISCONNECTED")]

    state = "DISCONNECTED"
    if str(status_text).startswith("Simulated"):
        state = "DISCONNECTED"

    disp_vals = list(values)
    if len(disp_vals) >= 6:
        disp_vals[5] = "ON" if int(disp_vals[5]) else "OFF"

    chips = [chip(n, disp_vals[i] if i < len(disp_vals) else "--", state) for i, n in enumerate(names)]
    return chips


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

def handle_training_sweep(_, start_clicks, stop_clicks, min_v, max_v, step, dwell_s, epochs):
    
    triggered_id = None
    
    try:
        ctx = dash.callback_context

        if ctx and ctx.triggered:
            triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    except Exception:
        triggered_id = None

    if triggered_id == "sweep-btn" and start_clicks and int(start_clicks) > 0:

        try:
            min_v = 0.0 if min_v is None else float(min_v)
            max_v = 3.3 if max_v is None else float(max_v)
            step = 0.05 if step is None or step <= 0 else float(step)
            dwell_s = 0.05 if dwell_s is None or dwell_s < 0 else float(dwell_s)
            epochs = 10 if epochs is None or int(epochs) <= 0 else int(epochs)

            started = start_training_sweep(min_v=min_v, max_v=max_v, step=step, dwell_s=dwell_s, epochs=epochs)

            if started:
                log.info("Training sweep initiated by user...")
        except Exception as fault:
            log.error(f"Failed to start training sweep: {fault}")

    elif triggered_id == "sweep-stop-btn" and stop_clicks and int(stop_clicks) > 0:
        try:
            stop_training_sweep()
            log.info("Training sweep cancelled by user...")
        except Exception as fault:
            log.error(f"Failed to stop training sweep: {fault}")

    try:
        status = get_sweep_status()
    except Exception:
        status = {"state": "unknown", "progress": 0.0, "message": ""}

    state = str(status.get("state", "idle"))
    progress = float(status.get("progress", 0.0))
    message = str(status.get("message", ""))

    percent_completion = int(max(0.0, min(1.0, progress)) * 100)
    text = f"Sweep: {state} ({percent_completion}%)" + (f" — {message}" if message else "")
    disabled = state == "running"

    if state == "running":
        color = "#3498db"  
    elif state == "completed":
        color = "#2ecc71"  
    elif state == "failed":
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
    Output("ml-sats", "children"),
    Input("update-interval", "n_intervals"),
)
def update_ml_tab(_):
    try:
        snap = get_ml_metrics()
    except Exception:
        snap = {}

    t = snap.get("time_series", [])
    v = snap.get("beam_series", [])
    du2 = snap.get("control_effort_series", [])
    sats = snap.get("saturations_series", [])

    # Beam graph
    if t and v:
        beam_fig = go.Figure(data=[go.Scatter(x=pd.to_datetime(t, unit="s"), y=v, mode="lines", line=dict(color="#0066cc"))])
        beam_fig.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30), xaxis_title="Time", yaxis_title="Diode voltage (V)")
    else:
        beam_fig = go.Figure()

    # Effort graph
    if t and du2:
        eff_fig = go.Figure(data=[go.Scatter(x=pd.to_datetime(t, unit="s"), y=du2, mode="lines", line=dict(color="#cc6600"))])
        eff_fig.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30), xaxis_title="Time", yaxis_title="Control effort (||Δu||^2)")
    else:
        eff_fig = go.Figure()

    # Attribution bar (saliency / shap placeholder)
    names = snap.get("feature_names", [])
    sal = snap.get("saliency", [])
    if names and sal and len(names) == len(sal):
        attrib_fig = go.Figure(data=[go.Bar(x=names, y=np.abs(np.array(sal)).tolist(), marker_color="#2c3e50")])
        attrib_fig.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30), yaxis_title="|Attribution|")
    else:
        attrib_fig = go.Figure()
        attrib_fig.add_annotation(text="Attribution unavailable (attach pipeline+controller or install SHAP)", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        attrib_fig.update_layout(template="plotly_white", margin=dict(l=40, r=10, t=30, b=30))

    bm = snap.get("beam_mean")
    bv = snap.get("beam_var")
    em = snap.get("effort_mean")
    st = snap.get("saturations_total", 0)

    bm_txt = f"Beam mean: {bm:.3f} V" if bm is not None else "Beam mean: --"
    bv_txt = f"Beam var: {bv:.5f}" if bv is not None else "Beam var: --"
    em_txt = f"Effort (avg ||Δu||^2): {em:.2f}" if em is not None else "Effort: --"
    st_txt = f"Saturations (window): {int(st)}"

    return beam_fig, eff_fig, attrib_fig, bm_txt, bv_txt, em_txt, st_txt

# -------------------------------------------------------------------------
#                               Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":

    log.info("Launching ESP-12F Control Dashboard...")
    app.run(debug=False, host="0.0.0.0", port=8050)
