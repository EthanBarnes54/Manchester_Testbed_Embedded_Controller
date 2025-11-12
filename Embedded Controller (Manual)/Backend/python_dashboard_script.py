
    # ----------     ESP12 F Control & Monitoring Dashboard     ---------- # 

    # A Dash web interface for real-time control and data visualization
    # of the ESP12F-based embedded system. Communicates via python_backend.

    # http://127.0.0.1:8050  <- past into browser for dashboard access

    #----------------------------------------------------------------------#

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import pkgutil, importlib.util
import time
import numpy as np

# Temporary shim for Python 3.14 removal of pkgutil.find_loader

if not hasattr(pkgutil, "find_loader"):
    pkgutil.find_loader = lambda name: importlib.util.find_spec(name)

import logging
from python_backend import reader

# -------------------------------------------------------------------------
#                                Logging setup
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("Dashboard")

# -------------------------------------------------------------------------
#                                 Initialize app
# -------------------------------------------------------------------------

app = Dash(__name__, title="ESP12-F Control Dashboard")
server = app.server  

# -------------------------------------------------------------------------
#                                     Layout
# -------------------------------------------------------------------------

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, sans-serif", "padding": "2em"},
    children=[
        html.H2("ESP12-F Control Dashboard", style={"textAlign": "center"}),

        dcc.Graph(id="live-graph", style={"height": "60vh"}),

        html.Hr(),

        html.Div(
            style={"marginTop": "1em"},
            children=[
                html.H4("Pin Controls (Manual): 5 PWM + 1 Logic Switch"),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "1em"},
                    children=[
                        html.Div([
                            html.Label("Squeeze Plate (0-1023)"),
                            dcc.Input(id="pwm1", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"}),
                        ]),
                        html.Div([
                            html.Label("Ion Source (0-1023)"),
                            dcc.Input(id="pwm2", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"}),
                        ]),
                        html.Div([
                            html.Label("Wein Filter (0-1023)"),
                            dcc.Input(id="pwm3", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"}),
                        ]),
                        html.Div([
                            html.Label("Upper Cone (Initial/Entry Cone) (0-1023)"),
                            dcc.Input(id="pwm4", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"}),
                        ]),
                        html.Div([
                            html.Label("Lower Cone (Final/Exit Cone) (0-1023)"),
                            dcc.Input(id="pwm5", type="number", min=0, max=1023, step=1, value=0, debounce=True, style={"width": "120px"}),
                        ]),
                        html.Div([
                            html.Label("Switch Logic"),
                            dcc.Checklist(
                                id="sw6",
                                options=[{"label": "ON", "value": "on"}],
                                value=[],
                                inputStyle={"marginRight": "0.5em"},
                            ),
                        ]),
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
            },
            children=[
                html.Div(id="latest-voltage", children="Voltage: -- V"),
                html.Div(id="num-points", children="Samples: --"),
                html.Div(id="status", children="OFFLINE", style={"color": "red", "fontWeight": "bold"}),
            ],
        ),

        html.Div(
            id="pin-status",
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "gap": "0.5em 1em",
                "marginTop": "0.5em",
            },
        ),

        dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
    ],
)

# -------------------------------------------------------------------------
#                                  Callbacks
# -------------------------------------------------------------------------

@app.callback(
    Output("live-graph", "figure"),
    Output("latest-voltage", "children"),
    Output("num-points", "children"),
    Input("update-interval", "n_intervals"),
)

def update_graph(_):
    
    df = reader.get_data()
    if df.empty:
        return go.Figure(), "Voltage: -- V", "Samples: 0"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(df["timestamp"], unit="s"),
            y=df["voltage"],
            mode="lines",
            line=dict(color="royalblue"),
        )
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Voltage (V)",
        margin=dict(l=50, r=20, t=40, b=40),
        template="plotly_white",
    )

    latest_voltage = f"Voltage: {df['voltage'].iloc[-1]:.3f} V"
    num_points = f"Samples: {len(df)}"
    return fig, latest_voltage, num_points

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
        status_text = getattr(reader, "get_status", lambda: "Connecting...")()
    except Exception:
        status_text = "Connecting..."

    if str(status_text).startswith("Simulated"):
        return "OFFLINE", {"color": "red", "fontWeight": "bold"}
    elif "Connected" in str(status_text):
        return "CONNECTED", {"color": "green", "fontWeight": "bold"}
    else:
        return "OFFLINE", {"color": "red", "fontWeight": "bold"}
    
# -------------------------------------------------------------------------
#                          Input Styles (UI Clarity)
# -------------------------------------------------------------------------

@app.callback(
    Output("pwm1", "style"),
    Output("pwm2", "style"),
    Output("pwm3", "style"),
    Output("pwm4", "style"),
    Output("pwm5", "style"),
    Input("pwm1", "value"),
    Input("pwm2", "value"),
    Input("pwm3", "value"),
    Input("pwm4", "value"),
    Input("pwm5", "value"),
)
def _validate_pwm_inputs(v1, v2, v3, v4, v5):
    def style_for(val, lo=0, hi=1023):
        base = {"width": "120px"}
        if val is None:
            return base
        try:
            fval = float(val)
        except Exception:
            return {**base, "border": "2px solid red", "boxShadow": "0 0 4px rgba(255,0,0,0.6)"}
        if lo <= fval <= hi:
            return base
        return {**base, "border": "2px solid red", "boxShadow": "0 0 4px rgba(255,0,0,0.6)"}

    return (
        style_for(v1),
        style_for(v2),
        style_for(v3),
        style_for(v4),
        style_for(v5),
    )

# -------------------------------------------------------------------------
#                             Pin control callback
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
            reader.set_pwm(1, int(v1))
        if v2 is not None:
            reader.set_pwm(2, int(v2))
        if v3 is not None:
            reader.set_pwm(3, int(v3))
        if v4 is not None:
            reader.set_pwm(4, int(v4))
        if v5 is not None:
            reader.set_pwm(5, int(v5))

        is_on = isinstance(sw6, (list, tuple)) and ("on" in sw6)
        reader.set_switch(bool(is_on))

        msg = (
            f"Pins updated: squeeze_plate={v1}, ion_source={v2}, wein_filter={v3}, "
            f"cone_1={v4}, cone_2={v5}, switch_logic={'ON' if is_on else 'OFF'}"
        )

        return msg, {"color": "green", "fontWeight": "bold"}
    except Exception as fault:
        return f"Error updating pins: {fault}", {"color": "red", "fontWeight": "bold"}

# -------------------------------------------------------------------------
#                         Pin status (connected/stale)
# -------------------------------------------------------------------------

@app.callback(
    Output("pin-status", "children"),
    Input("update-interval", "n_intervals"),
)
def update_pin_status(_):
    try:
        status_text = getattr(reader, "get_status", lambda: "Connecting...")()
    except Exception:
        status_text = "Connecting..."

    pins = getattr(reader, "get_pins", lambda: {"names": [], "values": [], "timestamp": 0.0})()
    names = pins.get("names", [])
    values = pins.get("values", [])
    
    if "Connected" in str(status_text):
        try:
            reader.send_command("PINS")
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
        state = "CONNECTED"

    disp_vals = list(values)
    if len(disp_vals) >= 6:
        disp_vals[5] = "ON" if int(disp_vals[5]) else "OFF"

    chips = [chip(n, disp_vals[i] if i < len(disp_vals) else "--", state) for i, n in enumerate(names)]
    return chips

# -------------------------------------------------------------------------
#                               Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Launching ESP32 Control Dashboard...")
    app.run(debug=False, host="0.0.0.0", port=8050)
