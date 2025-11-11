
    # ----------     ESP-12F Control & Monitoring Dashboard     ---------- # 

    # A Dash web interface for real-time control and data visualization
    # of the ESP12F-based embedded system. Communicates via python_backend.

    # http://127.0.0.1:8050  <- past into browser for dashboard access

    #----------------------------------------------------------------------#

from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import pandas as pd
import pkgutil, importlib.util
import numpy as np

# Temporary shim for Python 3.14 removal of pkgutil.find_loader

if not hasattr(pkgutil, "find_loader"):
    pkgutil.find_loader = lambda name: importlib.util.find_spec(name)

import logging
from python_backend import reader, start_training_sweep, get_sweep_status, stop_training_sweep
import dash

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

        html.Div(
            style={"marginTop": "2em", "display": "flex", "alignItems": "center", "gap": "1em"},
            children=[
                html.Label("Set DAC Voltage (V):", style={"fontWeight": "bold"}),
                dcc.Input(
                    id="voltage-input",
                    type="number",
                    min=0.0,
                    max=3.3,
                    step=0.000001,
                    value=1.0,
                    debounce=True,
                    style={"width": "140px"},
                ),
                html.Button("Send Command", id="send-btn", n_clicks=0, style={"padding": "0.5em 1em"}),
                html.Span(id="ack", style={"marginLeft": "1em", "fontWeight": "bold"}),
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

        dcc.Interval(id="update-interval", interval=1000, n_intervals=0),

        html.Div(

            style={
                "marginTop": "1.5em",
                "padding": "1em",
                "border": "1px solid #e0e0e0",
                "borderRadius": "8px",
                "maxWidth": "740px",
                "background": "#fafafa",
            },

            children=[
                html.Div(
                    children=[
                        html.H4("Training Sweep", style={"margin": 0}),
                        html.P(
                            "Generate the training dataset by sweeping over all ouput DAC values, then train the RNN controller.",
                            style={"margin": "0.25em 0 0.75em 0", "color": "#555"},
                        ),
                    ]
                ),

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


@app.callback(
        
    Output("ack", "children"),
    Output("ack", "style"),
    Input("send-btn", "n_clicks"),
    State("voltage-input", "value"),
    prevent_initial_call=True,
)

def send_voltage_command(n_clicks, voltage):

    if n_clicks == 0:
        raise PreventUpdate

    try:
        if voltage is None:
            raise ValueError("Please enter a voltage value.")

        v32 = float(np.float32(voltage))
        if not (0.0 <= v32 <= 3.3):
            raise ValueError("Voltage must be between 0.0 and 3.3 V.")

        reader.send_command(f"SET {v32:.2f}")
        log.info(f"Sent voltage command: {v32:.2f} V")
        return (
            f"Sent {v32:.2f} V successfully.",
            {"color": "green", "fontWeight": "bold"},
        )
    except Exception as e:
        log.error(f"Command failed: {e}")
        return (
            f"Error sending voltage: {str(e)}",
            {"color": "red", "fontWeight": "bold"},
        )

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
                log.info("Training sweep started by user from the dashboard...")
        except Exception as fault:
            log.error(f"Failed to start training sweep: {fault}")

    elif triggered_id == "sweep-stop-btn" and stop_clicks and int(stop_clicks) > 0:
        try:
            stop_training_sweep()
            log.info("Training sweep stop requested by user from the dashboard...")
        except Exception as fault:
            log.error(f"Failed to stop training sweep: {fault}")

    try:
        status = get_sweep_status()
    except Exception:
        status = {"state": "unknown", "progress": 0.0, "message": ""}

    state = str(status.get("state", "idle"))
    prog = float(status.get("progress", 0.0))
    msg = str(status.get("message", ""))

    pct = int(max(0.0, min(1.0, prog)) * 100)
    text = f"Sweep: {state} ({pct}%)" + (f" — {msg}" if msg else "")
    disabled = state == "running"

    if state == "running":
        color = "#3498db"  
    elif state == "completed":
        color = "#2ecc71"  
    elif state == "failed":
        color = "#e74c3c"  
    else:
        color = "#ccc"

    bar_style = {
        "height": "100%",
        "width": f"{pct}%",
        "background": color,
        "transition": "width 0.2s ease",
    }

    return text, disabled, bar_style

# -------------------------------------------------------------------------
#                               Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":

    log.info("Launching ESP-12F Control Dashboard...")
    app.run(debug=False, host="0.0.0.0", port=8050)
