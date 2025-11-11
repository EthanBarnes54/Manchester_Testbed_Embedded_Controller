
    # ----------     ESP32 Control & Monitoring Dashboard     ---------- # 

    # A Dash web interface for real-time control and data visualization
    # of the ESP32-based embedded system. Communicates via python_backend.

    #----------------------------------------------------------------------#

from dash import Dash, dcc, html, Input, Output, State, ctx
import plotly.graph_objs as go
import pandas as pd
import pkgutil, importlib.util

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

app = Dash(__name__, title="ESP32 Control Dashboard")
server = app.server  # Expose Flask server if deploying later

# -------------------------------------------------------------------------
#                                     Layout
# -------------------------------------------------------------------------

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, sans-serif", "padding": "2em"},
    children=[
        html.H2("ESP32 Real-Time Control Dashboard", style={"textAlign": "center"}),

        dcc.Graph(id="live-graph", style={"height": "60vh"}),

        html.Div(
            style={"marginTop": "2em", "display": "flex", "alignItems": "center", "gap": "1em"},
            children=[
                html.Label("Set DAC Voltage (V):", style={"fontWeight": "bold"}),
                dcc.Slider(
                    id="voltage-slider",
                    min=0.0,
                    max=3.3,
                    step=0.01,
                    value=1.0,
                    marks={0: "0", 1: "1", 2: "2", 3: "3", 3.3: "3.3"},
                    tooltip={"always_visible": False, "placement": "bottom"},
                    updatemode="drag",
                ),
                html.Button("Send", id="send-btn", n_clicks=0, style={"padding": "0.5em 1em"}),
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
                html.Div(id="status", children="Status: Connected"),
            ],
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


@app.callback(
        
    Output("ack", "children"),
    Output("ack", "style"),
    Input("send-btn", "n_clicks"),
    State("voltage-slider", "value"),
    prevent_initial_call=True,
)

def send_voltage_command(n_clicks, voltage):

    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    try:
        reader.send_command(f"SET {voltage:.2f}")
        log.info(f"Sent voltage command: {voltage:.2f} V")
        return (
            f"Sent {voltage:.2f} V successfully.",
            {"color": "green", "fontWeight": "bold"},
        )
    except Exception as e:
        log.error(f"Command failed: {e}")
        return (
            f"Error sending voltage: {str(e)}",
            {"color": "red", "fontWeight": "bold"},
        )

# -------------------------------------------------------------------------
#                               Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("Launching ESP32 Control Dashboard...")
    app.run(debug=False, host="0.0.0.0", port=8050)
