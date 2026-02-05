
"""
#----------- Data logger for ESP32 serial instrument. -------------#

#   Python module which reads measurement data from the backend queue, 
#   logs to CSV, and visualizes the voltage in real time. 

#----------------------------------------------------------------------#
"""

import time
import pandas as pd
import logging
import matplotlib.pyplot as plt
from python_Backend import lines

# -------------------------------------------------------------------------
#                           Logging configuration
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("DataLogger")

# -------------------------------------------------------------------------
#                       Configuration constants
# -------------------------------------------------------------------------

AUTOSAVE_INTERVAL = 10  # seconds
MAX_ROWS = 10_000
ENABLE_PLOT = True

timestamp_str = time.strftime("%Y%m%d_%H%M%S")
FILE_NAME = f"DataLog_{timestamp_str}.csv"

# -------------------------------------------------------------------------
#                           Data storage setup
# -------------------------------------------------------------------------

data_frame = pd.DataFrame(columns=["timestamp", "voltage", "raw_message"])
buffer = []
last_save = time.time()

if ENABLE_PLOT:
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 4))
    line, = ax.plot([], [], lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Live Voltage Trace ({timestamp_str})")

# -------------------------------------------------------------------------
#                           Main logging loop
# -------------------------------------------------------------------------

try:
    log.info("=" * 60)
    log.info("ES12F Data Logger started")
    log.info(f"Saving to {FILE_NAME} every {AUTOSAVE_INTERVAL}s")
    log.info("=" * 60)

    while True:
        try:
            ts, msg = lines.get(timeout=1.0)
        except Exception:
            continue

        voltage = None
        if msg.startswith("MEASURED"):
            try:
                voltage = float(msg.split()[1])
            except (IndexError, ValueError):
                pass

        buffer.append((ts, voltage, msg))

        if time.time() - last_save > AUTOSAVE_INTERVAL:
            new_data = pd.DataFrame(buffer, columns=["timestamp", "voltage", "raw_message"])
            data_frame = pd.concat([data_frame, new_data], ignore_index=True).tail(MAX_ROWS)
            data_frame.to_csv(FILE_NAME, index=False)
            log.info(f"Autosaved {len(data_frame)} rows → {FILE_NAME}")
            buffer.clear()
            last_save = time.time()

        if ENABLE_PLOT and voltage is not None:
            line.set_data(data_frame["timestamp"], data_frame["voltage"])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.05)

except KeyboardInterrupt:
    log.warning("Logging interrupted by user.")
    if buffer:
        pd.DataFrame(buffer, columns=["timestamp", "voltage", "raw_message"]).to_csv(
            FILE_NAME, mode="a", index=False
        )
    log.info("Final data saved before exit.")

except Exception as Exception:
    log.exception(f"Unexpected error during logging: {Exception}")
    if buffer:
        pd.DataFrame(buffer, columns=["timestamp", "voltage", "raw_message"]).to_csv(
            FILE_NAME, mode="a", index=False
        )
    log.info("Emergency data dump complete.")

