# Manchester Testbed Embedded Controller
Adaptive embedded control and ML feedback stack for the Manchester Ion Beam Testbed. Combines ESP32 firmware, a Python control backend, a Plotly Dash dashboard, and an RNN optimiser to tune beamline voltages in real time.

## TLDR
- ESP32 firmware drives five PWM DAC channels plus a digital switch, streams ADS1115 ADC readings, and supports OTA + serial commands.
- Python backend (`SerialBackend`) keeps the serial link alive, buffers measurements, clamps outputs, orchestrates sweeps, and can simulate data when ``OFFLINE``.
- Dash dashboard surfaces live traces, pin controls, training sweeps, model status, and feature importance; it talks directly to the backend and RNN.
- RNN controller (PyTorch GRU) trains on pin/voltage history, supports online updates, manual checkpoint saves, and control vector proposals; signal pipeline + ML toolbox supply features and diagnostics.
- Logging/metrics modules provide CSV capture, rolling statistics, and telemetry for both the dashboard and standalone analysis; a manual firmware/backend variant remains as a minimal fallback.

## Repository Layout
- `Embedded Controller (Automatic)/src` - ESP32 firmware with ADC sampling, PWM outputs, OTA plumbing, and serial command handler.
- `Embedded Controller (Automatic)/include` - shared headers (`log.h` logging macros).
- `Embedded Controller (Automatic)/Backend` - Python control, dashboard, ML, logging, and metrics modules.
- `Embedded Controller (Automatic)/platformio.ini` - PlatformIO targets for release/debug builds.
- `Embedded Controller (Manual)` - stripped-down firmware + Python control/logging for manual operation (please note that manual control is also available via the automatic modules).
- `.vscode/` - editor/task hints; `LICENSE`, `.gitignore`, `README.md`.

## System architecture
1. Firmware samples the diode via ADS1115, updates PWM outputs (MCP4725-compatible), and emits `MEASURED` - pin states frames over serial.
2. `SerialBackend` maintains the serial connection, queues raw messages, converts pulses <-> voltages, clamps outputs, and caches a rolling DataFrame of measurements + pin states.
3. The Dash UI displays backend data stream for live traces, pin values, sweep status, ML metrics, and model state through live reports from the RNN; user inputs are pushed back as serial commands.
4. The RNN controller consumes recent pin/voltage sequences, trains offline or online, and can propose next-step control vectors via input voltage optimisation; feature saliencies can be requested on demand.
5. Metrics + logging modules stream summaries to the dashboard, autosave CSVs, and expose diagnostic graphs.
6. Optional manual stack mirrors the flow with fewer dependencies for bench-top fallback.



                            +--------------------+         Serial (MEASURED / PINS)         +----------------------+
                            |   ESP32 Firmware   |----------------------------------------->|    SerialBackend     |
                            | ADS1115 ADC        |<-- PWM targets / switch timing (cmds) ---| queue + clamp + DF  |
                            | MCP4725 PWM/Switch |                                          | (pin/voltage cache) |
                            +--------------------+                                          +----------+-----------+
                                                                                                |  | 
                                                                                                |  | ML metrics
                                                                                                |  v
                                                                                    +------------+-------------+
                                                                                    |      RNN Controller      |
                                                                                    | PyTorch GRU: train/online|
                                                                                    | propose_control_vector   |
                                                                                    +------------+-------------+
                                                                                                |
                                                                                                | proposals / saliency
                                                                                                v
                            +------------------+      REST/callbacks      +-----------------------+-------------------+
                            |     Dash UI      |<------------------------>| ML metrics, pin state, traces, sweeps, RNN |
                            | Live graphs      |                          | status; user inputs -> serial commands      |
                            | Controls & sweeps|--------------------------> (PIN/TARGETS/SWITCH) via backend           |
                            +------------------+                          +--------------------------------------------+
                                    ^
                                    | logs/CSV + plots
                                    |
                            +------------------+
                            | Metrics/Logging  |
                            | CSV, summaries   |
                            +------------------+




## File-by-file 
- `Embedded Controller (Automatic)/Backend/python_Backend.py` - core `SerialBackend` class; serial connect/retry, offline simulator, thread-safe queue and DataFrame, voltage/PWM clamping, pin setters, switch timing, sweep runner, online model updater, and ML metric hooks.

- `Embedded Controller (Automatic)/Backend/python_dashboard_script.py` - Plotly Dash UI; live voltage graph, pin controls, training sweep control, model status cards, ML diagnostics, feature importance trigger, and server bootstrap on port 8050.

- `Embedded Controller (Automatic)/Backend/python_Logging_Script.py` - lightweight logger reading the backend queue, autosaving CSVs, optional live matplotlib plot.

- `Embedded Controller (Automatic)/Backend/python_ML_Metrics.py` - `MetricCollector` for rolling time-series stats, training history, and feature/saliency caches for the dashboard.

- `Embedded Controller (Automatic)/Backend/python_ML_Toolbox.py` - shared ML utilities: scaling, windowing, metrics (R2/RMSE/MAE), stability checks, moving averages, outlier filtering, train/test split.

- `Embedded Controller (Automatic)/Backend/python_RNN_Controller.py` - PyTorch GRU model definition, training loop, online update, control proposal (`propose_control_vector`), learning-rate/momentum setters, checkpoint save/load, and permutation-based feature saliency.

- `Embedded Controller (Automatic)/Backend/python_Signal_Pipeline.py` - live pulse feature extractor (`LivePulsePipeline`) with normalization, segmentation, and engineered pulse metrics for downstream ML.

- `Embedded Controller (Automatic)/src/main.cpp` - ESP32 firmware: PWM + switch control, voltage-to-PWM conversion, ADC reads, OTA setup, serial command parser (`PIN`, `TARGETS`, `SWITCH_PERIOD_US`, `READ`, `PING`, `PINS`), heartbeat, and safety clamping.

- `Embedded Controller (Automatic)/include/log.h` - compile-time log macros with adjustable `LOG_LEVEL`.

- `Embedded Controller (Automatic)/platformio.ini` - PlatformIO environments (`esp32dev`, `esp32_debug`), library deps (ADS1115, MCP4725, ArduinoJson), serial/OTA settings, log level flags.

## Manual Architecture
- `Embedded Controller (Manual)/src/main.cpp` - simpler firmware using ADS1115 + MCP4725 DAC, PWM outputs, and basic serial commands (SET voltage, PIN, READ, PINS, PING) with OTA and heartbeats.

- `Embedded Controller (Manual)/Backend/python_backend.py` - pared-down serial bridge for manual control.

- `Embedded Controller (Manual)/Backend/python_dashboard_script.py` - minimal Dash UI for manual pin/voltage control.

- `Embedded Controller (Manual)/Backend/python_logging_script.py` - basic CSV logger.

## Key packages and objects
- Firmware: ESP32 Arduino framework (https://www.temu.com/goods.html?_bg_fs=1&goods_id=601100125178665&sku_id=17594836700139&_x_msgid=210-20251118-19-B-932956906741186560-427-jJ4J48hV&_x_src=mail&_x_sessn_id=a2imhcsf0h&refer_page_name=bgt_order_detail&refer_page_id=10045_1764090342505_wir48oejfa&refer_page_sn=10045), ADS1115 ADC, MCP4725 DAC, OTA, FreeRTOS timers.

- Python: Dash/Plotly, pandas/numpy, `pyserial`, PyTorch, scikit-learn, matplotlib. (Paradigms and techniques learned through DataCamp)

- Core objects: `SerialBackend` (backend synchronisation), `MetricCollector` (telemetry), `_RNN`/`RNNController` (model + pipeline integration), `LivePulsePipeline`/`Normalizer` (feature extraction to backend), Dash `app` layout/callbacks (UI + control panels).

## Setup
1. Firmware toolchain: install PlatformIO. Update Wi-Fi credentials in `src/main.cpp` if using OTA; ensure `upload_port` in `platformio.ini` matches the board (default `COM4`).

2. Build/flash firmware:
   - `cd "Embedded Controller (Automatic)"`
   - `pio run -t upload`
   - For debug logs, use the `esp32_debug` env (`pio run -e esp32_debug -t upload`).

3. Python environment (3.12 recommended, 'torch' not compatible with 3.14):
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
   - `pip install dash plotly pandas numpy pyserial torch scikit-learn matplotlib`

4. Backend config: set `SERIAL_PORT` in `python_Backend.py` to the correct COM port; set `OFFLINE=1` to simulate data without hardware.

## Running the stack
- Dashboard + backend:  
  `python "Embedded Controller (Automatic)/Backend/python_dashboard_script.py"` then open `http://127.0.0.1:8050/`. The backend spins up on import; use the dashboard to send commands and view live data.
- Data logging:  
  `python "Embedded Controller (Automatic)/Backend/python_Logging_Script.py"` to mirror backend messages into a rolling CSV (with optional live plot).

## RNN workflow
- Data collection: use the dashboard's training sweep panel to span voltage ranges (baseline grid, factorial combinations, random samples). Optional dataset CSV saving is available when `save_dataset_enabled` is toggled in the backend.

- Initial training: sweeps call `train_model` to fit the RNN (GRU's); losses and timestamps are tracked via `MetricCollector`.

- Updates: the backend periodically calls `online_update` over the recent time window (`set_window_update_time`) with adjustable learning rate/momentum. Status is shown in the dashboard model card.

- Control proposals: `propose_control_vector` evaluates candidate PWM/voltage sets against the trained model with change penalties; integrate where autonomous actuation is needed.

- Feature saliency: `compute_feature_importance` performs permutation/Shapley-style analysis over recent data to highlight influential pins.

## Design choices
- Non-blocking backend: serial IO runs on a background thread with bounded queues to avoid UI stalls; sample data frames are truncated to 1000 rows to cap memory usage.

- Output safety: all voltage setters clamp to 0-3.3 V (10-bit resolution, according to board specs). Switch timing is bounded (1-20 us) and gated through a hardware timer (to avoid jitters).

- Testability: `OFFLINE` simulation enables UI/ML testing without hardware; heartbeats and ACK frames help verify connectivity in the software.

- OTA-ready firmware: ArduinoOTA hooks are included for cable-free updates once Wi-Fi credentials are set.

- Metrics separation: `MetricCollector` isolates dashboard-friendly summaries from raw data, keeping the UI lightweight.

## Troubleshooting
- Connection: confirm `SERIAL_PORT`/`upload_port` match the board; send a `PING` over serial to check link health.

- Dependencies: ensure PyTorch wheels match your Python version; reinstall Dash/Plotly if the UI fails to load. (Python 3.12 worked during development)

- No live data: enable `OFFLINE=1` to verify the UI flow; if using hardware, check ADS1115 wiring and that the board is streaming `MEASURED` frames.

- OTA: if OTA stalls, double-check Wi-Fi credentials and that the board hostname matches your network.

## License
Released under the MIT License. Copyright (c) 2025 - [Ethan Barnes]. Permission is granted, free of charge, to any person obtaining a copy of this software and associated documentation files to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the terms of the MIT License. See LICENSE for full text. Contributions and adaptations are welcome with attribution
