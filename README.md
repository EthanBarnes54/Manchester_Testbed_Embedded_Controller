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

- Output safety: all voltage setters clamp to 0-3.3 V (10-bit resolution, according to board specs). Switch timing is the time between edges (half a cycle), bounded to 1 us - 2 s. 1 us (a 500 kHz square wave) is the design floor; it has not yet been scoped on hardware. Up to 1 ms it is generated by the LEDC peripheral with no CPU work per edge; above that a hardware timer interrupt toggles the pin.

- Testability: `OFFLINE` simulation enables UI/ML testing without hardware; heartbeats and ACK frames help verify connectivity in the software.

- OTA-ready firmware: ArduinoOTA hooks are included for cable-free updates once Wi-Fi credentials are set.

- Metrics separation: `MetricCollector` isolates dashboard-friendly summaries from raw data, keeping the UI lightweight.

## Switch output stage (hardware)
The switch line reaches its load through an AND gate that the firmware has to enable, so nothing GPIO 16 does before the firmware is running can reach the load.

```
 GPIO 16 (switch_logic) ──────────┐
                                  ├─ AND ── Y ──┬── to the load
 GPIO 23 (ARMED) ──┬──────────────┘             │
                 10 kΩ                       10-100 kΩ
                   │                            │
                  GND                          GND
```

- Gate: 74LVC1G08 on 3.3 V. If the load needs 5 V logic, use a 74AHCT1G08 on 5 V: its TTL-level inputs accept the ESP32's 3.3 V, which a 5 V-powered 74LVC part does not reliably do.
- The 10 kΩ pull-down holds ARMED low from reset until the firmware drives it. The second pull-down keeps the gate output low if the gate is unpowered.
- Why it is needed: GPIO 16 is undriven from reset, and the Arduino core's PSRAM probe uses it as a chip-select during boot, before any of this firmware runs. The gate stays closed through all of that.
- Firmware behaviour: ARMED is driven low first thing in `setup()` and goes high only on an explicit `ARM` (see "Operating modes"). DISARM, the failsafe (host silent for 5 s) and any critical fault drop it before anything else. The switch's own state machine never touches ARMED.
- External watchdog (recommended): GPIO 18 (HEARTBEAT) toggles on every loop pass. Feed it to a watchdog supervisor such as a TPS3823 and AND its output into the gate (a 3-input 74LVC1G11 in place of the 74LVC1G08), so the gate closes within the watchdog's timeout if the firmware stops, independently of the ESP32. With nothing fitted the pin is harmless.

### Still to be decided before relying on 1 us
- **What the gate output drives. NOT YET DECIDED.** It could be an on-board driver a few centimetres away, or a cable to the HV switch. This decides:
  - whether the gate can drive the load directly, or needs a gate driver or line driver after it;
  - whether a series resistor at the gate output is needed to damp ringing on a cable;
  - which gate variant, 3.3 V or 5 V logic, from the load's input levels.

  At 1 us between edges, the edge at the load has to be well under 100 ns: a 100 ns edge is a tenth of each half-period. The ESP32 pin only drives the gate input on the same board, so its own drive strength is left at the default. Decide the load, then scope the edge at the load at 1 us.
- **A filter on the ADC input.** At 1 us the switching frequency, 500 kHz, is exactly twice the ADS1115's 250 kHz modulator rate, which its digital filter does not reject (TI datasheet, section 9.1.5). If the switching modulates the diode signal, or couples into the AIN0 wiring, readings can show an offset or a slow wander. The datasheet's remedy is a first-order RC low-pass at the input, with resistors under 1 kOhm (section 9.2.2.6). With readings taken at 20 Hz, a cutoff around 1 kHz costs nothing.

## Output verification (hardware)
Two checks compare the outputs with what was commanded. Both are compiled into every build but stay off until a build flag says their hardware is fitted, so a bench without it cannot raise false faults:

| Check | Hardware | Build flag | Faults (critical) |
|---|---|---|---|
| Gate loopback | The gate output Y wired back to GPIO 34 (input-only), through a 1 kOhm series resistor | `-DTESTBED_GATE_LOOPBACK=1` | `GATE_MISMATCH`: the gate output is high while disarmed, does not follow a held level, or passes edges while closed. `SWITCH_FREQUENCY`: the edges the pulse counter sees over a window of about four cycles (at least 20 ms) differ from the commanded period by more than 2 edges or 0.5% |
| Setpoint readback | Setpoints 1-3 (squeeze_plate, ion_source, wein_filter) wired to ADS1115 AIN1-AIN3 through the same filter as the outputs | `-DTESTBED_SETPOINT_READBACK=1` | `SETPOINT_MISMATCH`: three settled readings in a row differ from the command by more than 0.15 V |

- The readback scale (`READBACK_VOLTS_PER_VOLT`), tolerance and settling time are constants in `main.cpp`. Set them to match the board's network before enabling the check. cone_1 and cone_2 need a second ADS1115 at address 0x49 to be covered.
- Readback conversions fit in the gap after each diode conversion, so the 20 Hz `MEASURED` stream is unchanged. Each setpoint is read about every 150 ms.
- `SELFTEST` and `HEALTH` report `gate=` and `readback=` as `ok`, `FAIL`, or `off` when not fitted.

## Operating modes
The board is always in one of three modes, reported by `FAULTS` and shown on the dashboard:

| Mode | Outputs | How it is entered | How it is left |
|---|---|---|---|
| SAFE | Setpoints zero, switch held low, ARMED low | Boot, `DISARM`, the failsafe | `ARM` |
| ARMED | Live; `TARGETS`, `PIN` and `SWITCH_PERIOD_US` accepted | `ARM`, only if no critical fault is latched | `DISARM`, the failsafe, a critical fault |
| FAULT | As SAFE; `ARM` refused | A critical fault, including a reset by watchdog, panic or brownout | `CLEAR FAULTS`, once the fault's condition has gone |

- Commands that would drive an output are answered `ERROR: Not armed!` unless the board is ARMED. Commands that set an output to zero are always accepted.
- The host coming back after a failsafe does not re-arm the board; the operator presses ARM again.
- `FAULTS` reports the mode, active, latched and historical faults, counts, the boot count and the last reset reason. The history and the reset counters survive power cycles; `CLEAR LOG` resets them.
- A loop that stops for 5 s resets the chip (the loop watchdog), which comes back up in FAULT with the reset recorded. OTA uploads are only accepted while SAFE.
- On the dashboard, ARM asks for confirmation. Sweeps and auto control refuse to run unless the board is ARMED, and closing the backend cleanly sends DISARM.

## Deployment and security
- **Deploy build.** `pio run -e esp32_deploy -t upload` builds what goes on the rig: Wi-Fi and OTA compiled out (nothing listening; the image is about 26% of flash against 65%), and both output verification checks on. It expects the loopback and readback hardware to be fitted.
- **OTA rollback.** After an OTA update the bootloader runs the new image pending verification. It keeps the image only if the power-on self-test shows it brought up its own clocks and switch generators; otherwise it goes back to the previous image. OTA is only accepted while the board is SAFE. Secure boot and flash encryption are **not** enabled: they burn one-time eFuses and need a board set aside for the procedure.
- **Dashboard access.** Off loopback the dashboard refuses to start without a password and TLS (`DASHBOARD_TLS_CERT`, `DASHBOARD_TLS_KEY`), unless `DASHBOARD_ALLOW_PLAINTEXT=1` deliberately accepts the risk. A second, read-only login (`DASHBOARD_OBSERVER_USER`, `DASHBOARD_OBSERVER_PASSWORD`) can watch everything and can DISARM or stop a sweep, but cannot make anything live.
- **Audit trail.** Every operator action on the dashboard (with user, role and address), every refused observer action, and every ARM, DISARM and CLEAR FAULTS the backend sends are appended as JSON lines to `audit_log.jsonl` (or `TESTBED_AUDIT_LOG`).
- **Provenance.** A saved sweep dataset gets a `.json` sidecar with the board's VERSION, the backend's git revision, the sweep settings, the switch period and the CSV's SHA-256. A model checkpoint carries the hash of the data it was trained on, alongside its validation score.
- **SBOM.** `python tools/generate_sbom.py` writes a CycloneDX 1.5 bill of materials of every pinned Python package, the PlatformIO platform and libraries, and the installed Arduino core. CI builds all three firmware environments and keeps the images and the SBOM as artefacts.

## Auto control safeguards
Auto control lets the RNN steer the rig unattended, so `python_Autonomy_Guard.py` sits between its proposals and the board as an independent monitor with rules simple enough to check by reading:

- **A validated model only.** Auto control will not switch on unless the current weights earned a held-out R² of at least 0.5 when last trained. The score is saved with the weights in every checkpoint, and online updates since then are counted.
- **Envelope and rate limits.** Every proposal is held within 0-3.3 V per channel and moves each channel by at most 0.25 V per decision. A proposal that is not five finite numbers is rejected and nothing is sent.
- **Drift.** If the last 2 s of inputs sit more than four training standard deviations from the data the model was trained on, nothing new is sent.
- **Fallback.** Whenever a proposal is rejected or the inputs have drifted, the guard holds the targets already on the rig.
- **Operator override.** A manual target edit on the dashboard switches auto control off.
- Auto control also needs the board ARMED (see "Operating modes"). The guard's counts of accepted, limited, rejected and drift-held decisions are reported with the auto control state.

## Serial link integrity
The full interface, every command, reply and fault, is in [docs/protocol.md](docs/protocol.md).

- **Framing.** Every line in both directions ends with `*XXXX`, a CRC-16/CCITT-FALSE of the text (`include/line_protocol.h`; `binascii.crc_hqx` on the backend). The board refuses a command whose CRC does not match (`ERROR: Bad checksum!`, fault `BAD_CHECKSUM`), and such a line does not count as the host being alive. Lines typed by hand at a terminal carry no CRC and are accepted. Once the board has confirmed its protocol, the backend drops corrupted lines and protocol lines without a CRC, and counts both.
- **Version handshake.** `VERSION` answers `firmware=<git describe> protocol=<n> build=<env> gate_loopback=<0|1> setpoint_readback=<0|1>`. The firmware is stamped at build time by `tools/firmware_version.py`. The backend asks on connect and will not arm a board whose protocol differs from its own `PROTOCOL_VERSION`.
- **Readings.** `MEASURED <volts> V seq=<n> t_ms=<board ms>`. `seq` counts from 1 at boot, so the backend counts readings that went missing and notices a board restart.
- **Replies.** The backend tracks every command it sends against the reply it expects, and counts replies, rejections (`ERROR`) and commands left unanswered for more than 1 s. The dashboard's link line shows the firmware, the protocol check and these counts.

## Built-in test
- Power-on self-test, repeated on demand by `SELFTEST`: checks the clocks the switch timing assumes (APB 80 MHz, REF_TICK = APB / 80), both switch generators, the ADC and memory. It answers `SELFTEST PASS|FAIL clocks=.. switch=.. adc=.. memory=..`. A clock or switch-generator failure is critical and leaves the board in FAULT.
- Continuous self-test, every loop pass: each pass is timed against a 20 ms budget, the ADC is watched for timeouts, and heap and loop-task stack are checked against floors once a second. Each raises its fault (`LOOP_OVERRUN`, `ADC_LOST`, `LOW_MEMORY`); over-long command lines raise `SERIAL_OVERFLOW`.
- `HEALTH` reports the worst loop pass since the last report and since boot, the budget, overruns, free and minimum heap, loop-task stack headroom, ADC conversions and timeouts, and serial overflows. The backend polls it every 2 s and the dashboard shows a summary. This is how "the checks do not slow the board down" is measured on the bench.

## Verification
What the system must do is in [docs/requirements.md](docs/requirements.md), each requirement with an ID and how it is verified. Every test that verifies one is tagged `@pytest.mark.req("<ID>")`, and the suite fails if a requirement has no test or a test names an unknown ID.

From `Embedded Controller/Backend`, with `requirements-dev.txt` installed:

| What | Command | In CI |
|---|---|---|
| Static checks | `ruff check ..` | yes |
| Tests, with the coverage floor in `.coveragerc` | `python -m pytest --cov` | yes |
| Traceability matrix | `python ../tools/traceability.py` | yes, kept as an artefact |
| Mutation tests (a few minutes) | `python -m pytest -m mutation` | yes, own job |
| Rig tests (a board on the port) | `TESTBED_HIL_PORT=COM5 python -m pytest hil` | no; see [hil/README.md](Embedded%20Controller/Backend/hil/README.md) |

- **Host harnesses.** The firmware's switch, ADC, safety, output-verification and link code is compiled from `main.cpp` and `include/` with g++ and run on the PC, with every warning an error and under UndefinedBehaviorSanitizer (and AddressSanitizer in CI, where its runtime exists). `TESTBED_HOST_SANITIZE=0` turns the sanitizers off.
- **Fuzzing.** Property tests (Hypothesis) feed the backend's line parsers malformed and corrupted input, and cross-check the firmware's CRC check against the backend's on generated lines.
- **Mutation tests.** Each case breaks one safety check on purpose (a refusal dropped, an order swapped, a fault demoted) and the suite must fail. The rig tests are checked the same way against a simulated board (`TESTBED_HIL_PORT=sim`) told to misbehave.
- **Firmware gates.** The project's own sources build with every warning an error, including the ones the Arduino core exempts. `python tools/check_firmware_size.py` fails an image over 80% of its application partition, which would leave no room for an update.

## Documentation
| Document | What it holds |
|---|---|
| [docs/requirements.md](docs/requirements.md) | Every requirement, with an ID and how it is verified |
| [docs/protocol.md](docs/protocol.md) | The serial interface: commands, replies, faults, framing |
| [docs/timing-budget.md](docs/timing-budget.md) | Where the loop's time goes, each deadline and its margin |
| [docs/safety/hazard-analysis.md](docs/safety/hazard-analysis.md) | Hazards and failure modes, each traced to its mitigations and requirements |
| [docs/acceptance-test-procedure.md](docs/acceptance-test-procedure.md) | The bench steps (ATP-01 to ATP-13), with tables for the results |
| [docs/adr/](docs/adr/README.md) | Why the design is the way it is: one record per decision |

`tests/test_docs.py` checks the documents against the code: every command the firmware accepts is in the protocol document, every fault with its severity, every requirement a hazard or a bench step names exists, and every rig test the procedure names exists.

## Troubleshooting
- Connection: confirm `SERIAL_PORT`/`upload_port` match the board; send a `PING` over serial to check link health.

- Dependencies: ensure PyTorch wheels match your Python version; reinstall Dash/Plotly if the UI fails to load. (Python 3.12 worked during development)

- No live data: enable `OFFLINE=1` to verify the UI flow; if using hardware, check ADS1115 wiring and that the board is streaming `MEASURED` frames.

- OTA: if OTA stalls, double-check Wi-Fi credentials and that the board hostname matches your network.

## License
Released under the MIT License. Copyright (c) 2025 - [Ethan Barnes]. Permission is granted, free of charge, to any person obtaining a copy of this software and associated documentation files to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the terms of the MIT License. See LICENSE for full text. Contributions and adaptations are welcome with attribution
