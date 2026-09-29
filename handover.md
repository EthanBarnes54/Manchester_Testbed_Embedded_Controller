# Handover - Manchester Testbed Embedded Controller

Status as of 2026-09-29, at `4.3.7`. Written for whoever takes the rig through bench testing and deployment.

The short version: sixteen defects have been fixed and pushed, and a number of features that had never actually worked are now wired in. **None of it has been run against real hardware.** Everything below was verified in software only, so the first job for whoever picks this up is bench validation, not more code.

---

## 1. Where the project stands

The stack is ESP32 firmware (PlatformIO) driving five PWM channels and one switched digital line, streaming ADS1115 readings over serial to a Python backend, a Plotly Dash dashboard on port 8050, and a PyTorch GRU controller.

Before this pass, `main` was broken at HEAD: a half-finished rename left `python_RNN_Controller.py` unable to import, and because `python_Backend.py` wraps that import in a bare `except Exception`, the whole ML stack was silently falling back to stubs. The dashboard loaded, reported no errors, and had no model behind it. That is fixed, along with a set of problems that would each have cost real beam time.

Both firmware environments build clean (`esp32dev` and `esp32_debug`, 63.5% of a 1.25 MB app partition, so OTA still fits). All seven Python modules compile, and the six importable ones import cleanly. `python_Logging_Script.py` is a standalone script rather than a module - it starts its logging loop at import - which is itself worth tidying.

## 2. What changed

Two passes, each commit independent and pushed on its own.

### Critical defects (`4.2.2` - `4.2.8`)

| Version | Change | Why it mattered |
|---|---|---|
| `4.2.2` | Completed the optimiser rename across all three modules; stopped the setter clobbering the live optimiser object | The module would not import, so training, online learning and the model itself were silently absent |
| `4.2.3` | Rewrote the Wi-Fi/OTA service as a non-blocking state machine | The old reconnect path blocked the main loop for up to 25 s every 2 s. With placeholder credentials that was the default state, so serial commands and ADC sampling were being starved |
| `4.2.4` | Stamped every buffered sample with its provenance; tagged simulated frames in the raw message | Simulated frames were byte-identical to diode readings and were reaching the RNN. A real run could not be told from a fabricated one after the fact |
| `4.2.5` | Fixed checkpoint loading; scaler state now round-trips | The loader rebound its `model` argument to the loaded dict, so loading always failed silently and every restart began from random weights |
| `4.2.6` | Switch timing wired end to end, bounds shared between firmware and backend | Backend sent `SWITCH_PERIOD`, firmware parsed `SWITCH_PERIOD_US`; the dashboard called a method that does not exist, behind a `getattr` default that reported success anyway |
| `4.2.7` | Output failsafe on host loss, plus a backend keepalive and a serial write lock | Targets stayed latched on the rig indefinitely if the host process died |
| `4.2.8` | Dashboard on loopback with optional basic auth; OTA requires a password from build flags | The panel bound to every interface with no authentication and OTA would accept a reflash from anyone on the network |

### Features that silently did nothing (`4.2.9` - `4.3.7`)

| Version | Change | Why it mattered |
|---|---|---|
| `4.2.9` | Learning rate now reaches the optimiser | The setter wrote a `learning_rate` key while torch reads `lr`, so the dashboard reported the new rate and the model trained at the old one |
| `4.3.0` | `online_update` returns a consistent triple on every exit | Two early exits returned four values where callers unpack three, so a routine skip surfaced as an unpacking error |
| `4.3.1` | `OFFLINE` parsed case-insensitively | `OFFLINE=FALSE` fell through the check and started the backend in simulation |
| `4.3.2` | Exact command matches tested before prefix matches in the serial handler | `PINS` and `PING` both begin with `PIN`, so both were pulled into the pin setter and answered with a syntax error. The dashboard pin readback and the keepalive had never worked |
| `4.3.3` | Pin panel works to the real 3.3 V rail and confirms what it applied | The UI accepted up to 2000 V and silently clamped; the confirmation line was built and then thrown away |
| `4.3.4` | Signal pipeline runs for the first time | Every entry point raised on a `normalise`/`normalize` mismatch and on `logging.log` being imported as if it were a logger. Pulse windows were also read as seconds despite being named `_us` |
| `4.3.5` | RNN controller pipeline interface usable | The constructor passed `lr=` to a factory taking `learning_rate=`, so the class raised on every instantiation |
| `4.3.6` | ML toolbox wired into training with held-out scoring | 164 lines had sat unimported. Sweeps now report R², RMSE and MAE against a held-back chronological tail |
| `4.3.7` | Ignore rules tidied | `*.md` and `*.yml` meant no documentation or CI workflow could ever be added |

## 3. Before anything else: hardware validation

Everything above was verified against models of the hardware, not the hardware. These specific changes alter runtime behaviour on the board and need a bench session with a real ESP32, ADS1115 and the actual rig wiring:

1. **Output failsafe** (`4.2.7`). Set non-zero targets, pull the USB cable, confirm all five channels drop to 0 V within ~5 s and that `FAILSAFE outputs zeroed` appears. Reconnect and confirm `FAILSAFE cleared`. Then confirm it does **not** trip during normal passive monitoring, which is what the 2 s keepalive is there to prevent.
2. **Serial dispatch** (`4.3.2`). Send `PING`, `PINS`, `GET PINS`, `READ`, `PIN 3 512`, `TARGETS ...`, `SWITCH_PERIOD_US 250` and confirm each is answered correctly rather than with a syntax error.
3. **Switch timing** (`4.2.6`). Scope GPIO 16 and confirm the period matches what was asked for, at both ends of the range.
4. **Non-blocking Wi-Fi** (`4.2.3`). With real credentials set via build flags, confirm the measurement stream does not stutter during association, and that pulling the access point does not stall the loop.
5. **OTA with a password** (`4.2.8`). Confirm `pio run -t upload` over `espota` works with `--auth` and is refused without it.
6. **ADC sanity**. Feed a known voltage and check the reported value. See the resolution note in section 4.

## 4. Outstanding work

### Blocks deployment

- **No dependency manifest.** There is no `requirements.txt` or `pyproject.toml`. The stack needs Python 3.12 (torch has no 3.14 wheels) and nothing in the repo records that. Pin at least dash, plotly, pandas, numpy, pyserial, torch, scikit-learn, matplotlib.
- **`platform = espressif32` is unpinned in `platformio.ini`.** The firmware uses `ledcSetup`, `ledcAttachPin`, `timerAlarmWrite` and `timerAlarmEnable`, all of which were **removed in arduino-esp32 v3.x**. It currently builds only because this machine has 6.12.0 cached. A clean checkout elsewhere will resolve a newer platform and fail to compile. Pin the version before anyone else builds this.
- **`SERIAL_PORT` is hardcoded to `COM4`** in `python_Backend.py`, as is `upload_port` in `platformio.ini`. Both should read from the environment.
- **No tests and no CI.** `4.3.7` unblocked the workflow directory; nothing has been put in it. The verification scripts written during this pass were deliberately throwaway, but the behaviours they check are exactly what a test suite should cover.

### Known defects still open

- **Auto-control runs inside the plot-rendering callback** (`python_dashboard_script.py`, `update_graph`). Three consequences: the actuation rate cannot beat the 1 s `dcc.Interval` so the 100 ms setting is unreachable; **the control loop stops when the browser tab closes**, leaving the last targets latched; and multiple open tabs race on `LAST_AUTO_TS`. This wants to be a backend thread, not a UI callback. Of everything outstanding, this is the one most likely to cause a bad afternoon.
- **`PLOT_HISTORY` grows without bound** in the dashboard, with a `concat` and a `copy` every second. At 20 Hz that is ~72,000 rows an hour, and it defeats the point of the bounded backend buffer.
- **ADC resolution is thrown away twice.** The firmware prints `MEASURED` via `String(float)` at two decimal places, roughly 10 mV quantisation on a 16-bit part, and `ads.setGain()` is never called so it sits on the ±6.144 V default for a 0–3.3 V signal. Together that is a large amount of the ADS1115's resolution discarded before the data ever reaches the model.
- **Saturation metric counts 0 as saturated** (`python_ML_Metrics.py`), so every idle pin flags saturation at rest and the dashboard total is meaningless.
- **No locking around `model`, `optimiser` and `scaler`**, which are shared between the online update thread, the sweep thread and Dash workers.
- **Sweeps generate far more data than the buffer retains.** A default sweep runs well over a thousand steps while `get_training_data()` is capped at 1000 rows, so most of a sweep is discarded before training sees it.
- **Blocking `delay()` calls remain in the firmware loop**: the heartbeat blocks 50 ms every 2 s, and `blink_error` blocks 280 ms per unrecognised command.
- `*.json` is still ignored repo-wide, which will bite if a JSON config is ever needed.
- `.DS_Store` and the `__pycache__/*.pyc` files are still tracked from before the ignore rules were fixed. They need `git rm --cached` to actually leave the index.

### Decisions that need a person, not a patch

- **Switch timing floor is now 50 µs, not 1 µs.** A 1 µs timer ISR is not serviceable on an ESP32: ISR entry alone costs several microseconds, so the documented 1–20 µs range would have hung the board rather than switched it. If sub-50 µs switching is genuinely required, the answer is the **LEDC hardware peripheral at 50% duty** rather than a timer ISR. That generates the waveform in hardware at no CPU cost and with no jitter, which also matches what the README already claims the design does. It is a contained change to `ChannelController` but it is a design decision, so it was left alone.
- **The model's framing looks questionable.** While wiring up held-out scoring it became clear that with independently varying pin settings, the next diode voltage is close to unpredictable from the current state, and long training runs drive the training R² to 1.0 while held-out R² collapses to around −6. On sweep-shaped data, where a setting is held for several samples, it generalises properly (train 0.72, held-out 0.77). That is worth understanding before any weight is put on the model's proposals, and it is the reason the held-out metrics were added.
- **Training now holds back 20% of the window by default** (`validation_ratio` on `train_model`), and the scaler is fitted on the training rows only. This is the correct practice and it is what makes the metrics honest, but it does mean sweeps train on slightly less data than before. Pass `validation_ratio=0.0` to restore the old behaviour if that matters.
- **The README describes a repository that does not exist.** Every path in it says `Embedded Controller (Automatic)/` or `Embedded Controller (Manual)/`; the only real directory is `Embedded Controller/`, and the entire "Manual" fallback stack it documents in two sections was removed in 4.0.0. It also cites a Temu product link in place of the ESP32 Arduino framework reference, which should go before anyone outside the group reads it.
- **Dead code that is now functional but still uncalled.** `python_Signal_Pipeline.py` and `RNNController` both work as of `4.3.4` and `4.3.5`, but nothing in the running system instantiates them. Either wire them into the live data path or drop them; leaving them working-but-unused is the state that let them rot unnoticed in the first place.

## 5. Running it as it stands

Firmware, from `Embedded Controller/`:

```
pio run -t upload                     # release
pio run -e esp32_debug -t upload      # verbose logging
```

Site credentials go in `platformio.ini` `build_flags`, never in `main.cpp`. OTA stays disabled until a password is supplied:

```
build_flags = -DTESTBED_WIFI_SSID='"YourNetwork"' -DTESTBED_WIFI_PASSWORD='"YourKey"' -DTESTBED_OTA_PASSWORD='"YourOtaPassword"'
```

Dashboard, from `Embedded Controller/Backend/`:

```
python python_dashboard_script.py     # http://127.0.0.1:8050
```

Environment variables it honours:

| Variable | Default | Notes |
|---|---|---|
| `OFFLINE` | unset | `1`, `true` or `yes` starts in simulation. Simulated samples are tagged and withheld from training unless this is set |
| `DASHBOARD_HOST` | `127.0.0.1` | Binding to anything non-loopback without a password is refused outright |
| `DASHBOARD_PORT` | `8050` | |
| `DASHBOARD_USER` | `operator` | |
| `DASHBOARD_PASSWORD` | unset | Setting it enables basic auth on every request |

Note that the backend opens the serial port and starts three threads **at import time**, so importing it for any purpose touches the hardware. That is the main reason nothing here is unit-testable yet and it should be moved behind a factory.

## 6. How this work was verified, and how to re-verify

Every defect was reproduced by running the code before being fixed, and the fix was confirmed the same way rather than by inspection. That mattered: it caught three plausible-looking findings that turned out to be wrong, and it caught two of my own faulty test expectations. Worth keeping up, particularly on this codebase, where several defects had survived precisely because the failure was swallowed by a bare `except`.

A few specifics if you are re-checking any of it:

- Import `python_RNN_Controller` and `python_Signal_Pipeline` directly, but **not** `python_Backend`, which opens `COM4` on import.
- Torch import takes around 30 s on this machine, so give any check a generous timeout.
- For the firmware, the useful technique was to parse the dispatch chain straight out of `main.cpp` and route candidate commands through a model of it. That stays honest as the file changes, unlike a hand-copied list.
- Synthetic data for the model must hold a pin vector for several consecutive samples. Independent per-sample pins make the target genuinely unpredictable and any result meaningless.

### Suggested order of work

1. Pin the PlatformIO platform version, before anyone else builds the firmware.
2. Add `requirements.txt`.
3. Bench-validate the six items in section 3.
4. Move auto-control out of the plot callback and bound `PLOT_HISTORY`.
5. Fix the ADC gain and the `MEASURED` precision, then re-baseline the model.
6. Rewrite the README against the directory layout that actually exists.
