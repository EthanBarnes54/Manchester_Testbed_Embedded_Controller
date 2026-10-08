# Timing budget

Where time goes on the board, what each deadline is, and how much margin it has. This is
an **analysis**: the figures come from the design, the datasheets and the installed
Arduino core, not from a board. ATP-07 measures the loop on the bench (`HEALTH`
`loop_max_us` and `loop_peak_us`) and its results belong in the table at the end.

## What runs where

| Work | Runs on | Depends on the loop? |
|---|---|---|
| Switch line, 1-1000 µs between edges | LEDC hardware | No: no CPU work per edge |
| Switch line, 1001 µs-2 s | Hardware timer interrupt | No: an interrupt, delayed only by other interrupts and flash writes |
| Setpoint PWM, 5 kHz, 10-bit | LEDC hardware | No |
| Gate loopback edge count | Pulse counter hardware | No; read once per pass |
| ARMED, heartbeat, failsafe, faults, commands, ADC, reporting | `loop()` on core 1 | Yes |
| Wi-Fi and OTA (not in the deployment build) | Core 0 tasks, plus `ArduinoOTA.handle()` from `loop()` | Partly |

So a slow loop pass delays commands, readings and the failsafe, but never moves a switch
edge up to 1 ms.

## Deadlines

| Deadline | Value | Source | Set by |
|---|---|---|---|
| Loop pass | 20 ms | `LOOP_BUDGET_US` | A normal pass is about 1 ms plus `delay(1)`; the longest legitimate one primes the LEDC for two periods (2 ms at 1000 µs). A pass over 20 ms raises `LOOP_OVERRUN` |
| Reading | every 50 ms | `MEASUREMENT_INTERVAL_MS` | 20 Hz |
| ADC conversion | 7.8 ms nominal (±10%) at 128 SPS; polled from 7 ms; lost after 40 ms | ADS1115 datasheet; `ADC_FIRST_POLL_MS`, `ADC_CONVERSION_TIMEOUT_MS` | |
| Readback conversion start | within 10 ms of a diode conversion's start | `READBACK_START_WINDOW_MS` = 50 − 40 | So even a readback that times out ends before the next diode conversion is due |
| Failsafe | 5 s of host silence | `COMMAND_TIMEOUT_MS` | |
| Host keepalive | every 2 s | backend `KEEPALIVE_INTERVAL_SEC` | 2.5 keepalives inside the failsafe timeout |
| Loop watchdog | 5 s | core `CONFIG_ESP_TASK_WDT_TIMEOUT_S` | Resets the chip if `loop()` stops |
| Health checks | every 1 s | `HEALTH_CHECK_INTERVAL_MS` | Heap and stack floors |

## Serial output, the largest cost in a pass

At 115200 baud 8N1 the UART sends 11.52 characters per millisecond. The Arduino core
installs the UART driver with **no transmit buffer by default**, only the 128-byte hardware
FIFO, so a write that does not fit blocks the loop until enough has gone out.

Replies, with the CRC suffix and line ending:

| Line | Characters | Time on the wire |
|---|---|---|
| `OK` | 9 | 0.8 ms |
| `MEASURED` | 49 | 4.3 ms |
| `VERSION` | 112 | 9.7 ms |
| `FAULTS`, nothing ever raised | 121 | 10.5 ms |
| `HEALTH` | 278 | 24.1 ms |
| `FAULTS`, every fault active, latched and counted | 876 | 76.0 ms |

The backend's keepalive sends `PING`, `FAULTS` and `HEALTH` together every 2 s. Their
replies plus a reading come to 457 characters, about 40 ms on the wire. With only the FIFO,
the pass that writes `HEALTH` behind `FAULTS` would block for roughly 22-29 ms: over the
20 ms budget every 2 s, raising `LOOP_OVERRUN` and delaying the next reading. The largest
burst the protocol can produce (every fault listed, `VERSION`, `HEALTH`, two readings) is
1,373 characters.

Since 4.7.1 the firmware gives the UART a 2,048-byte transmit buffer
(`SERIAL_TX_BUFFER_BYTES`), so every reply is copied and the loop moves on; the UART driver
drains it from its interrupt. That holds the worst burst with room to spare, for 2 KiB of
heap out of about 200 KiB free. The output is still limited by the baud rate: the buffer
moves the waiting off the loop, it does not make the link faster.

## Loop pass, estimated

| Step | Estimate | Notes |
|---|---|---|
| Heartbeat | < 5 µs | One `digitalWrite` |
| Serial input | ~10 µs per command | Parsing and dispatch; a CRC over at most 256 characters |
| ADC service | ~0.5 ms when it touches the bus | One I²C transaction at 100 kHz (start, poll or read) of 3-4 bytes; most passes do not touch the bus |
| Reporting | ~50 µs | Copy into the TX buffer (was up to 29 ms, above) |
| Output verification | ~10 µs | Read the pulse counter and one GPIO |
| Supervisor | ~10 µs; ~50 µs once a second | Heap and stack checks once a second |
| `delay(1)` | 1 ms | Yields to other tasks. The core's loop task feeds the loop watchdog after each pass |
| **Typical pass** | **~1.1-1.6 ms** | |
| Longest legitimate pass | ~3.6 ms | A switch period change at 1000 µs primes the LEDC for two periods |

The budget of 20 ms leaves more than ten times the typical pass. The failsafe, the
heartbeat and the reading schedule are all tied to the loop, so the margin is what
protects them.

## Switch edges

| Range | Edge placement | Jitter |
|---|---|---|
| 1-1000 µs (LEDC) | Exact: whole divider on a crystal-derived 1 MHz clock | The crystal's (tens of ppm) plus the GPIO matrix's fixed delay, which shifts both edges equally |
| 1001 µs-2 s (interrupt) | Exact on average | Interrupt latency, typically single microseconds; longer during flash writes (NVS, OTA) and Wi-Fi association |

## Measured on the bench (ATP-07)

| Date | Build | Backend connected | `loop_max_us` | `loop_peak_us` | Overruns | Notes |
|---|---|---|---|---|---|---|
| | | | | | | |
