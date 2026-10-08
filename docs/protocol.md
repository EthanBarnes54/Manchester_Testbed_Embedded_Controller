# Serial protocol

The interface between the board and the host: every command the firmware accepts, every
line it sends, and what each means. Protocol version **2**. The firmware's
`PROTOCOL_VERSION` and the backend's must match, and `tests/test_docs.py` checks that this
document lists every command the firmware dispatches.

## Link

- USB serial, 115200 baud, 8N1, no flow control.
- One message per line, ended by `\n`. A `\r` is ignored.
- A command longer than 256 characters is discarded and counted (`rx_overflows`, warning
  `SERIAL_OVERFLOW`); the board keeps serving commands.
- Text is ASCII.

## Framing and integrity

Every line the board sends ends in `*XXXX`: four uppercase hex digits of the CRC-16/CCITT-FALSE
(polynomial 0x1021, initial value 0xFFFF, no reflection, no final XOR) of the text before
the `*`. `123456789` gives `29B1`. Implemented in `include/line_protocol.h` on the board
and with `binascii.crc_hqx(text, 0xFFFF)` on the host.

Commands from the host:

| Command ends in | Board does |
|---|---|
| `*XXXX` that matches | Acts on the text before the `*` |
| `*XXXX` that does not match (hex digits in either case) | Replies `ERROR: Bad checksum!`, counts it (`bad_checksums`, warning `BAD_CHECKSUM`), and does **not** count the line as the host being alive |
| Anything else | Acts on the whole line. This is for typing at a terminal; the backend always frames its commands |

The backend drops a board line whose CRC does not match. Once the board has answered
`VERSION` with this protocol version, it also drops protocol lines (those starting with
`MEASURED`, `PINS`, `ACK`, `ERROR`, `OK`, `FAULT`, `FAILSAFE`, `HEALTH`, `SELFTEST` or
`VERSION`) that carry no CRC. Other unframed text (the debug build's log lines, Wi-Fi
status) is shown but never acted on.

## Commands

Exact commands are matched without regard to case. `TARGETS`, `PIN` and `SWITCH_PERIOD_US`
are prefixes and must be upper case. Any other line gets `ERROR: Unknown command received!`.

"Refused unless ARMED" means the reply is `ERROR: Not armed!` and nothing changes, unless the
board is ARMED. A command that sets an output to zero is always accepted.

| Command | Reply | Notes |
|---|---|---|
| `PING` | `OK` | Keeps the failsafe away, like any valid command |
| `VERSION` | `VERSION firmware=<git describe> protocol=<n> build=<environment> gate_loopback=<0\|1> setpoint_readback=<0\|1>` | |
| `ARM` | `ACK ARM`, or `ERROR: Cannot arm - <reason>!` | Opens the ARMED gate (GPIO 23). Refused while a critical fault is active (`a critical fault is active`) or latched (`a critical fault is latched, send CLEAR FAULTS`) |
| `DISARM` | `ACK DISARM` | Closes the gate, then zeroes every output. Leaves FAULT as FAULT |
| `FAULTS` | `FAULTS mode=<SAFE\|ARMED\|FAULT> active=<list> latched=<list> history=<list> counts=<NAME:n,...> boots=<n> unexpected_resets=<n> last_reset=<reason>` | Lists are comma-separated fault names or `none`. `history`, `boots` and `unexpected_resets` survive power cycles |
| `CLEAR FAULTS` | `ACK CLEAR FAULTS mode=<mode>` | Clears latches whose condition has gone |
| `CLEAR LOG` | `ACK CLEAR LOG` | Clears the persisted history and the unexpected reset count |
| `SELFTEST` | `SELFTEST <PASS\|FAIL> clocks=<ok\|FAIL> switch=<ok\|FAIL> adc=<ok\|FAIL> memory=<ok\|FAIL> gate=<ok\|FAIL\|off> readback=<ok\|FAIL\|off> mode=<mode>` | Also run once at power-on. `off` means the check's hardware is not declared |
| `HEALTH` | `HEALTH mode=<mode> uptime_ms=<n> loop_max_us=<n> loop_peak_us=<n> loop_budget_us=<n> overruns=<n> heap_free=<n> heap_min=<n> stack_free=<n> adc=<ok\|lost> adc_conversions=<n> adc_timeouts=<n> rx_overflows=<n> bad_checksums=<n> gate=<ok\|FAIL\|off> gate_edges=<n> readback=<ok\|FAIL\|off>` | `loop_max_us` is the worst pass since the last `HEALTH`; `loop_peak_us` since boot |
| `READ` | The next `MEASURED` line | Answered by the next conversion to finish, normally within 10 ms |
| `PINS` or `GET PINS` | `PINS squeeze_plate=<d> ion_source=<d> wein_filter=<d> cone_1=<d> cone_2=<d> switch_logic=<0\|1>` | Setpoints as 10-bit duty (0-1023). `switch_logic` is the held level, or 1 while switching |
| `TARGETS v1 v2 v3 v4 v5` | `ACK PIN <n> <d>` for each channel, then `ACK TARGETS` | Volts, 0-3.3, for squeeze_plate, ion_source, wein_filter, cone_1, cone_2. Spaces or commas. Each is clamped to 0-3.3 V and rounded to a 10-bit duty. Refused unless ARMED if any duty would be above zero. Fewer than five values: `ERROR: TARGETS requires five voltages!` |
| `PIN <name\|1-6> <value>` | `ACK PIN <n> <value>` | Names: `squeeze_plate`, `ion_source`, `wein_filter`, `cone_1`, `cone_2`, `switch_logic` (or 1-6). Channels 1-5 take a duty, clamped to 0-1023; channel 6 holds the switch line at 0 or 1 (non-zero). Refused unless ARMED if value > 0. Errors: `ERROR: PIN index out of range!`, `ERROR: Invalid PIN syntax!` |
| `SWITCH_PERIOD_US <n>` | `ACK SWITCH_PERIOD_US <n>` | Time between edges, 1-2,000,000 µs (half a cycle: 1 µs is a 500 kHz square wave). `n <= 0` stops switching and holds the line low: `ACK SWITCH_PERIOD_US 0 (disabled)`, always accepted. Otherwise refused unless ARMED. Errors: `ERROR: Switch time out of bounds!`, `ERROR: Invalid switch time!`, `ERROR: Switch timer unavailable!` |

## Lines the board sends on its own

| Line | When |
|---|---|
| `MEASURED <volts> V seq=<n> t_ms=<ms>` | Every 50 ms. Volts to 5 decimal places. `seq` counts from 1 at boot, so a gap is a lost reading and a drop is a restart. `t_ms` is the board's uptime |
| `FAULT <NAME> <CRITICAL\|WARNING> mode=<mode>` | Once at the onset of each fault |
| `FAILSAFE outputs zeroed, no command received from host` | No valid command for 5 s since the host first spoke. The board is now SAFE |
| `FAILSAFE cleared, host link restored` | The next valid command. The board stays SAFE until `ARM` |
| `ERROR: ADC conversion timed out!` | A conversion did not finish within 40 ms; once per outage, and in answer to `READ` during one |
| `OTA image confirmed by the power-on self-test` / `ERROR: OTA image failed its power-on self-test, rolling back!` | First boot after an over-the-air update |
| `WARNING: ...`, `WiFi ...`, `OTA ...` | Status of the radio, in builds that have one |

## Faults

| Name | Severity | Raised when |
|---|---|---|
| `UNEXPECTED_RESET` | Critical | The last reset was a panic, a watchdog or a brownout |
| `CLOCK_CONFIG` | Critical | APB is not 80 MHz, or REF_TICK not APB / 80 |
| `SWITCH_GENERATOR` | Critical | The switch line's LEDC timer or interrupt timer did not come up |
| `GATE_MISMATCH` | Critical | Gate loopback fitted: the gate output does not follow ARMED and the switch line |
| `SWITCH_FREQUENCY` | Critical | Gate loopback fitted: the edge count over a window is off by more than 0.5% or 2 edges |
| `SETPOINT_MISMATCH` | Critical | Readback fitted: three settled readings in a row more than 0.15 V from the command |
| `HOST_TIMEOUT` | Warning | The failsafe tripped |
| `ADC_LOST` | Warning | The ADS1115 stopped answering |
| `LOOP_OVERRUN` | Warning | A loop pass took longer than 20 ms |
| `LOW_MEMORY` | Warning | Free heap under 32 KiB or loop stack headroom under 1 KiB |
| `SERIAL_OVERFLOW` | Warning | A command line over 256 characters |
| `BAD_CHECKSUM` | Warning | A command whose CRC did not match |

A critical fault disarms the board and leaves it in FAULT, where `ARM` is refused until
`CLEAR FAULTS` once the condition has gone. A warning is reported and counted, nothing more.

## Known quirks

- `TARGETS` reads each value with Arduino's `toFloat()`, so a value that is not a number is
  read as 0 V rather than refused. A sixth value is ignored. `PIN` and `SWITCH_PERIOD_US` read
  their numbers with `toInt()` the same way, so `SWITCH_PERIOD_US abc` stops the switch. All
  of these fail towards zero.
- Unframed commands are accepted so the board can be driven from a terminal. Anything on the
  serial port can therefore command it; the port is the trust boundary.

## Version history

| Version | Change |
|---|---|
| 2 (4.6.7) | The first numbered version, reported by the new `VERSION`. CRC on every line, and `seq=` and `t_ms=` on `MEASURED`. Includes the safety commands (`ARM`, `DISARM`, `FAULTS`, `CLEAR FAULTS`, `CLEAR LOG`, 4.6.4) and `SELFTEST` and `HEALTH` (4.6.5) |
| 1 | Everything before 4.6.7: unframed, with no way to ask the version |
