# Requirements

What the testbed controller must do, each with an ID and how it is verified. The IDs are
stable: a requirement that is dropped keeps its ID retired rather than reused.

Verification methods:

- **Test**: automated, runs on every push. Every test that verifies a requirement carries
  `@pytest.mark.req("<ID>")`, and `tests/test_traceability.py` fails if a Test requirement
  has no test or a test names an unknown ID. `python tools/traceability.py` prints the full
  matrix.
- **Bench ATP-NN**: at the rig, by the numbered step of
  [acceptance-test-procedure.md](acceptance-test-procedure.md). Steps marked automated
  there are run by the hardware-in-the-loop suite in `Backend/hil/`.
- **Inspection**: by reading the design, configuration or CI definition.
- **Analysis**: by calculation, recorded in the document named.

A requirement passes only when every listed method passes. Host tests prove the logic;
only the bench proves the hardware does what the logic asks.

## Switch output (SW)

| ID | Requirement | Verification |
|----|-------------|--------------|
| SW-01 | The switch line shall toggle at any whole-number time between edges from 1 µs to 2 s. | Test; Bench ATP-02 |
| SW-02 | A switch time the hardware cannot produce exactly shall be refused, never rounded or clamped. | Test |
| SW-03 | The switch line and the ARMED line shall be driven low before any other start-up work, and shall stay low whenever the board is not armed. | Test; Bench ATP-01 |
| SW-04 | Changing the switch time, including between the hardware-timed and interrupt-timed ranges, shall never produce a pulse longer or shorter than either the old or the new setting. | Test; Bench ATP-03 |
| SW-05 | The firmware, backend and dashboard shall share one set of switch time bounds. | Test |

## Safety (SAF)

| ID | Requirement | Verification |
|----|-------------|--------------|
| SAF-01 | The board shall start in SAFE with every output de-energised. | Test; Bench ATP-01 |
| SAF-02 | A command that would energise an output shall be refused unless the board is ARMED, and only an explicit ARM command shall arm it. Commands that set outputs to zero shall always be accepted. | Test; Bench ATP-05 |
| SAF-03 | If no valid command arrives for 5 s, the board shall close the ARMED gate, then zero every output and disarm. The host returning shall not re-arm it. | Test; Bench ATP-04 |
| SAF-04 | A critical fault shall latch the board in FAULT, refuse ARM until the operator clears it, and be recorded in a fault history kept across resets. | Test; Bench ATP-09 |
| SAF-05 | After a reset caused by a panic, a watchdog or a brownout, the board shall come up in FAULT. | Test; Bench ATP-13 |
| SAF-06 | A main loop that stops running shall reset the chip, and every loop pass shall toggle a heartbeat line for an external watchdog. | Test; Bench ATP-06 |
| SAF-07 | A firmware update over the air shall only be accepted while the board is disarmed. | Test |

## Built-in test (BIT)

| ID | Requirement | Verification |
|----|-------------|--------------|
| BIT-01 | At power-on the board shall check its clocks, the switch generator, the ADC, free memory and any fitted verification hardware, and shall not arm if a check fails. | Test; Bench ATP-07 |
| BIT-02 | The board shall continuously monitor loop time, free heap, free stack and the ADC, and raise a fault when one leaves its limit. | Test; Bench ATP-07 |
| BIT-03 | The operator shall be able to run the self-test and read a health report on demand. | Test; Bench ATP-07 |

## Output verification (OUT)

| ID | Requirement | Verification |
|----|-------------|--------------|
| OUT-01 | When the gate loopback is fitted, the board shall compare the gate's actual state and switch edge count with what it commanded, and raise a critical fault on a sustained mismatch. | Test; Bench ATP-09 |
| OUT-02 | When setpoint readback is fitted, the board shall compare each read-back setpoint with its command once settled, and raise a critical fault on a sustained mismatch. | Test; Bench ATP-09 |
| OUT-03 | Verification hardware shall be assumed absent unless a build declares it, and the deployment build shall declare it. | Test |

## Measurement (MEAS)

| ID | Requirement | Verification |
|----|-------------|--------------|
| MEAS-01 | The diode voltage shall be reported at 20 Hz without the main loop ever waiting on a conversion. | Test; Bench ATP-08 |
| MEAS-02 | Every reading shall carry a sequence number and a board timestamp, and the host shall count missing readings and board restarts. | Test; Bench ATP-08 |
| MEAS-03 | The ADC range and the reported precision shall keep the converter's full resolution over the signal range. | Test |

## Serial link (LINK)

| ID | Requirement | Verification |
|----|-------------|--------------|
| LINK-01 | Every line in both directions shall carry a CRC-16/CCITT-FALSE. A line that fails its check shall be refused and counted, and a refused command shall not count as the host being alive. | Test; Bench ATP-10 |
| LINK-02 | The host shall not arm a board whose protocol version differs from its own. | Test; Bench ATP-10 |
| LINK-03 | The host shall account for every command it sends as answered, refused or unanswered. | Test |
| LINK-04 | The host shall poll the board well inside the failsafe timeout, keeping its view of the board's faults and health current. | Test |
| LINK-05 | An over-long or malformed input line shall be discarded and counted, and the board shall keep serving commands. | Bench ATP-10 |

## Auto control (AUTO)

| ID | Requirement | Verification |
|----|-------------|--------------|
| AUTO-01 | Auto control shall only drive the rig with a model whose held-out R² is at least 0.5. | Test |
| AUTO-02 | Every proposal shall be held to 0–3.3 V per channel and to at most 0.25 V of movement per decision, and a malformed proposal shall send nothing. | Test |
| AUTO-03 | While recent inputs sit more than four training standard deviations from the training data, auto control shall hold the targets already on the rig. | Test |
| AUTO-04 | Auto control shall act only while enabled, armed and fed fresh readings with no sweep running, and a manual edit shall take over from it. | Test |

## Data and model (DATA)

| ID | Requirement | Verification |
|----|-------------|--------------|
| DATA-01 | Simulated readings shall never be used for training on a hardware run. | Test |
| DATA-02 | A sweep shall train on everything it recorded, and an aborted sweep shall not train. | Test |
| DATA-03 | Training shall report a held-out score, and that score shall travel with the saved model. | Test |
| DATA-04 | The shared model shall never be used by two threads at once. | Test |

## Operator interface (UI)

| ID | Requirement | Verification |
|----|-------------|--------------|
| UI-01 | The dashboard shall change the rig only in response to an operator input, never from a display refresh. | Test |
| UI-02 | The dashboard shall show the board's mode, faults, health and link state as the board reports them. | Test |

## Security and configuration (SEC)

| ID | Requirement | Verification |
|----|-------------|--------------|
| SEC-01 | An observer shall be able to watch the rig and make it safe, but never arm it or change an output. | Test |
| SEC-02 | The dashboard shall refuse to serve beyond the local machine without a password and TLS. | Test |
| SEC-03 | Every operator action that changes the rig shall be recorded in an audit log, and a failing audit log shall not stop the rig. | Test |
| SEC-04 | Every saved dataset and model shall record what produced it and a hash of its contents. | Test |
| SEC-05 | The deployment build shall contain no radio. | Test; Inspection |
| SEC-06 | An updated firmware image shall be kept only if its power-on self-test passes, and rolled back otherwise. | Test; Bench ATP-11 |
| SEC-07 | Each build shall produce a software bill of materials listing every dependency at its pinned version. | Test |
| SEC-08 | Every platform, library and tool version shall be pinned exactly, and nothing in the build shall depend on one machine. | Test |

## Build quality (QA)

| ID | Requirement | Verification |
|----|-------------|--------------|
| QA-01 | The project's own firmware sources shall build with every compiler warning treated as an error. | Test |
| QA-02 | The backend shall pass static checks, and its line coverage shall not fall below the recorded floor. | Test; Inspection |
| QA-03 | Each firmware image shall use at most 80% of its application partition, leaving room for updates. | Test; Inspection |
| QA-04 | The tests that guard safety logic shall be shown to fail when that logic is broken. | Test |
| QA-05 | The board shall run for the soak period with no fault, no missed reading and no loop overrun. | Bench ATP-12 |
