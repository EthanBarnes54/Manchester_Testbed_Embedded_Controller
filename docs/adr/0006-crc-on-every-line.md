# 0006 Frame every serial line with a CRC-16

Status: accepted, 4.6.7.

## Context

The serial link carries commands that make outputs live and the readings the model trains
on. USB serial is checked by USB itself, but bytes can still be lost or garbled between
the UART and the USB bridge, by a buffer overrun, or by a noisy cable on a UART-only
setup. Before 4.6.7 nothing could tell a corrupted line from a good one: `TARGETS 1` could
arrive as `TARGETS 7`.

## Decision

Every line in both directions ends in `*XXXX`, a CRC-16/CCITT-FALSE of the text. The board
refuses a command whose CRC does not match, and a refused line does not count as the host
being alive. Unframed commands are still accepted, so the board can be driven from a
terminal. A `VERSION` handshake carries a protocol number, and the backend will not arm a
board that speaks another.

## Reasons

- CRC-16/CCITT-FALSE detects every single-bit and every burst error up to 16 bits, which
  covers a byte garbled in transit. It costs four characters per line.
- It is standard: `binascii.crc_hqx` on the host, 20 lines of C on the board, and the check
  value `123456789` → `29B1` pins both.
- A refused command not counting as the host being alive means a link delivering only
  garbage still trips the failsafe.

## Alternatives

- A binary protocol (COBS framing, sequence numbers, acknowledgements). More robust, but
  the board could no longer be driven or read by hand, and every tool would need a codec.
- No integrity check, relying on USB. That leaves the UART side and any future
  UART-only link unprotected.

## Consequences

- Unframed commands remain a way in for anything on the serial port; the port is the trust
  boundary (protocol.md).
- `MEASURED` lines carry a sequence number and board time, so the host counts lost readings
  and notices restarts.
