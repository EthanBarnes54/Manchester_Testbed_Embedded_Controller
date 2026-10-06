#pragma once

#include <stddef.h>
#include <stdint.h>

// Framing for every line on the serial link, in both directions: the text, then '*' and a
// CRC-16 of the text as four upper-case hex digits, for example "PING*5A7C". The CRC is
// CRC-16/CCITT-FALSE (polynomial 0x1021, initial value 0xFFFF; the check value for
// "123456789" is 0x29B1). Unlike NMEA's XOR it catches every burst error up to 16 bits and
// every odd number of flipped bits. Pure logic with no Arduino dependency, so the host
// tests compile this exact file; the backend computes the same CRC with binascii.crc_hqx.

namespace line_protocol {

constexpr size_t CHECKSUM_LENGTH = 5;  // "*XXXX"

inline uint16_t crc16(const char* data, size_t length) {
  uint16_t crc = 0xFFFF;

  for (size_t index = 0; index < length; ++index) {
    crc ^= static_cast<uint16_t>(static_cast<uint8_t>(data[index]) << 8);

    for (int bit = 0; bit < 8; ++bit) {
      crc = (crc & 0x8000) ? static_cast<uint16_t>((crc << 1) ^ 0x1021) : static_cast<uint16_t>(crc << 1);
    }
  }

  return crc;
}

// Writes "*XXXX" and a terminating NUL into suffix, which must hold CHECKSUM_LENGTH + 1 bytes.
inline void format_suffix(uint16_t crc, char* suffix) {
  static const char digits[] = "0123456789ABCDEF";
  suffix[0] = '*';

  for (int nibble = 0; nibble < 4; ++nibble) {
    suffix[1 + nibble] = digits[(crc >> (12 - 4 * nibble)) & 0xF];
  }

  suffix[CHECKSUM_LENGTH] = '\0';
}

enum class Check : uint8_t { Absent, Valid, Invalid };

// Looks for a "*XXXX" suffix. With one, body_length is set to the length of the text before
// it and the result says whether the CRC matches. Without one, the whole line is the body:
// lines typed by hand at a terminal carry no checksum.
inline Check verify(const char* line, size_t length, size_t* body_length) {
  *body_length = length;

  if (length < CHECKSUM_LENGTH || line[length - CHECKSUM_LENGTH] != '*') {
    return Check::Absent;
  }

  uint16_t received = 0;

  for (size_t index = length - (CHECKSUM_LENGTH - 1); index < length; ++index) {
    const char digit = line[index];
    uint8_t value = 0;

    if (digit >= '0' && digit <= '9') {
      value = static_cast<uint8_t>(digit - '0');
    } else if (digit >= 'A' && digit <= 'F') {
      value = static_cast<uint8_t>(digit - 'A' + 10);
    } else if (digit >= 'a' && digit <= 'f') {
      value = static_cast<uint8_t>(digit - 'a' + 10);
    } else {
      return Check::Absent;
    }

    received = static_cast<uint16_t>((received << 4) | value);
  }

  *body_length = length - CHECKSUM_LENGTH;
  return crc16(line, *body_length) == received ? Check::Valid : Check::Invalid;
}

}  // namespace line_protocol
