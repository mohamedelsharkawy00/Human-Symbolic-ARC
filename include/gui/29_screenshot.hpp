#pragma once
// A picture of the window, written where anything can read it. PNG and not a
// raw dump because the point is that someone else opens it — the agent driving
// the command port cannot see the screen, and a format only this program knows
// would not help it.
//
// NO COMPRESSION, and no library for it either. Deflate allows a "stored"
// block: a length and the bytes themselves. That is a legal PNG every reader
// accepts, it costs about forty lines, and the only thing it spends is disk on
// a file nobody keeps. A compressor here would be a dependency bought with the
// one currency this project does not spend.
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace arc {
using namespace std;

// The two checksums PNG asks for, said once each.
inline uint32_t png_crc(const uint8_t* bytes, size_t count, uint32_t crc = 0xFFFFFFFFu) {
    static uint32_t table[256];
    static bool made = false;
    if (!made) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int bit = 0; bit < 8; ++bit)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : (c >> 1);
            table[i] = c;
        }
        made = true;
    }
    for (size_t i = 0; i < count; ++i)
        crc = table[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

inline uint32_t png_adler(const vector<uint8_t>& bytes) {
    uint32_t low = 1, high = 0;
    for (uint8_t byte : bytes) {
        low = (low + byte) % 65521;
        high = (high + low) % 65521;
    }
    return (high << 16) | low;
}

// One chunk: length, name, payload, checksum over name and payload.
inline void png_chunk(ofstream& file, const char name[4],
                      const vector<uint8_t>& payload) {
    auto big_endian = [&](uint32_t value) {
        const uint8_t four[4] = { (uint8_t)(value >> 24), (uint8_t)(value >> 16),
                                  (uint8_t)(value >> 8), (uint8_t)value };
        file.write((const char*)four, 4);
    };
    big_endian((uint32_t)payload.size());
    uint32_t crc = png_crc((const uint8_t*)name, 4);
    if (!payload.empty()) crc = png_crc(payload.data(), payload.size(), crc);
    file.write(name, 4);
    if (!payload.empty()) file.write((const char*)payload.data(), payload.size());
    big_endian(crc ^ 0xFFFFFFFFu);
}

// `pixels` is one row after another, four bytes a pixel, BLUE FIRST — the order
// the swapchain hands them over. Rows may be padded, so the stride is asked for
// rather than assumed.
inline bool write_png(const string& path, int width, int height,
                      const uint8_t* pixels, int row_stride) {
    if (width <= 0 || height <= 0 || !pixels) return false;
    ofstream file(path, ios::binary);
    if (!file) return false;
    const uint8_t signature[8] = { 137, 'P', 'N', 'G', '\r', '\n', 26, '\n' };
    file.write((const char*)signature, 8);

    vector<uint8_t> header;
    auto put32 = [&](vector<uint8_t>& into, uint32_t value) {
        into.push_back((uint8_t)(value >> 24)); into.push_back((uint8_t)(value >> 16));
        into.push_back((uint8_t)(value >> 8));  into.push_back((uint8_t)value);
    };
    put32(header, (uint32_t)width);
    put32(header, (uint32_t)height);
    // 8 bits a channel, three channels, no interlace.
    header.insert(header.end(), { 8, 2, 0, 0, 0 });
    png_chunk(file, "IHDR", header);

    // Each row is a filter byte (0, meaning none) then the pixels as red,
    // green, blue — the alpha the window drew with is not a picture of it.
    vector<uint8_t> raw;
    raw.reserve((size_t)height * (1 + 3 * (size_t)width));
    for (int row = 0; row < height; ++row) {
        raw.push_back(0);
        const uint8_t* line = pixels + (size_t)row * row_stride;
        for (int col = 0; col < width; ++col) {
            raw.push_back(line[4 * col + 2]);
            raw.push_back(line[4 * col + 1]);
            raw.push_back(line[4 * col + 0]);
        }
    }

    // A zlib stream of stored blocks: two bytes of preamble, then at most
    // 65535 bytes at a time, then the checksum of everything said.
    vector<uint8_t> zlib{ 0x78, 0x01 };
    for (size_t at = 0; at < raw.size(); at += 65535) {
        const uint16_t block = (uint16_t)min<size_t>(65535, raw.size() - at);
        const bool last = (at + block >= raw.size());
        zlib.push_back(last ? 1 : 0);
        zlib.push_back((uint8_t)(block & 0xFF));
        zlib.push_back((uint8_t)(block >> 8));
        zlib.push_back((uint8_t)(~block & 0xFF));
        zlib.push_back((uint8_t)(~block >> 8));
        zlib.insert(zlib.end(), raw.begin() + at, raw.begin() + at + block);
    }
    put32(zlib, png_adler(raw));
    png_chunk(file, "IDAT", zlib);
    png_chunk(file, "IEND", {});
    return (bool)file;
}

}  // namespace arc
