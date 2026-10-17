#!/usr/bin/env python3
"""Renders the application icon (film frame + neural network 'play' button) with no dependencies.

Writes icon/icon-256.png (embedded in the executable as the window icon) and icon/icon.ico.
"""
import math, os, struct, zlib

BRACKET = (70, 220, 235)
WHITE = (235, 250, 255)


def sd_rrect(x, y, cx, cy, hw, hh, r):
    qx, qy = abs(x - cx) - hw + r, abs(y - cy) - hh + r
    return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - r


def sd_seg(x, y, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x - ax - t * dx, y - ay - t * dy)


def inside_tri(x, y, p):
    s = []
    for i in range(3):
        a, b = p[i], p[(i + 1) % 3]
        s.append((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]))
    return all(v >= 0 for v in s) or all(v <= 0 for v in s)


def render(size):
    k = size / 256.0
    # shapes in 256-space: (kind, params, colour, alpha)
    A, B, C = (88, 92), (88, 164), (152, 128)
    M = (109, 128)
    N1, N2, N3 = (178, 98), (182, 158), (174, 128)
    edges = [(A, B), (B, C), (C, A), (M, A), (M, B), (M, C), (C, N3), (N3, N1), (N3, N2), (C, N1), (C, N2)]
    nodes = [(A, 6.5), (B, 6.5), (C, 7.5), (M, 5), (N1, 5.5), (N2, 5.5), (N3, 5)]
    lo, hi, L = 58, 198, 30
    brackets = []
    for (cx, cy, sx, sy) in [(lo, lo, 1, 1), (hi, lo, -1, 1), (lo, hi, 1, -1), (hi, hi, -1, -1)]:
        brackets.append(((cx, cy + sy * L), (cx, cy)))
        brackets.append(((cx, cy), (cx + sx * L, cy)))
    holes = [(x, 56 + 26 * i) for x in (30, 226) for i in range(7)]

    pix = bytearray(size * size * 4)
    for py in range(size):
        for px in range(size):
            x, y = (px + 0.5) / k, (py + 0.5) / k
            d = sd_rrect(x, y, 128, 128, 120, 120, 54) * k
            a_bg = max(0.0, min(1.0, 0.5 - d))
            if a_bg <= 0:
                continue
            t = (x + y) / 512.0
            r = 12 + (6 - 12) * t + 10 * (1 - t) ** 3
            g = 20 + (70 - 20) * t
            b = 40 + (96 - 40) * t
            col = [r, g, b]

            def over(c, a):
                if a <= 0:
                    return
                for i in range(3):
                    col[i] = col[i] * (1 - a) + c[i] * a

            for (hx, hy) in holes:
                over((255, 255, 255), 0.16 * max(0.0, min(1.0, 0.5 - sd_rrect(x, y, hx, hy, 5, 8, 2.5) * k)))
            if inside_tri(x, y, (A, B, C)):
                over(BRACKET, 0.16)
            for (p, q) in brackets:
                over(BRACKET, max(0.0, min(1.0, 0.5 - (sd_seg(x, y, *p, *q) - 3.5) * k)))
            for (p, q) in edges:
                over(BRACKET, 0.9 * max(0.0, min(1.0, 0.5 - (sd_seg(x, y, *p, *q) - 1.8) * k)))
            for (c, rad) in nodes:
                dd = math.hypot(x - c[0], y - c[1])
                over(BRACKET, 0.25 * max(0.0, min(1.0, 0.5 - (dd - rad - 4) * k)))
                over(WHITE, max(0.0, min(1.0, 0.5 - (dd - rad) * k)))
            o = (py * size + px) * 4
            pix[o:o + 4] = bytes([int(col[0]), int(col[1]), int(col[2]), int(a_bg * 255)])
    return bytes(pix)


def png(size, rgba):
    raw = b"".join(b"\x00" + rgba[y * size * 4:(y + 1) * size * 4] for y in range(size))

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def main():
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "icon")
    os.makedirs(out, exist_ok=True)
    sizes = [256, 64, 48, 32, 16]
    pngs = {s: png(s, render(s)) for s in sizes}
    open(os.path.join(out, "icon-256.png"), "wb").write(pngs[256])
    head = struct.pack("<HHH", 0, 1, len(sizes))
    entries, data, off = b"", b"", 6 + 16 * len(sizes)
    for s in sizes:
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(pngs[s]), off + len(data))
        data += pngs[s]
    open(os.path.join(out, "icon.ico"), "wb").write(head + entries + data)


if __name__ == "__main__":
    main()
