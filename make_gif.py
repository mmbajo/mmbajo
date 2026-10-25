#!/usr/bin/env python3
"""CatPole-v1: a cat balancing on a cart-pole, as a seamlessly looping pixel-art GIF.

The scene is drawn at 200x80 from one fixed palette and scaled up 8x with
nearest-neighbour. Shown at 800px wide it stays crisp on both regular and HiDPI
screens. Every moving part is periodic in the loop length, so the last frame
flows straight back into the first.

    python make_gif.py                      # writes assets/catpole.gif
    python make_gif.py --sheet sheet.png    # contact sheet of a few frames
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 200, 80
SCALE = 8
FPS = 25
N = 150  # frames per loop (6 s)

PALETTE = [
    # night sky, zenith to horizon
    ("sky0", "0a0a1e"), ("sky1", "0f0f2a"), ("sky2", "151538"), ("sky3", "1c1b47"),
    ("sky4", "252257"), ("sky5", "302a66"), ("sky6", "3d3171"), ("sky7", "4c3a7a"), ("sky8", "5e4283"),
    # moon and stars
    ("moon_hi", "fff8dc"), ("moon", "f2e6bd"), ("moon_sh", "d9c896"), ("moon_cr", "cdbb8a"),
    ("star_hi", "ffffff"), ("star", "dfe2ff"), ("star_mid", "a3a8e6"), ("star_dim", "6c70b8"),
    # city
    ("bld_far", "2a2457"), ("bld_far_hi", "3d3577"), ("bld", "17142f"), ("bld_hi", "2e2963"),
    ("win1", "ffd27a"), ("win2", "ffb061"), ("win3", "8fd3ff"), ("win_far", "7a6aa0"),
    ("beacon", "ff5a6a"),
    # ground and track
    ("grass_hi", "3a7a5e"), ("grass", "2a5e4e"), ("ground", "1d4140"), ("ground_dk", "142e33"),
    ("ground_dkr", "0e2128"), ("flower1", "b48cf0"), ("flower2", "f0e6ff"),
    ("rail_hi", "c3c9e8"), ("rail", "8088b4"), ("rail_sh", "4b5080"),
    ("sleeper", "5a4034"), ("sleeper_dk", "3d2a24"), ("flag", "ff6b7a"), ("post", "8a8fb0"),
    # cardboard-box cart
    ("box_hi", "f0c587"), ("box", "d9a05e"), ("box_sh", "b07a42"), ("box_dk", "85592f"),
    ("box_ol", "3b2418"), ("tape", "f6dca8"),
    ("tire", "241f38"), ("tire_hi", "3c3658"), ("hub", "c8c3e3"),
    ("bolt", "d8dcf0"), ("bolt_sh", "8a8fb0"),
    # scratching-post pole and perch
    ("sisal_hi", "f0dcae"), ("sisal", "d9bd84"), ("sisal_sh", "b0925a"),
    ("carpet_hi", "a98ce0"), ("carpet", "8467c2"), ("carpet_sh", "5f4699"),
    # cat
    ("cat_ol", "2a1418"), ("cat_dk", "b45a24"), ("cat_md", "d9772c"), ("cat", "f0913a"),
    ("cat_lt", "ffb65e"), ("cat_wh", "fff3e2"), ("cat_pink", "ff9aa8"),
    ("eye", "b9f25e"), ("pupil", "170f1f"), ("shine", "ffffff"), ("whisker", "cfc6e8"),
    # rewards, fireflies
    ("spark_hi", "fff3a0"), ("spark", "ffd84a"), ("spark_dim", "a8883a"),
    ("ff_core", "fdffc2"), ("ff", "d6f06a"), ("ff_dim", "7d9a3e"),
]
C = {name: i for i, (name, _) in enumerate(PALETTE)}
SKY = [C[f"sky{k}"] for k in range(9)]

# Front-facing orange tabby, left half; the right half is mirrored from it.
CAT_HALF = [
    "..o.......",
    ".oLo......",
    ".oPLo.....",
    ".oPPLooooo",
    ".oOLLLLDLD",
    ".oOOOOODOD",
    ".oOOOOOOOO",
    ".oOOOOOOOO",
    "oDOOOOOOOO",
    "oOOOOOOOOO",
    "oDOOOOOWWP",
    "oOOOOWWWWo",
    ".oOOWWWWoW",
    "..ooOWWWWW",
    "..oOOOWWWW",
    ".oOOOOWWWW",
    ".oDOOOWWWW",
    ".oOOOoWWWW",
    ".oOOOoWWWo",
    "..oooooooo",
]
CAT = [row + row[-2::-1] for row in CAT_HALF]
CAT_W, CAT_H = len(CAT[0]), len(CAT)
# right ear folded down a pixel, for the ear twitch
CAT_TWITCH = [row[:13] + (CAT[r - 1][13:] if r else "." * 6) if r < 3 else row for r, row in enumerate(CAT)]
CAT_MAP = {"o": "cat_ol", "O": "cat", "L": "cat_lt", "D": "cat_dk", "W": "cat_wh", "P": "cat_pink"}
EYE_X, EYE_Y = (4, 12), 7
EYE_OPEN = [["pupil", "pupil", "shine"], ["pupil", "pupil", "pupil"], ["eye", "pupil", "pupil"]]
EYE_HALF = [["cat_md", "cat_md", "cat_md"], ["pupil", "pupil", "pupil"], ["eye", "pupil", "pupil"]]
EYE_SHUT = [["cat", "cat", "cat"], ["cat", "cat", "cat"], ["cat_ol", "cat_ol", "cat_ol"]]

BOX = (
    ["o" * 24, "o" + "h" * 9 + "tttt" + "h" * 9 + "o"]
    + ["os" + "b" * 8 + "tttt" + "b" * 9 + "o"] * 2
    + ["os" + "b" * 21 + "o"] * 5
    + ["od" + "s" * 21 + "o", "o" * 24]
)
BOX_MAP = {"o": "box_ol", "h": "box_hi", "b": "box", "s": "box_sh", "d": "box_dk", "t": "tape"}
PAW = [".d.d.", "d...d", ".ddd.", ".ddd."]

WHEEL = ["..ooo..", ".oTTTo.", "oTtttTo", "oTtHtTo", "oTtttTo", ".oTTTo.", "..ooo.."]
WHEEL_MAP = {"o": "box_ol", "T": "tire", "t": "tire_hi", "H": "hub"}

PERCH = ["o" + "h" * 13 + "o", "o" + "c" * 13 + "o", ".o" + "s" * 11 + "o.", "..ooooooooooo.."]
PERCH_MAP = {"o": "box_ol", "h": "carpet_hi", "c": "carpet", "s": "carpet_sh"}

GROUND_Y = 69
RAIL_Y = 67
WHEEL_CY = 63
BOX_TOP = 51
HINGE_Y = 55
POLE_LEN = 27
MOON = (172, 16, 8)
TRACK = (12, 187)
LIMITS = (24, 176)  # CartPole terminates past x = +-2.4

# Cart motion: a few harmonics of the loop. The pole angle is the forced response
# of a linearised inverted pendulum to that motion, so the top lags the cart the
# way a balanced pole does.
HARMONICS = [(1, 7.4, 0.0), (2, 14.8, 3.07), (3, 3.5, 2.55), (5, 1.8, 2.5)]  # (cycles/loop, px, phase)
GRAVITY = 208.0  # px/s^2, picked for a +-15 degree sway

BLINKS = [37, 44, 113]
BLINK_SHAPE = [EYE_HALF, EYE_SHUT, EYE_SHUT, EYE_HALF]
TWITCH = (80, 3)  # (start frame, length)
REWARDS = [10, 40, 70, 100, 130]
REWARD_LIFE = 24
SHOOTING_STAR = (96, 10)
FLICKER = [(3, 20, 95), (11, 60, 140), (17, 0, 40), (24, 105, 30)]  # (window, on, off)
TOWER_X = 143


def put(c, x, y, col):
    if 0 <= x < W and 0 <= y < H:
        c[y, x] = C[col]


def rect(c, x0, y0, x1, y1, col):
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
    if x0 <= x1 and y0 <= y1:
        c[y0:y1 + 1, x0:x1 + 1] = C[col]


def blit(c, rows, x, y, cmap):
    for j, row in enumerate(rows):
        for i, ch in enumerate(row):
            if ch != ".":
                put(c, x + i, y + j, cmap[ch])


def stroke(c, pts, radius, fill, outline="box_ol"):
    """Draw an outlined polyline; radius and fill may depend on the segment index."""
    n = len(pts) - 1
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    for py in range(int(min(ys)) - 2, int(max(ys)) + 3):
        for px in range(int(min(xs)) - 2, int(max(xs)) + 3):
            best, seg = 99.0, 0
            for k in range(n):
                (ax, ay), (bx, by) = pts[k], pts[k + 1]
                vx, vy = bx - ax, by - ay
                u = ((px + 0.5 - ax) * vx + (py + 0.5 - ay) * vy) / (vx * vx + vy * vy)
                u = min(1.0, max(0.0, u))
                d = math.hypot(px + 0.5 - ax - u * vx, py + 0.5 - ay - u * vy)
                if d < best:
                    best, seg = d, k
            r = radius(seg)
            if best < r:
                put(c, px, py, fill(seg))
            elif best < r + 0.75:
                put(c, px, py, outline)


def wave(i, cycles, phase=0.0):
    return math.sin(2 * math.pi * cycles * i / N + phase)


def age(i, start):
    return (i - start) % N


# ---------------------------------------------------------------- background

BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) + 0.5) / 16


def draw_sky(c):
    yy, xx = np.mgrid[0:H, 0:W]
    level = 8.0 * np.clip(yy / 64.0, 0, 1) ** 1.3
    mx, my, mr = MOON
    d = np.hypot(xx - mx, yy - my)
    level += 1.8 * np.clip(1 - (d - mr) / 11.0, 0, 1)
    lo = np.floor(level)
    frac = np.clip((level - lo - 0.5) * 2.5 + 0.5, 0, 1)
    idx = np.clip(lo + (BAYER[yy % 4, xx % 4] < frac), 0, len(SKY) - 1).astype(int)
    c[:] = np.array(SKY)[idx]


def draw_moon(c):
    mx, my, r = MOON
    craters = [(-3, -1, 1.7), (2, 3, 1.3), (-1, 4, 1.0), (3, -3, 0.9), (-4, 3, 0.8)]
    for y in range(my - r - 1, my + r + 2):
        for x in range(mx - r - 1, mx + r + 2):
            dx, dy = x - mx, y - my
            if dx * dx + dy * dy > (r + 0.5) ** 2:
                continue
            col = "moon"
            if math.hypot(dx - 2, dy + 2) < 3.5:
                col = "moon_hi"
            if any(math.hypot(dx - cx, dy - cy) < cr for cx, cy, cr in craters):
                col = "moon_cr"
            if math.hypot(dx, dy) > r - 1.3 and dx - dy < -1:
                col = "moon_sh"
            put(c, x, y, col)


def make_stars():
    rng = random.Random(11)
    stars = []
    mx, my, mr = MOON
    while len(stars) < 42:
        x, y = rng.randrange(2, W - 2), rng.randrange(1, 50)
        if math.hypot(x - mx, y - my) < mr + 7:
            continue
        if any(abs(x - sx) + abs(y - sy) < 6 for sx, sy, *_ in stars):
            continue
        twinkle = rng.random() < 0.3
        stars.append((x, y, twinkle, rng.choice([1, 2, 3, 5]), rng.uniform(0, 2 * math.pi),
                      rng.choice(["star_dim", "star_dim", "star_mid"])))
    return stars


def make_city():
    rng = random.Random(5)
    far, near, windows = [], [], []
    x = -3
    while x < W:
        w, h = rng.randint(6, 13), rng.randint(5, 12)
        far.append((x, GROUND_Y - 6 - h, x + w - 1))
        x += w + rng.randint(-3, 2)
    x = -2
    while x < W:
        w, h = rng.randint(5, 11), rng.randint(2, 8)
        top = GROUND_Y - 2 - h
        near.append((x, top, x + w - 1))
        for wy in range(top + 2, GROUND_Y - 1, 2):
            for wx in range(x + 1, x + w - 1, 2):
                if rng.random() < 0.28:
                    windows.append((wx, wy, rng.choice(["win1"] * 6 + ["win2"] * 3 + ["win3"])))
        x += w + rng.randint(1, 4)
    return far, near, windows


STARS = make_stars()
CITY_FAR, CITY_NEAR, WINDOWS = make_city()
TOWER_TOP = min(top for x0, top, x1 in CITY_FAR if x0 <= TOWER_X <= x1) - 9
FLOWERS = [(6, 75, "flower1"), (8, 76, "flower2"), (46, 73, "flower2"), (71, 77, "flower1"),
           (73, 76, "flower1"), (117, 75, "flower2"), (148, 77, "flower1"), (151, 76, "flower2"),
           (193, 74, "flower1")]


def draw_city(c):
    for x0, top, x1 in CITY_FAR:
        rect(c, x0, top, x1, GROUND_Y, "bld_far")
    for x0, top, x1 in CITY_FAR:
        rect(c, x0 + 1, top, x1, top, "bld_far_hi")
        rect(c, x1, top, x1, GROUND_Y, "bld_far_hi")
    rect(c, TOWER_X, TOWER_TOP, TOWER_X, GROUND_Y, "bld_far")
    put(c, TOWER_X - 1, TOWER_TOP + 4, "bld_far")
    put(c, TOWER_X + 1, TOWER_TOP + 4, "bld_far")
    rng = random.Random(9)
    for x0, top, x1 in CITY_FAR:
        for wy in range(top + 2, GROUND_Y - 6, 3):
            for wx in range(x0 + 2, x1 - 1, 3):
                if rng.random() < 0.12:
                    put(c, wx, wy, "win_far")
    for x0, top, x1 in CITY_NEAR:
        rect(c, x0, top, x1, GROUND_Y, "bld")
        rect(c, x0 + 1, top, x1, top, "bld_hi")
        rect(c, x1, top, x1, GROUND_Y, "bld_hi")
    for x, y, col in WINDOWS:
        put(c, x, y, col)


def draw_ground(c):
    rng = random.Random(3)
    rect(c, 0, GROUND_Y, W - 1, H - 1, "ground")
    rect(c, 0, GROUND_Y, W - 1, GROUND_Y + 1, "grass")
    for x in range(W):
        r = rng.random()
        if r < 0.35:
            put(c, x, GROUND_Y - 1, "grass")
            put(c, x, GROUND_Y, "grass_hi")
            if r < 0.1:
                put(c, x, GROUND_Y - 2, "grass_hi")
    for _ in range(70):
        put(c, rng.randrange(W), rng.randrange(GROUND_Y + 3, H), "ground_dk")
    for x, y, col in FLOWERS:
        put(c, x, y + 1, "grass")
        put(c, x, y, col)


def draw_track(c):
    x0, x1 = TRACK
    for x in range(x0 + 2, x1 - 1, 6):
        rect(c, x, RAIL_Y + 1, x + 2, RAIL_Y + 2, "sleeper")
        rect(c, x, RAIL_Y + 2, x + 2, RAIL_Y + 2, "sleeper_dk")
    rect(c, x0, RAIL_Y, x1, RAIL_Y, "rail_hi")
    rect(c, x0, RAIL_Y + 1, x1, RAIL_Y + 1, "rail")
    for x in (x0, x1):
        rect(c, x - 1, RAIL_Y - 2, x + 1, RAIL_Y + 1, "rail_sh")
    for x in LIMITS:
        rect(c, x, RAIL_Y - 6, x, RAIL_Y - 1, "post")
        rect(c, x + 1, RAIL_Y - 6, x + 3, RAIL_Y - 5, "flag")
        put(c, x + 1, RAIL_Y - 4, "flag")


def build_background():
    c = np.zeros((H, W), np.uint8)
    draw_sky(c)
    draw_moon(c)
    for x, y, twinkle, *_, col in STARS:
        if not twinkle:
            put(c, x, y, col)
    draw_city(c)
    draw_ground(c)
    draw_track(c)
    return c


BACKGROUND = build_background()


# ---------------------------------------------------------------- per-frame details

def draw_stars(c, i):
    for x, y, twinkle, cycles, phase, _ in STARS:
        if not twinkle:
            continue
        v = wave(i, cycles, phase)
        if v > 0.8:
            put(c, x, y, "star_hi")
            for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                put(c, x + ox, y + oy, "star_dim")
        else:
            put(c, x, y, "star" if v > 0.3 else "star_mid" if v > -0.4 else "star_dim")


def draw_shooting_star(c, i):
    start, frames = SHOOTING_STAR
    a = age(i, start)
    if a >= frames:
        return
    vx, vy = -3.0, 1.4
    hx, hy = 154 + vx * a, 3 + vy * a
    n = int(min(a + 1, 4) * 2.2)
    for k in range(n, -1, -1):
        f = k / max(n, 1)
        col = "star_hi" if k == 0 else "star" if f < 0.3 else "star_mid" if f < 0.65 else "star_dim"
        if k == 0 and a >= frames - 3:
            col = "star_mid"
        put(c, round(hx - vx / 3.3 * k), round(hy - vy / 3.3 * k), col)


def draw_city_lights(c, i):
    for k, on_at, off_at in FLICKER:
        if age(i, on_at) >= age(off_at, on_at):
            x, y, _ = WINDOWS[k]
            put(c, x, y, "bld")
    if i % 25 < 6:
        put(c, TOWER_X, TOWER_TOP - 1, "beacon")


# ---------------------------------------------------------------- the cart-pole

def cart_state(i):
    """Cart offset (px) and pole angle (rad, + leans right) at frame i."""
    t = i / FPS
    period = N / FPS
    x = theta = 0.0
    for k, amp, phase in HARMONICS:
        w = 2 * math.pi * k / period
        s = math.sin(w * t + phase)
        x += amp * s
        theta -= amp * w * w / (w * w * POLE_LEN + GRAVITY) * s
    return x, theta


def pose(i):
    """Hinge x, pole tip (x, y), pole angle and cart offset at frame i."""
    xo, theta = cart_state(i)
    hx = round(W / 2 + xo)
    tx = hx + round(POLE_LEN * math.sin(theta))
    ty = HINGE_Y - round(POLE_LEN * math.cos(theta))
    return hx, tx, ty, theta, xo


def cat_origin(tx, ty):
    return tx - CAT_W // 2, ty - len(PERCH) - CAT_H + 1


def draw_wheel(c, cx, cy, spin):
    blit(c, WHEEL, cx - 3, cy - 3, WHEEL_MAP)
    put(c, cx + round(1.4 * math.cos(spin)), cy + round(1.4 * math.sin(spin)), "hub")


def draw_cart(c, hx, spin):
    x0 = hx - 12
    for side in (-1, 1):  # open flaps
        base = x0 + 1 if side < 0 else x0 + 22
        stroke(c, [(base + 0.5, BOX_TOP + 0.5), (base + 0.5 + 4.2 * side, BOX_TOP - 3.5)],
               lambda k: 0.7, lambda k: "box_sh")
    blit(c, BOX, x0, BOX_TOP, BOX_MAP)
    blit(c, PAW, x0 + 16, BOX_TOP + 4, BOX_MAP)
    draw_wheel(c, x0 + 5, WHEEL_CY, spin)
    draw_wheel(c, x0 + 18, WHEEL_CY, spin)


def draw_shadow(c, hx):
    rect(c, hx - 11, GROUND_Y + 2, hx + 10, GROUND_Y + 2, "ground_dk")
    rect(c, hx - 8, GROUND_Y + 2, hx + 7, GROUND_Y + 2, "ground_dkr")
    rect(c, hx - 7, GROUND_Y + 3, hx + 6, GROUND_Y + 3, "ground_dk")


def draw_pole(c, hx, tx, ty):
    for y in range(ty, HINGE_Y + 1):
        s = (HINGE_Y - y) / max(HINGE_Y - ty, 1)
        x = round(hx + (tx - hx) * s - 2)
        put(c, x, y, "box_ol")
        put(c, x + 4, y, "box_ol")
        for j, base in ((1, "sisal_hi"), (2, "sisal"), (3, "sisal")):
            put(c, x + j, y, "sisal_sh" if (HINGE_Y - y + j) % 3 == 0 else base)
    put(c, hx, HINGE_Y, "bolt")
    for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        put(c, hx + ox, HINGE_Y + oy, "bolt_sh")


TAIL_REST = [1.45, 1.2, 0.85, 0.5, 0.2, 0.0, -0.1, -0.25, -0.6, -1.1, -1.7, -2.3]


def draw_tail(c, bx, by, theta, i):
    """Ringed tail hanging off the perch; it swings against the pole's lean."""
    swing = max(-0.7, min(0.7, -4.0 * theta)) + 0.12 * wave(i, 3)
    pts = [(bx + 0.5, by + 0.5)]
    for k, rest in enumerate(TAIL_REST):
        ang = rest + swing * min(1.0, k / 4) + 0.15 * wave(i, 3, -0.6 * k)
        x, y = pts[-1]
        pts.append((x + 1.1 * math.sin(ang), y + 1.1 * math.cos(ang)))
    n = len(TAIL_REST)
    stroke(c, pts, lambda k: 1.0 - 0.3 * k / n,
           lambda k: "cat_dk" if k % 4 == 3 or k >= n - 2 else "cat", outline="cat_ol")


def draw_cat(c, x, y, i, look):
    twitching = age(i, TWITCH[0]) < TWITCH[1]
    blit(c, CAT_TWITCH if twitching else CAT, x, y, CAT_MAP)
    eye = EYE_OPEN
    for s in BLINKS:
        if age(i, s) < len(BLINK_SHAPE):
            eye = BLINK_SHAPE[age(i, s)]
    for ex in EYE_X:
        for j in range(3):
            for k in range(3):
                put(c, x + ex + k + look, y + EYE_Y + j, eye[j][k])
    put(c, x + 3, y + 10, "cat_pink")
    put(c, x + CAT_W - 4, y + 10, "cat_pink")
    for side, wx in ((-1, x - 1), (1, x + CAT_W)):
        put(c, wx, y + 10, "whisker")
        put(c, wx + side, y + 10, "whisker")
        put(c, wx, y + 12, "whisker")
        put(c, wx + side, y + 13, "whisker")


def firefly(i, cx, cy):
    """The firefly the cat watches: a tilted orbit, passing behind its head and in front below."""
    phi = 2 * math.pi * i / N + 0.4
    dx = 25 * math.cos(phi)
    y = cy + 11 + 12 * math.sin(phi) + 1.5 * wave(i, 5)
    return cx + CAT_W // 2 + dx, y, math.sin(phi) > 0, dx


GRASS_FLIES = [(34, 62, 4, 1, 0.0), (63, 60, 3, 2, 2.0), (134, 61, 4, 1, 4.0), (164, 63, 3, 3, 1.0)]


def draw_glow(c, fx, fy, glow):
    x, y = round(fx), round(fy)
    if glow > 0.2:
        for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            put(c, x + ox, y + oy, "ff_dim")
        put(c, x, y, "ff_core")
    else:
        put(c, x, y, "ff" if glow > -0.5 else "ff_dim")


def draw_grass_flies(c, i):
    for x, y, r, cyc, ph in GRASS_FLIES:
        draw_glow(c, x + r * wave(i, cyc, ph), y + 0.6 * r * wave(i, 2 * cyc, ph + 1), wave(i, cyc + 1, ph))


def draw_rewards(c, i, cx, cy):
    """A reward per step, as in CartPole; each sparkle rises beside the cat and fades."""
    for j, s in enumerate(REWARDS):
        a = age(i, s)
        if a >= REWARD_LIFE:
            continue
        x = cx + (CAT_W + 4 if j % 2 == 0 else -5)
        y = cy + 8 - a // 3
        arm = 0 if a < 2 or a >= REWARD_LIFE - 3 else 1 if a < 5 or a >= REWARD_LIFE - 8 else 2
        dim = a >= REWARD_LIFE - 8
        for k in range(1, arm + 1):
            col = "spark_dim" if dim or k == 2 else "spark"
            for ox, oy in ((k, 0), (-k, 0), (0, k), (0, -k)):
                put(c, x + ox, y + oy, col)
        put(c, x, y, "spark" if dim else "spark_hi")


def render(i):
    i %= N
    c = BACKGROUND.copy()
    draw_stars(c, i)
    draw_shooting_star(c, i)
    draw_city_lights(c, i)
    draw_grass_flies(c, i)

    hx, tx, ty, theta, xo = pose(i)
    draw_shadow(c, hx)
    draw_cart(c, hx, xo / 3.0)
    draw_pole(c, hx, tx, ty)
    blit(c, PERCH, tx - len(PERCH[0]) // 2, ty - len(PERCH), PERCH_MAP)
    cx, cy = cat_origin(tx, ty)
    fx, fy, in_front, dx = firefly(i, cx, cy)
    glow = wave(i, 3, 0.7)
    if not in_front:
        draw_glow(c, fx, fy, glow)
    draw_tail(c, cx + CAT_W - 3, cy + CAT_H - 3, theta, i)
    draw_cat(c, cx, cy, i, -1 if dx < -6 else 1 if dx > 6 else 0)
    if in_front:
        draw_glow(c, fx, fy, glow)
    draw_rewards(c, i, cx, cy)
    return c


# ---------------------------------------------------------------- output

FLAT_PALETTE = [v for _, h in PALETTE for v in bytes.fromhex(h)]


def to_image(frame, scale=SCALE):
    big = frame.repeat(scale, axis=0).repeat(scale, axis=1)
    im = Image.fromarray(big, "P")
    im.putpalette(FLAT_PALETTE)
    return im


def write_gif(path):
    frames = [to_image(render(i)) for i in range(N)]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=1000 // FPS,
                   loop=0, optimize=True, disposal=1)


def write_sheet(path, picks=(0, 19, 38, 57, 76, 95, 114, 133), scale=3, cols=2):
    rows = math.ceil(len(picks) / cols)
    sheet = Image.new("RGB", (cols * W * scale + (cols - 1) * 6, rows * H * scale + (rows - 1) * 6), "white")
    for k, i in enumerate(picks):
        sheet.paste(to_image(render(i), scale).convert("RGB"),
                    ((k % cols) * (W * scale + 6), (k // cols) * (H * scale + 6)))
    sheet.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=Path(__file__).parent / "assets" / "catpole.gif")
    ap.add_argument("--sheet", help="write a PNG contact sheet instead of the GIF")
    args = ap.parse_args()
    if args.sheet:
        write_sheet(args.sheet)
    else:
        write_gif(args.out)


if __name__ == "__main__":
    main()
