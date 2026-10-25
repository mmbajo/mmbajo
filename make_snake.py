#!/usr/bin/env python3
"""A Tonkinese cat eats a GitHub contribution graph, snake style, as a looping pixel-art GIF.

The graph is the public one from github.com/users/<user>/contributions. The cat runs
the rows like the snake in Snake, eating every day that has contributions for a +1.
Once the year is gone it naps while the graph grows back, then runs off the right
edge and comes back on the left, so the loop is seamless.

The art is 400x72, scaled up 4x with nearest-neighbour and meant to be shown 800px
wide. One GIF is written per GitHub theme, in GitHub's own contribution colours.

    python make_snake.py                    # writes assets/catsnake-{light,dark}.gif
    python make_snake.py --user octocat     # someone else's graph
"""

import argparse
import datetime as dt
import math
import random
import re
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

W, H = 400, 72
SCALE = 4
FPS = 25
WEEKS, DAYS = 53, 7
CELL, PITCH = 5, 7
GRID_X, GRID_Y = 15, 17  # headroom above the top row for its +1s
SPEED = 5  # px per frame
HOP = 8  # frames to turn round at the end of a row
JUMP = 3  # px the hop rises
CRUMB_LIFE = 7
REWARD_LIFE = 10
RISE = 3  # a +1 starts this far above its day and floats up 1px a frame
NAP = 80  # frames asleep; the graph grows back meanwhile
POP = 3  # frames for a cell to grow back

# Colours that follow the GitHub theme: the page background, contribution levels 0-4
# (Primer's), the crumbs knocked off cells of levels 1-4, and the +1 for each day eaten.
THEMES = {
    "light": ("ffffff", ["eff2f5", "aceebb", "4ac26b", "2da44e", "116329"], ["4ac26b", "4ac26b", "2da44e", "116329"],
              "1a7f37"),
    "dark": ("0d1117", ["151b23", "033a16", "196c2e", "2ea043", "56d364"], ["2ea043", "2ea043", "56d364", "56d364"],
             "56d364"),
}
THEME_NAMES = (["bg"] + [f"lv{k}" for k in range(5)] + [f"lv{k}c" for k in range(5)]
               + [f"crumb{k}" for k in range(1, 5)] + ["plus", "plus_dim"])
# The cat looks the same on both: a platinum mink is pale silvery grey with warm
# overtones, frosty grey points, lavender-pink nose and aqua eyes.
SHARED = [
    ("cat_ol", "4d4552"), ("cat_hi", "ece6e0"), ("cat", "d6cdc5"), ("cat_sh", "b6aca5"),
    ("mask", "b9b0b6"), ("point", "978f9b"), ("point_dk", "776f7e"), ("ear", "d9b3c2"),
    ("eye", "3fd1c6"), ("pupil", "2a2433"), ("nose", "c890a8"),
    ("bubble", "9fbfe0"), ("bubble_hi", "f4faff"),
]
C = {name: i for i, name in enumerate(THEME_NAMES + [n for n, _ in SHARED])}

CAT_MAP = {"o": "cat_ol", "H": "cat_hi", "B": "cat", "S": "cat_sh", "m": "mask", "P": "point",
           "Q": "point_dk", "i": "ear", "e": "eye", "k": "pupil", "n": "nose"}

# Facing right. A running pose's mouth is REACH columns ahead of its centre column and
# FEET rows above its paws; every sprite stands on its bottom row.
HEAD = [
    "...oo...oo..",
    "..oPPo.oPPo.",
    ".oPiPPoPiPo.",
    ".oHHHHHHHHo.",
    "oHBBBBBBBBBo",
    "oBBBBBBmekPo",
    "oBBBBBBmeePn",
    "oBBBBBmPPPPo",
]
TAIL_UP = [".", ".oo", "oQQo", "oQPo", ".oPPo", "..oPPo", "...oPP", "....oS", "...oSS"]
TAIL_OUT = [".", ".", ".", "oo", "oQoo", "ooPPPo", "..ooPP", "....oS", "...oSS"]


def runner(tail, legs):
    """A running pose: the tail, head and back shared by every stride, plus its legs."""
    rows = [tail[y].ljust(17, ".") + HEAD[y] for y in range(5)]
    rows.append(tail[5].ljust(17, "o") + HEAD[5])
    rows.append(tail[6].ljust(18, "H") + HEAD[6][1:])
    rows.append(tail[7].ljust(18, "B") + HEAD[7][1:])
    rows.append(tail[8].ljust(23, "B") + "mPPoo.")
    return rows + ["...oSSBBBBBBBBBBBBBBoooo....."] + legs


SPRITES = {
    "A": runner(TAIL_UP, [  # reaching out
        "...oSSSBBBBBBBBBBBBSSo.......",
        "..oPPSSoooooooooSSQPPo.......",
        ".oPQPo..........oQQPPPo......",
        "oPQPo............oQQPPPo.....",
        "oQQo..............oQQoPPPo...",
        "ooo................oo.oooo...",
    ]),
    "B": runner(TAIL_OUT, [  # front paws land
        "...oSSSBBBBBBBBBBBBSSo.......",
        "...oSSPPoooooooooSQPPo.......",
        "....oPPPPo......oQQPPo.......",
        ".....ooPPPo.....oQQPPo.......",
        ".......oQQQo....oQQPPPo......",
        "........oooo....oooooo.......",
    ]),
    "C": runner(TAIL_UP, [  # gathered
        "...oSSSBBBBBBBBBBBBSSo.......",
        "...oSSSPoooooooQQPPPo........",
        "....oPPPPo...oQQPPoo.........",
        ".....ooPPPooQQPPoo...........",
        ".......oQQoQQQoo.............",
        "........oooooo...............",
    ]),
    "D": runner(TAIL_OUT, [  # hind paws push off
        "...oSSSBBBBBBBBBBBBSSoooo....",
        "..oPPPSoooooooooSSQQPPPPo....",
        "..oPQPo.........oooooQQPPPo..",
        "..oPQPo..............ooQQQo..",
        "..oQQQo................ooo...",
        "..ooooo......................",
    ]),
}
SPRITES["sleep"] = [
    "..............oo..oo.",
    ".....ooooo...oPPooPPo",
    "...ooHHHHHoo.oPiPPiPo",
    "..oHHBBBBBHHoHHHHHHHo",
    ".oHBBBBBBBBBBBBBBBBBo",
    "oHBBBBBBBBBBBBBBmBBPo",
    "oBBBBBBBBBBBBBBBkkPPn",
    "oSBBBBBBBBBBBBBmPPPPo",
    "oSSBBBBBBBQQQQPPPPPo.",
    ".oSSSQQQQQQPPPPPPPo..",
    "..ooooooooooooooooo..",
]
SPRITES["wake"] = SPRITES["sleep"][:5] + ["oHBBBBBBBBBBBBBBekBPo", "oBBBBBBBBBBBBBBBeePPn"] + SPRITES["sleep"][7:]
SPRITES["stretch"] = [
    ".oo..........................",
    "oQQo.........................",
    "oQPo.........................",
    ".oPPo........................",
    "..oPPoooooo..................",
    "...oPHHHHHHoo................",
    "...oBBBBBBBHHoo..oo...oo.....",
    "...oBBBBBBBBBHHooPPo.oPPo....",
    "...oSBBBBBBBBBBHoPiPPoPiPo...",
    "...oSBBBBBBBBBBBoHHHHHHHHo...",
    "...oSSBBBBBBBBBoHBBBBBBBBBo..",
    "...oSSSooooBBBBoBBBBBBmekPo..",
    "...oPPo....oSSBBBBBBBBmeePn..",
    "...oPPo.....oSBBBBBBBmPPPPo..",
    "...oQQo.....oQQPPPPPPPPPPPPo.",
    "...oooo......ooooooooooooooo.",
]
PLUS_ONE = [
    ".....#",
    ".#..##",
    "###..#",
    ".#...#",
    ".....#",
]
BOB = {"A": -1, "C": -1}  # the body rises while all four paws are off the ground
RUN = ["A", "B", "C", "D"]
REACH, FEET = 14, 8


def put(c, x, y, col):
    if 0 <= x < W and 0 <= y < H:
        c[y, x] = C[col]


def rect(c, x0, y0, x1, y1, col):
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W - 1), min(y1, H - 1)
    if x0 <= x1 and y0 <= y1:
        c[y0:y1 + 1, x0:x1 + 1] = C[col]


def blit(c, rows, x, y, cmap, flip=False):
    for j, row in enumerate(rows):
        for i, ch in enumerate(row[::-1] if flip else row):
            if ch != ".":
                put(c, x + i, y + j, cmap[ch])


# ---------------------------------------------------------------- the graph

def fetch_graph(user):
    req = urllib.request.Request(f"https://github.com/users/{user}/contributions",
                                 headers={"User-Agent": "make_snake.py"})
    return parse_graph(urllib.request.urlopen(req, timeout=30).read().decode())


def parse_graph(html):
    """Contribution levels as a (7, 53) array, Sunday at the top; -1 where there is no day."""
    days = {}
    for tag in re.findall(r"<td[^>]*ContributionCalendar-day[^>]*>", html):
        date = re.search(r'data-date="([\d-]+)"', tag).group(1)
        days[dt.date.fromisoformat(date)] = int(re.search(r'data-level="(\d)"', tag).group(1))
    sunday = min(days) - dt.timedelta(days=(min(days).weekday() + 1) % 7)
    grid = np.full((DAYS, WEEKS), -1, int)
    for day, level in days.items():
        k = (day - sunday).days
        grid[k % 7, k // 7] = level
    return grid


def cell_x(col):
    return GRID_X + PITCH * col


def cell_y(row):
    return GRID_Y + PITCH * row


# ---------------------------------------------------------------- the route

def plan(grid):
    """Walk the cat through the graph.

    Returns the cat per frame as (centre x, mouth y, facing, sprite), the frame each
    cell is eaten (-1 if never) and the frame it starts growing back.
    """
    poses = []
    eaten = np.full(grid.shape, -1)

    def eat(row, x0, x1):
        for col in range(WEEKS):
            if grid[row, col] > 0 and eaten[row, col] < 0 and x0 <= cell_x(col) + CELL // 2 <= x1:
                eaten[row, col] = len(poses) - 1

    cx, facing, stride = -REACH - 1, 1, 0
    for row in range(DAYS):
        y = cell_y(row) + CELL // 2
        cols = [col for col in range(WEEKS) if grid[row, col] >= 0]
        end = cell_x(cols[-1]) + CELL if facing > 0 else cell_x(cols[0]) - 1
        while (end - (cx + REACH * facing)) * facing > 0:
            mouth = cx + REACH * facing
            cx += SPEED * facing
            poses.append((cx, y, facing, RUN[stride // 2 % 4]))
            eat(row, *sorted((mouth, cx + REACH * facing)))
            stride += 1
        if row == DAYS - 1:
            break
        for k in range(1, HOP + 1):  # hop down a row, turning round at the top
            u = k / HOP
            poses.append((cx + facing * SPEED * HOP * u * (1 - u), y + PITCH * u - 4 * JUMP * u * (1 - u),
                          facing if u < 0.5 else -facing, "C"))
        facing = -facing
        eat(row + 1, *sorted((cx + REACH * facing, cx - REACH * facing)))  # lands on the row's end
        stride = 2  # front paws first

    for v in (4, 3, 2, 1):  # brake
        cx += v
        poses.append((cx, y, 1, "B"))
    nap = len(poses)
    poses += [(cx, y, 1, "sleep")] * NAP + [(cx, y, 1, "wake")] * 8 + [(cx, y, 1, "stretch")] * 14
    v = 0
    while cx - REACH < W:  # run off the right edge; the loop brings it back on the left
        v = min(SPEED, v + 1)
        cx += v
        poses.append((cx, y, 1, RUN[stride // 2 % 4]))
        stride += 1

    # the graph grows back roughly left to right through the middle of the nap
    rng = random.Random(7)
    cells = sorted(zip(*np.nonzero(eaten >= 0)), key=lambda rc: rc[1] + 8 * rng.random())
    grown = np.full(grid.shape, -1)
    for k, (row, col) in enumerate(cells):
        grown[row, col] = nap + 10 + (NAP - 26) * k // len(cells)
    return poses, eaten, grown


# ---------------------------------------------------------------- drawing

def draw_cell(c, row, col, level, size=CELL):
    x, y = cell_x(col) + (CELL - size) // 2, cell_y(row) + (CELL - size) // 2
    rect(c, x, y, x + size - 1, y + size - 1, f"lv{level}")
    if size == CELL:  # soften the corners, as GitHub rounds them
        for cx, cy in ((x, y), (x + size - 1, y), (x, y + size - 1), (x + size - 1, y + size - 1)):
            put(c, cx, cy, f"lv{level}c")


def draw_graph(c, grid, eaten, grown, t):
    for row in range(DAYS):
        for col in range(WEEKS):
            level = grid[row, col]
            if level < 0:
                continue
            if level > 0 and eaten[row, col] <= t < grown[row, col] + POP:
                draw_cell(c, row, col, 0)
                if t >= grown[row, col]:
                    draw_cell(c, row, col, level, 1 + 2 * (t - grown[row, col]))
            else:
                draw_cell(c, row, col, level)


def draw_crumbs(c, grid, eaten, t):
    for row, col in zip(*np.nonzero((eaten >= 0) & (eaten <= t) & (t < eaten + CRUMB_LIFE))):
        a = t - eaten[row, col]
        rng = random.Random(row * WEEKS + col)
        x0, y0 = cell_x(col) + 2, cell_y(row) + 2
        for _ in range(3):
            vx, vy = rng.uniform(-1.6, 1.6), rng.uniform(-2.2, -1.0)
            put(c, round(x0 + vx * a), round(y0 + vy * a + 0.25 * a * a), f"crumb{grid[row, col]}")


def draw_rewards(c, eaten, t):
    """A +1 floats up off each day the frame after it is eaten, from behind the cat's head."""
    for row, col in zip(*np.nonzero((eaten >= 0) & (eaten < t) & (t < eaten + REWARD_LIFE))):
        a = t - eaten[row, col]
        blit(c, PLUS_ONE, cell_x(col), cell_y(row) - RISE - a, {"#": "plus" if a < REWARD_LIFE - 4 else "plus_dim"})


def draw_cat(c, cx, y, facing, name):
    rows = SPRITES[name]
    w = len(rows[0])
    top = round(y) + FEET - (len(rows) - 1) + BOB.get(name, 0)
    blit(c, rows, round(cx) - w // 2, top, CAT_MAP, flip=facing < 0)


def draw_bubble(c, cx, y, t):
    """A snot bubble swelling and shrinking with each breath."""
    r = 1.6 + 2.0 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 40))
    bx, by = round(cx + 11 + r), round(y) + 3
    for py in range(by - 4, by + 5):
        for px in range(bx - 4, bx + 5):
            if abs(math.hypot(px - bx, py - by) - r) < 0.55:
                put(c, px, py, "bubble")
    put(c, bx + round(r * 0.5), by - round(r * 0.5), "bubble_hi")


def render(t, grid, route):
    poses, eaten, grown = route
    c = np.full((H, W), C["bg"], np.uint8)
    draw_graph(c, grid, eaten, grown, t)
    draw_crumbs(c, grid, eaten, t)
    draw_rewards(c, eaten, t)
    draw_cat(c, *poses[t])
    if poses[t][3] == "sleep":
        draw_bubble(c, poses[t][0], poses[t][1], t)
    return c


# ---------------------------------------------------------------- output

def blend(a, b, f):
    return bytes(round(x + (y - x) * f) for x, y in zip(bytes.fromhex(a), bytes.fromhex(b))).hex()


def palette(theme):
    bg, levels, crumbs, plus = THEMES[theme]
    corners = [blend(lv, bg, 0.35) for lv in levels]
    rewards = [plus, blend(plus, bg, 0.55)]
    return [v for h in [bg] + levels + corners + crumbs + rewards + [h for _, h in SHARED] for v in bytes.fromhex(h)]


def to_image(frame, theme, scale=SCALE):
    big = frame.repeat(scale, axis=0).repeat(scale, axis=1)
    im = Image.fromarray(big, "P")
    im.putpalette(palette(theme))
    return im


def poster_frame(route):
    """Start the GIF mid-graph, so a paused GIF still shows the cat eating."""
    poses = route[0]
    return next(t for t, (cx, y, facing, _) in enumerate(poses) if y == cell_y(3) + CELL // 2 and cx < W / 2)


def write_gifs(grid, out_dir):
    route = plan(grid)
    n = len(route[0])
    start = poster_frame(route)
    frames = [render((start + i) % n, grid, route) for i in range(n)]
    for theme in THEMES:
        ims = [to_image(f, theme) for f in frames]
        ims[0].save(out_dir / f"catsnake-{theme}.gif", save_all=True, append_images=ims[1:],
                    duration=1000 // FPS, loop=0, optimize=True, disposal=1)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--user", default="mmbajo")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "assets")
    args = ap.parse_args()
    n = write_gifs(fetch_graph(args.user), args.out)
    print(f"{n} frames, {n / FPS:.1f} s loop")


if __name__ == "__main__":
    main()
