#!/usr/bin/env python3
"""Two Russian Blues clear a maze in the style of Pac-Man, as a looping pixel-art GIF.

The cats climb out of a cardboard box in the middle and each sweeps its half of the
maze on the shortest round trip that eats everything on it: kibble +1, catnip +5 and
fish +10. Back in the box, they watch the walls flash and the food come back, then
set off again, so the loop is seamless.

The art is 400x130, scaled up 4x with nearest-neighbour and meant to be shown 800px
wide. One GIF is written per GitHub theme.

    python make_maze.py                    # writes assets/catmaze-{light,dark}.gif
"""

import argparse
import math
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

T = 10  # px per tile
SCALE = 4
FPS = 25
SPEED = 4  # px per frame
REACH = 6  # px ahead of a cat's centre that it eats at
HOP = 8  # frames to climb out of or into the box
LIFE = 14  # frames a reward floats for
FLASHES, FLASH = 4, 6  # the walls flash this many times, for this many frames each
REGROW = 16  # frames for the food to come back, spreading out from the box
BLINK = 20  # frames per blink of the catnip; the loop is a whole number of them

# The left half; the right half mirrors it. # wall, . kibble, o catnip, f fish, B the box.
LEFT = [
    "####################",
    "#o................##",
    "#.###.#####.#####.##",
    "#.###.#####.#####.##",
    "#...................",
    "#.####.#####.###.BBB",
    "#f####.#####.###.BBB",
    "#.####.#####.###.BBB",
    "#...................",
    "#.###.####.######.##",
    "#.###.####.######.##",
    "#o................##",
    "####################",
]
MAZE = [row + row[::-1] for row in LEFT]
ROWS, COLS = len(MAZE), len(MAZE[0])
W, H = COLS * T, ROWS * T
REWARD = {".": 1, "o": 5, "f": 10}
HOME = [(4, 18), (4, 21)]  # the tile each cat climbs out onto, left cat first
IN_BOX = 57  # y of a cat's centre while it sits in the box

THEMES = {
    "light": dict(bg="ffffff", wall="0969da", flash="b6e3ff", kibble="b07a45", nip="1a7f37", nip_hi="4ac26b",
                  fish="fb8f44", fish_dk="bc4c00", plus1="7d4e24", plus5="1a7f37", plus10="bc4c00"),
    "dark": dict(bg="0d1117", wall="4493f8", flash="cae8ff", kibble="e3b37f", nip="3fb950", nip_hi="7ee787",
                 fish="ffa657", fish_dk="db6d28", plus1="f0d0a8", plus5="7ee787", plus10="ffa657"),
}
DIMMED = ["plus1", "plus5", "plus10"]  # faded towards the background as a reward ends
THEME_NAMES = list(THEMES["light"]) + [k + "_dim" for k in DIMMED]
# The cats look the same on both: slate-blue with silver tipping, green eyes, mauve
# inner ears; and so does the box.
SHARED = [
    ("cat_ol", "2c3440"), ("cat_hi", "b9c6d8"), ("cat", "8d9db3"), ("cat_sh", "6e7f96"), ("cat_dk", "56667c"),
    ("eye", "46d47e"), ("pupil", "10231a"), ("nose", "6c6f86"), ("ear", "c4a2b6"),
    ("box_ol", "6b4a2b"), ("box_hi", "ddb985"), ("box", "c9a06a"), ("box_sh", "a77c49"), ("box_in", "7a5634"),
    ("tape", "e9d7ae"),
]
C = {name: i for i, name in enumerate(THEME_NAMES + [n for n, _ in SHARED])}

CAT_MAP = {"o": "cat_ol", "H": "cat_hi", "B": "cat", "S": "cat_sh", "D": "cat_dk", "e": "eye", "k": "pupil",
           "n": "nose", "i": "ear"}
# Facing right, drawn centred on the cat's position.
SIDE = {
    "A": [  # reaching out
        "............o...o...",
        "...........oio.oio..",
        ".oo........oiBoBiBo.",
        "oDDo.......oHHHHHHo.",
        ".oDo......oHBBBBBBBo",
        "..oSo.oooooBBBBBekBo",
        "...oSoHHHHBBBBBBeeBn",
        "...oBBBBBBBBBBBBBBo.",
        "...oSBBBBBBBBBBBBo..",
        "....oSSSooooooSSSo..",
        "...oSSo.......oSSo..",
        "..oDDo.........oSSo.",
        "..ooo...........ooo.",
    ],
    "B": [  # front paws land
        "............o...o...",
        "...........oio.oio..",
        "...........oiBoBiBo.",
        ".oo........oHHHHHHo.",
        "oDDo......oHBBBBBBBo",
        ".oDSooooooBBBBBBekBo",
        "..oSHHHHHHBBBBBBeeBn",
        "..oBBBBBBBBBBBBBBBo.",
        "..oSBBBBBBBBBBBBBo..",
        "...oSSSoooooooSSo...",
        "....oSSo.....oSSo...",
        ".....oDDo...oSSo....",
        "......ooo...ooo.....",
    ],
    "C": [  # gathered
        "............o...o...",
        "...........oio.oio..",
        ".oo........oiBoBiBo.",
        "oDDo.......oHHHHHHo.",
        ".oDo......oHBBBBBBBo",
        "..oSo.oooooBBBBBekBo",
        "...oSoHHHHBBBBBBeeBn",
        "...oBBBBBBBBBBBBBBo.",
        "...oSBBBBBBBBBBBBo..",
        "....oSSoooooSSSSo...",
        ".....oSSo.oSSoo.....",
        "......oDDoSSo.......",
        ".......oooooo.......",
    ],
    "D": [  # hind paws push off
        "............o...o...",
        "...........oio.oio..",
        "...........oiBoBiBo.",
        ".oo........oHHHHHHo.",
        "oDDo......oHBBBBBBBo",
        ".oDSooooooBBBBBBekBo",
        "..oSHHHHHHBBBBBBeeBn",
        "..oBBBBBBBBBBBBBBBo.",
        "..oSBBBBBBBBBBBBBoo.",
        "..oSSSooooooooSSSSo.",
        "..oSSo........ooSSSo",
        "..oDDo..........oSSo",
        "..oooo...........ooo",
    ],
}
FACE = [
    ".o.........o.",
    "oio.......oio",
    "oiio.....oiio",
    "oBBBoooooBBBo",
    "oHHHHHHHHHHHo",
    "oBBBBBBBBBBBo",
    "oBekeBBBekeBo",
    "oBBBBBnBBBBBo",
    ".oBBBBBBBBBo.",
]
BACK = [
    ".o.........o.",
    "oDo.......oDo",
    "oBDo.....oDBo",
    "oBBHoooooHBBo",
    "oBBBHHHHHBBBo",
    "oBBBBBBBBBBBo",
    "oBBBBBBBBBBBo",
    "oSBBBBBBBBBSo",
    ".oSSSSSSSSSo.",
]
SPRITES = dict(SIDE)
SPRITES["down1"] = FACE + ["..ooSBBBSoo..", "..oSBBBBBSo..", "..oSo.oSSSo..", "..oSo..oSo...", "..ooo..ooo..."]
SPRITES["down2"] = FACE + ["..ooSBBBSoo..", "..oSBBBBBSo..", "..oSSSo.oSo..", "...oSo..oSo..", "...ooo..ooo.."]
SPRITES["up1"] = BACK + ["..oSBBBBBSo.o", "..oSBBBBBSoDo", "..oSo.oSSSoDo", "..oSo..oSoDo.", "..ooo..ooooo."]
SPRITES["up2"] = BACK + ["..oSBBBBBSo.o", "..oSBBBBBSoDo", "..oSSSo.oSoDo", "...oSo..oSDo.", "...ooo..oooo."]
SPRITES["sit"] = SPRITES["down1"]  # in the box only the head shows
SPRITES["blink"] = FACE[:6] + ["oBoooBBBoooBo"] + FACE[7:] + SPRITES["down1"][len(FACE):]
STRIDE = 8  # px per change of pose

NIP = [  # a catnip leaf
    "..ab..",
    ".abbb.",
    "abbbbb",
    "abbbbb",
    ".bbbb.",
    "..b...",
]
FISH = [
    "...bbb....",
    ".bbaaab.bb",
    "baaaaaabab",
    "bkaaaaaaab",
    ".baaaaab.b",
    "..bbbbb...",
]
FONT = {
    "+": ["...", ".#.", "###", ".#.", "..."],
    "0": ["###", "#.#", "#.#", "#.#", "###"],
    "1": [".#", "##", ".#", ".#", ".#"],
    "5": ["###", "#..", "##.", "..#", "##."],
}


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


def centre(tile):
    r, c = tile
    return T * c + T // 2, T * r + T // 2


# ---------------------------------------------------------------- the walls

def shift(m, dy, dx, fill):
    """m moved down by dy and right by dx, padded with fill."""
    out = np.full_like(m, fill)
    h, w = m.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = m[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def disk(r):
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dy * dy + dx * dx <= r * r + 0.5]


def erode(m, r):
    out = m.copy()
    for dy, dx in disk(r):
        out &= shift(m, dy, dx, True)
    return out


def dilate(m, r):
    out = m.copy()
    for dy, dx in disk(r):
        out |= shift(m, dy, dx, False)
    return out


def outline(m):
    inner = m.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        inner &= shift(m, dy, dx, True)
    return m & ~inner


def walls():
    """Wall lines in the arcade style: each block a rounded outline 5px in from the
    corridors around it, and a second line round the outside wall."""
    tiles = np.array([[ch != "#" for ch in row] for row in MAZE])
    open_px = np.kron(tiles, np.ones((T, T), bool))
    dist = np.full(open_px.shape, np.inf)
    for dy, dx in disk(9):
        dist = np.where(shift(open_px, dy, dx, False), np.minimum(dist, math.hypot(dy, dx)), dist)
    blocks = dilate(erode(dist > 4.5, 3), 3)  # rounds the blocks' corners
    outer = np.zeros(tiles.shape, bool)  # wall tiles joined to the edge of the maze
    todo = [(0, 0)]
    while todo:
        r, c = todo.pop()
        if 0 <= r < ROWS and 0 <= c < COLS and not tiles[r, c] and not outer[r, c]:
            outer[r, c] = True
            todo += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
    far = dilate(erode(dist > 7.5, 1), 1) & np.kron(outer, np.ones((T, T), bool))
    return outline(blocks) | outline(far)


# ---------------------------------------------------------------- the routes

def half(side):
    return [(r, c) for r in range(ROWS) for c in range(COLS)
            if MAZE[r][c] in REWARD and (c < COLS // 2) == (side == 0)]


def neighbours(tile, side):
    r, c = tile
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        if MAZE[r + dr][c + dc] in REWARD and (c + dc < COLS // 2) == (side == 0):
            yield r + dr, c + dc


def shortest(src, dst, side, edges=None):
    """A shortest path from src to dst within a cat's half, or over the given edges only."""
    prev = {src: None}
    todo = deque([src])
    while todo:
        t = todo.popleft()
        for n in neighbours(t, side):
            if n not in prev and (edges is None or edges[min(t, n), max(t, n)]):
                prev[n] = t
                todo.append(n)
    if dst not in prev:
        return None
    path = [dst]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


def tour(side):
    """The shortest round trip from a cat's home tile over every tile of its half.

    That is the Chinese postman's walk: every corridor once, plus the corridors that
    pair up the odd junctions most cheaply, again. It is walked as an Euler circuit
    that eats before it backtracks and goes straight, or else turns right, where it can.
    """
    tiles = half(side)
    edges = Counter()
    for t in tiles:
        for n in neighbours(t, side):
            if t < n:
                edges[t, n] += 1
    odd = [t for t in tiles if len(list(neighbours(t, side))) % 2]
    dist = [[len(shortest(a, b, side)) - 1 for b in odd] for a in odd]

    @lru_cache(None)
    def pairing(left):
        """The cheapest pairs of the odd junctions numbered in left, as (cost, pairs)."""
        if not left:
            return 0, ()
        i, rest = left[0], left[1:]
        options = []
        for k, j in enumerate(rest):
            cost, pairs = pairing(rest[:k] + rest[k + 1:])
            options.append((cost + dist[i][j], ((i, j),) + pairs))
        return min(options)

    for i, j in pairing(tuple(range(len(odd))))[1]:
        path = shortest(odd[i], odd[j], side)
        for a, b in zip(path, path[1:]):
            edges[min(a, b), max(a, b)] += 1

    walk, eaten, heading = [HOME[side]], {HOME[side]}, (-1, 0)
    turn = {(-1, 0): (0, 1), (0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0)}  # right of each heading
    while sum(edges.values()):
        t = walk[-1]

        def rank(n):
            key = min(t, n), max(t, n)
            edges[key] -= 1
            bridge = edges[key] == 0 and shortest(n, t, side, edges) is None and sum(edges.values()) > 0
            edges[key] += 1
            d = (n[0] - t[0], n[1] - t[1])
            return bridge, n in eaten, [heading, turn[heading], turn[turn[turn[heading]]], turn[turn[heading]]].index(d)

        n = min((n for n in neighbours(t, side) if edges[min(t, n), max(t, n)]), key=rank)
        edges[min(t, n), max(t, n)] -= 1
        heading = (n[0] - t[0], n[1] - t[1])
        walk.append(n)
        eaten.add(n)
    assert walk[-1] == HOME[side] and set(walk) == set(tiles)
    return walk


def run(walk):
    """The cat along its walk, a frame at a time: (x, y, heading, px travelled), and the
    frame each tile is eaten, counted from leaving home."""
    pts = [centre(t) for t in walk]
    frames, eaten = [], {}
    total = T * (len(walk) - 1)
    marks = {}
    for k, t in enumerate(walk):
        marks.setdefault(t, T * k)
    s = 0
    while True:
        k = min(s // T, len(walk) - 2)
        f = (s - T * k) / T
        (x0, y0), (x1, y1) = pts[k], pts[k + 1]
        frames.append((round(x0 + (x1 - x0) * f), round(y0 + (y1 - y0) * f), (y1 - y0, x1 - x0), s))
        for t, m in marks.items():
            if t not in eaten and s + REACH >= m:
                eaten[t] = len(frames) - 1
        if s >= total:
            return frames, eaten
        s = min(s + SPEED, total)


def plan():
    """Both cats over the whole loop, when each tile is eaten and grows back, when the
    walls flash, and the loop's length.

    Each cat's frames are (x, y, sprite, facing left). The loop starts as they climb
    out and ends with them sitting in the box, the food all back."""
    runs = [run(tour(side)) for side in (0, 1)]
    busy = max(len(frames) for frames, _ in runs)
    flash = HOP + busy + HOP + 10
    regrow = flash + FLASHES * 2 * FLASH
    n = regrow + REGROW + 14
    n += -n % BLINK

    cats, eaten = [], {}
    for side, (frames, ate) in enumerate(runs):
        hx, hy = centre(HOME[side])
        poses = []
        for k in range(HOP):  # climb out
            u = (k + 1) / HOP
            poses.append((hx, round(IN_BOX + (hy - IN_BOX) * u - 8 * u * (1 - u)), "up1", False))
        for x, y, (dy, dx), s in frames:
            if dx:
                poses.append((x, y, "ABCD"[s // STRIDE % 4], dx < 0))
            else:
                poses.append((x, y, ("up1", "up2", "down1", "down2")[(dy > 0) * 2 + s // STRIDE % 2], False))
        for k in range(HOP):  # climb in
            u = (k + 1) / HOP
            poses.append((hx, round(hy + (IN_BOX - hy) * u - 8 * u * (1 - u)), "down1", False))
        while len(poses) < n:
            poses.append((hx, IN_BOX, "blink" if (len(poses) + 37 * side) % 80 < 4 else "sit", False))
        cats.append(poses)
        eaten.update({tile: HOP + k for tile, k in ate.items()})

    # the food comes back spreading out from the box
    far, todo = {}, deque((t, 0) for t in HOME)
    while todo:
        t, d = todo.popleft()
        if t not in far:
            far[t] = d
            todo += [((t[0] + dr, t[1] + dc), d + 1) for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0))
                     if MAZE[t[0] + dr][t[1] + dc] in REWARD]
    grown = {t: regrow + REGROW * far[t] // max(far.values()) for t in eaten}
    return cats, eaten, grown, flash, n


# ---------------------------------------------------------------- drawing

def draw_food(c, eaten, grown, t):
    for tile, k in eaten.items():
        if k <= t < grown[tile]:
            continue
        ch, (x, y) = MAZE[tile[0]][tile[1]], centre(tile)
        if ch == ".":
            rect(c, x - 1, y - 1, x, y, "kibble")
        elif ch == "o" and t % BLINK < BLINK - 6:
            blit(c, NIP, x - 3, y - 3, {"a": "nip_hi", "b": "nip"})
        elif ch == "f":
            blit(c, FISH, x - 5, y - 3, {"a": "fish", "b": "fish_dk", "k": "pupil"}, flip=tile[1] < COLS // 2)


def draw_rewards(c, eaten, t):
    """+1, +5 or +10 floats up from above the cat as it eats, on a halo of background."""
    for tile, k in eaten.items():
        a = t - k
        if not 0 <= a < LIFE:
            continue
        text = f"+{REWARD[MAZE[tile[0]][tile[1]]]}"
        glyphs = [FONT[ch] for ch in text]
        w = sum(len(g[0]) for g in glyphs) + len(glyphs) - 1
        x, y = centre(tile)
        x, y = x - w // 2, max(y - 12 - a // 2, 1)
        col = f"plus{text[1:]}" + ("_dim" if a >= LIFE - 4 else "")
        pixels = []
        for g in glyphs:
            pixels += [(x + i, y + j) for j, row in enumerate(g) for i, ch in enumerate(row) if ch == "#"]
            x += len(g[0]) + 1
        for px, py in pixels:
            rect(c, px - 1, py - 1, px + 1, py + 1, "bg")
        for px, py in pixels:
            put(c, px, py, col)


def draw_cat(c, x, y, name, flip):
    rows = SPRITES[name]
    blit(c, rows, x - len(rows[0]) // 2, y - 7, CAT_MAP, flip)


BOX = (174, 225, 58, 75)  # the front of the box: left, right, top, bottom


def draw_box_back(c):
    x0, x1, y0, _ = BOX
    rect(c, x0 + 2, y0 - 6, x1 - 2, y0 - 1, "box_ol")
    rect(c, x0 + 3, y0 - 5, x1 - 3, y0 - 5, "box_sh")
    rect(c, x0 + 3, y0 - 4, x1 - 3, y0 - 1, "box_in")


def draw_box_front(c):
    x0, x1, y0, y1 = BOX
    rect(c, x0, y0, x1, y1, "box_ol")
    rect(c, x0 + 1, y0 + 1, x1 - 1, y1 - 1, "box")
    rect(c, x0 + 1, y0 + 1, x1 - 1, y0 + 1, "box_hi")
    rect(c, x0 + 1, y1 - 2, x1 - 1, y1 - 1, "box_sh")
    mid = (x0 + x1) // 2
    rect(c, mid - 1, y0 + 1, mid + 2, y0 + 7, "tape")
    for k in range(5):  # the side flaps, folded open
        rect(c, x0 - 1 - k, y0 - k, x0 + 4 - k, y0 - k, "box_ol")
        rect(c, x1 - 4 + k, y0 - k, x1 + 1 + k, y0 - k, "box_ol")
        if 0 < k < 4:
            rect(c, x0 - k, y0 - k, x0 + 3 - k, y0 - k, "box_hi" if k == 1 else "box")
            rect(c, x1 - 3 + k, y0 - k, x1 + k, y0 - k, "box_hi" if k == 1 else "box")


def render(t, route, wall_px):
    cats, eaten, grown, flash, n = route
    c = np.full((H, W), C["bg"], np.uint8)
    flashing = flash <= t < flash + FLASHES * 2 * FLASH and (t - flash) // FLASH % 2 == 0
    c[wall_px] = C["flash" if flashing else "wall"]
    draw_rewards(c, eaten, t)
    draw_food(c, eaten, grown, t)
    draw_box_back(c)
    for poses in cats:
        draw_cat(c, *poses[t])
    draw_box_front(c)
    return c


# ---------------------------------------------------------------- output

def blend(a, b, f):
    return bytes(round(x + (y - x) * f) for x, y in zip(bytes.fromhex(a), bytes.fromhex(b))).hex()


def palette(theme):
    cols = THEMES[theme]
    cols = [cols[k] for k in THEMES["light"]] + [blend(cols[k], cols["bg"], 0.55) for k in DIMMED]
    return [v for h in cols + [h for _, h in SHARED] for v in bytes.fromhex(h)]


def to_image(frame, theme, scale=SCALE):
    big = frame.repeat(scale, axis=0).repeat(scale, axis=1)
    im = Image.fromarray(big, "P")
    im.putpalette(palette(theme))
    return im


def poster_frame(route):
    """Start the GIF a third of the way round, both cats running, so a paused GIF still shows them."""
    cats, *_, n = route
    t = HOP + (n - HOP) // 3
    while not all(poses[t][2] in SIDE for poses in cats):
        t += 1
    return t


def write_gifs(out_dir):
    route = plan()
    wall_px = walls()
    n = route[-1]
    start = poster_frame(route)
    frames = [render((start + i) % n, route, wall_px) for i in range(n)]
    for theme in THEMES:
        ims = [to_image(f, theme) for f in frames]
        ims[0].save(out_dir / f"catmaze-{theme}.gif", save_all=True, append_images=ims[1:],
                    duration=1000 // FPS, loop=0, optimize=True, disposal=1)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "assets")
    args = ap.parse_args()
    n = write_gifs(args.out)
    print(f"{n} frames, {n / FPS:.1f} s loop")


if __name__ == "__main__":
    main()
