import os

# ================================================================
#  Render-Ready Configuration (Environment Variables)
# ================================================================
DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]                      # from Render env
GUILD_ID = int(os.environ.get("GUILD_ID", "1270144230525763697"))
WELCOME_CHANNEL_ID = int(os.environ.get("WELCOME_CHANNEL_ID", "1418651393480327219"))
BACKGROUND_SRC = os.environ.get("BACKGROUND_SRC", "https://i.ibb.co/5xMZdycp/ttttt-2.jpg")
CIRCLE_MODE = os.environ.get("CIRCLE_MODE", "auto")
MANUAL_CIRCLE_LEFT = int(os.environ.get("MANUAL_CIRCLE_LEFT", "620"))
MANUAL_CIRCLE_TOP = int(os.environ.get("MANUAL_CIRCLE_TOP", "170"))
MANUAL_CIRCLE_DIAMETER = int(os.environ.get("MANUAL_CIRCLE_DIAMETER", "300"))
WELCOME_MESSAGE = os.environ.get(
    "WELCOME_MESSAGE",
    "@USERNAME ist jetzt Teil der German Voice World 🔥\n"
    "Du bist Member #PLACE und wir freuen uns riesig, dass du jetzt dabei bist"
)

# ================================================================
#  Web health (Render) — bind a port so Render doesn't kill the app
# ================================================================
from fastapi import FastAPI
import uvicorn

api = FastAPI()

@api.get("/")
async def root():
    return {"ok": True}

@api.get("/health")
async def health():
    return {"ok": True}

# -------------------
# CODE (do not edit)
# -------------------
import io
import asyncio
from typing import Optional, Tuple, Iterable

import aiohttp
import discord
from discord.ext import commands

# Pillow imports + resampling compatibility (Pillow 9/10/11)
try:
    from PIL import Image, ImageOps, ImageDraw, ImageStat
    from PIL.Image import Resampling as _Resampling
    RESAMPLE_LANCZOS = _Resampling.LANCZOS
except Exception:  # older Pillow
    from PIL import Image, ImageOps, ImageDraw, ImageStat
    RESAMPLE_LANCZOS = Image.LANCZOS

# ---------- bot + intents ----------
intents = discord.Intents.default()
intents.members = True  # required for on_member_join (also enable in Dev Portal)
intents.guilds = True
bot = commands.Bot(command_prefix="!", intents=intents)

_http: Optional[aiohttp.ClientSession] = None
_cached_bg_rgba: Optional[Image.Image] = None
_cached_circle: Optional[Tuple[int, int, int]] = None  # (left, top, diameter)


# ---------- http helpers ----------
async def ensure_http():
    global _http
    if _http is None or _http.closed:
        _http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))

async def fetch_bytes(url: str) -> Optional[bytes]:
    await ensure_http()
    try:
        async with _http.get(url, headers={"User-Agent": "gvw-welcome/1.2"}) as r:
            if r.status != 200:
                print(f"[http] {url} -> status {r.status}")
                return None
            return await r.read()
    except Exception as e:
        print(f"[http] {url} -> exception {e}")
        return None


# ---------- background loading & detection ----------
async def get_background_rgba() -> Optional[Image.Image]:
    """
    Load and cache background as RGBA. Supports URL or local path.
    """
    global _cached_bg_rgba
    if _cached_bg_rgba is not None:
        return _cached_bg_rgba

    try:
        if BACKGROUND_SRC.lower().startswith(("http://", "https://")):
            data = await fetch_bytes(BACKGROUND_SRC)
            if not data:
                return None
            img = Image.open(io.BytesIO(data)).convert("RGBA")
        else:
            img = Image.open(BACKGROUND_SRC).convert("RGBA")
        _cached_bg_rgba = img
        return img
    except Exception as e:
        print(f"[bg] load failed: {e}")
        return None


def _neighbors4(x: int, y: int, w: int, h: int) -> Iterable[Tuple[int, int]]:
    if x > 0: yield (x-1, y)
    if x+1 < w: yield (x+1, y)
    if y > 0: yield (x, y-1)
    if y+1 < h: yield (x, y+1)


def _auto_detect_circle_bbox(bg: Image.Image) -> Optional[Tuple[int, int, int]]:
    """
    Heuristic: find the largest bright connected component (likely the white circle),
    then return (left, top, diameter) in original pixels.
    """
    # Work on a small copy for speed
    max_w = 800
    scale = 1.0
    small = bg
    if bg.width > max_w:
        scale = bg.width / max_w
        new_h = int(round(bg.height / scale))
        small = bg.resize((max_w, new_h), RESAMPLE_LANCZOS)

    # Convert to grayscale, threshold on bright
    gray = small.convert("L")
    mean = ImageStat.Stat(gray).mean[0]
    cutoff = min(255, max(200, int(mean + 40)))  # push into bright range
    mask = gray.point(lambda p: 255 if p >= cutoff else 0, mode="1")

    # Connected-component search for the largest region
    pix = mask.load()
    w, h = mask.size
    visited = set()
    best_area = 0
    best_bbox = None

    for y in range(h):
        for x in range(w):
            if pix[x, y] != 255 or (x, y) in visited:
                continue
            # BFS
            stack = [(x, y)]
            visited.add((x, y))
            minx = maxx = x
            miny = maxy = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                if cx < minx: minx = cx
                if cx > maxx: maxx = cx
                if cy < miny: miny = cy
                if cy > maxy: maxy = cy
                for nx, ny in _neighbors4(cx, cy, w, h):
                    if (nx, ny) not in visited and pix[nx, ny] == 255:
                        visited.add((nx, ny))
                        stack.append((nx, ny))
            if area > best_area:
                best_area = area
                best_bbox = (minx, miny, maxx, maxy)

    if not best_bbox:
        return None

    minx, miny, maxx, maxy = best_bbox
    bw = maxx - minx + 1
    bh = maxy - miny + 1
    diameter_small = min(bw, bh)

    # Convert back to original coordinates
    if scale != 1.0:
        left = int(round(minx * scale))
        top = int(round(miny * scale))
        diameter = int(round(diameter_small * scale))
    else:
        left, top, diameter = minx, miny, diameter_small

    # Slight inner inset to avoid overlapping a dark outline
    pad = max(0, int(diameter * 0.02))
    left += pad
    top += pad
    diameter = max(10, diameter - 2 * pad)

    return (left, top, diameter)


def _resolve_circle_bbox(bg: Image.Image) -> Tuple[int, int, int]:
    """
    Decide where to paste the avatar:
      - If CIRCLE_MODE == "auto": try auto-detect once (cache). If it fails, use manual.
      - If CIRCLE_MODE == "manual": always use manual constants.
    """
    global _cached_circle
    if CIRCLE_MODE.lower() == "manual":
        return (MANUAL_CIRCLE_LEFT, MANUAL_CIRCLE_TOP, MANUAL_CIRCLE_DIAMETER)

    if _cached_circle is not None:
        return _cached_circle

    bbox = _auto_detect_circle_bbox(bg)
    if bbox is None:
        print("[detect] auto failed; falling back to manual coordinates.")
        bbox = (MANUAL_CIRCLE_LEFT, MANUAL_CIRCLE_TOP, MANUAL_CIRCLE_DIAMETER)
    else:
        print(f"[detect] auto-detected circle: left={bbox[0]} top={bbox[1]} d={bbox[2]}")
    _cached_circle = bbox
    return bbox


# ---------- image composition ----------
def compose_card(bg: Image.Image, avatar_bytes: Optional[bytes]) -> bytes:
    """
    Create the final PNG welcome card: circle-cropped avatar pasted onto bg.
    """
    left, top, diameter = _resolve_circle_bbox(bg)

    # Prepare avatar
    size = (diameter, diameter)
    if avatar_bytes:
        try:
            avatar = Image.open(io.BytesIO(avatar_bytes)).convert("RGBA")
        except Exception:
            avatar = Image.new("RGBA", size, (200, 200, 200, 255))
    else:
        avatar = Image.new("RGBA", size, (200, 200, 200, 255))

    avatar = ImageOps.fit(avatar, size, method=RESAMPLE_LANCZOS)

    # Circle mask (anti-aliased)
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((0, 0, size[0], size[1]), fill=255)

    av_circle = Image.new("RGBA", size)
    av_circle.paste(avatar, (0, 0), mask=mask)

    # Paste on a copy so we don't mutate cached background
    canvas = bg.copy()
    canvas.paste(av_circle, (left, top), mask=av_circle)

    out = io.BytesIO()
    canvas.save(out, format="PNG")
    out.seek(0)
    return out.read()


# ---------- message templating ----------
def fill_template(template: str, member: discord.Member) -> str:
    server = member.guild.name if member.guild else "Server"
    place_num = member.guild.member_count if member.guild else 0
    replacements = {
        "@USERNAME": member.mention,                # your exact token
        "#PLACE": f"#{place_num}",                  # your exact token
        "{SERVER}": server,                         # alternates for convenience
        "{USERNAME}": member.display_name,
        "{MENTION}": member.mention,
        "{PLACE}": str(place_num),
    }
    text = template
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


# ---------- events ----------
@bot.event
async def on_ready():
    print(f"[ready] Logged in as {bot.user} (ID: {bot.user.id})")
    try:
        g = bot.get_guild(GUILD_ID)
        if g:
            print(f"[ready] Guild resolved: {g.name} ({g.id}) • members: {g.member_count}")
    except Exception:
        pass
    await bot.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="Willkommen"))


@bot.event
async def on_member_join(member: discord.Member):
    # resolve channel
    g = member.guild
    if not g:
        return
    channel: Optional[discord.abc.Messageable] = g.get_channel(WELCOME_CHANNEL_ID)
    if channel is None:
        try:
            channel = await bot.fetch_channel(WELCOME_CHANNEL_ID)
        except Exception:
            channel = None
    if channel is None or not hasattr(channel, "send"):
        print(f"[welcome] channel {WELCOME_CHANNEL_ID} not found or not sendable")
        return

    # load background
    bg = await get_background_rgba()
    if bg is None:
        await channel.send("Willkommen! (Hinweis: Hintergrundbild konnte nicht geladen werden.)")
        return

    # fetch avatar
    try:
        avatar_url = member.display_avatar.replace(size=512, static_format="png").url
    except Exception:
        avatar_url = None
    avatar_bytes = await fetch_bytes(avatar_url) if avatar_url else None

    # compose card
    png = compose_card(bg, avatar_bytes)

    # message
    msg = fill_template(WELCOME_MESSAGE, member)

    # send
    try:
        await channel.send(content=msg, file=discord.File(io.BytesIO(png), filename="welcome.png"))
    except Exception as e:
        print(f"[welcome] send failed: {e}")


# ---------- graceful shutdown ----------
async def _graceful_close():
    global _http
    try:
        if _http and not _http.closed:
            await _http.close()
    except Exception:
        pass

@bot.event
async def on_disconnect():
    # Discord may reconnect; avoid closing session here to keep it available.
    pass


# ---------- run both: uvicorn (web) + discord bot ----------
async def _run_web():
    port = int(os.environ.get("PORT", "8000"))
    config = uvicorn.Config(api, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

async def _keep_render_awake():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    if not url.endswith("/"):
        url = url + "/"
    endpoint = url + "health"

    # reuse the global client if available
    while True:
        try:
            await ensure_http()
            assert _http is not None
            async with _http.get(endpoint, headers={"User-Agent": "render-keepawake/1.0"}) as r:
                _ = r.status  # touch to ensure request executes
        except Exception:
            pass
        await asyncio.sleep(240)

async def main():
    try:
        await asyncio.gather(
            _run_web(),
            _keep_render_awake(),
            bot.start(DISCORD_TOKEN),
        )
    finally:
        await _graceful_close()

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("Please set DISCORD_TOKEN in the environment.")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
