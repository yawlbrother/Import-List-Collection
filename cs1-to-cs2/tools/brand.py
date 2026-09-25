"""Operator identities painted over the CS1 asset's branding on the body atlas.

A brand is a mark (drawn here), a wordmark, a subtitle, a fleet code and a colour. `apply` clears the
original operator's crests and lettering (regions measured on the X40 atlas, scaled to the atlas in
hand) and paints the new identity into the BaseColor slot; the mark and the wordmark also go into the
Emissive texture with the light index of the windows, so they glow in the brand colour at night."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONTS = dict(bold=('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
             regular=('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
             mono=('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', '/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf'))

def font(kind, size):
    for path in FONTS[kind]:
        try:
            return ImageFont.truetype(path, int(size))
        except OSError:
            continue
    return ImageFont.load_default()

# SJ branding on the 2048 x 1024 X40 atlas: (x0, y0, x1, y1); `at` = where the replacement goes
REGIONS = dict(band=(584, 256, 1366, 323), door_a=(387, 382, 439, 422), door_b=(1522, 382, 1574, 422),
               nose=(1930, 690, 2004, 748), number=(1032, 686, 1096, 716), plate=(6, 424, 190, 444), vtext=(1284, 508, 1310, 702))

def mark_helix(sz, col):
    """Hexagon with two crossing strands."""
    im = Image.new('RGBA', (sz, sz), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = sz / 2; r = sz * 0.46; w = sz * 0.13
    d.polygon([(c + r * math.cos(math.radians(60 * k + 30)), c + r * math.sin(math.radians(60 * k + 30))) for k in range(6)], outline=col, width=max(2, sz // 14))
    d.polygon([(c - r * 0.55, c - r * 0.5), (c - r * 0.55 + w, c - r * 0.5), (c + r * 0.55, c + r * 0.5), (c + r * 0.55 - w, c + r * 0.5)], fill=col)
    d.polygon([(c + r * 0.55, c - r * 0.5), (c + r * 0.55 - w, c - r * 0.5), (c - r * 0.55, c + r * 0.5), (c - r * 0.55 + w, c + r * 0.5)], fill=(255, 255, 255, 255))
    return im

def mark_vanta(sz, col):
    im = Image.new('RGBA', (sz, sz), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = sz / 2; w = sz * 0.16
    d.polygon([(sz * 0.05, sz * 0.12), (sz * 0.05 + w, sz * 0.12), (c, sz * 0.86), (c - w * 0.55, sz * 0.86)], fill=col)
    d.polygon([(sz * 0.95, sz * 0.12), (sz * 0.95 - w, sz * 0.12), (c, sz * 0.86), (c + w * 0.55, sz * 0.86)], fill=(255, 255, 255, 255))
    d.rectangle([sz * 0.30, sz * 0.02, sz * 0.70, sz * 0.02 + w * 0.5], fill=col)
    return im

def mark_nexus(sz, col):
    im = Image.new('RGBA', (sz, sz), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = sz / 2; r = sz * 0.44
    d.ellipse([c - r, c - r, c + r, c + r], outline=col, width=max(2, sz // 12))
    for k in range(3):
        a = math.radians(90 + 120 * k); x, y = c + r * math.cos(a), c - r * math.sin(a)
        d.line([(c, c), (x, y)], fill=(255, 255, 255, 255), width=max(2, sz // 12))
        d.ellipse([x - sz * 0.09, y - sz * 0.09, x + sz * 0.09, y + sz * 0.09], fill=col)
    d.ellipse([c - sz * 0.12, c - sz * 0.12, c + sz * 0.12, c + sz * 0.12], fill=(255, 255, 255, 255))
    return im

def mark_okada(sz, col):
    im = Image.new('RGBA', (sz, sz), (0, 0, 0, 0)); d = ImageDraw.Draw(im); c = sz / 2; r = sz * 0.47
    d.ellipse([c - r, c - r, c + r, c + r], fill=col)
    d.text((c, c + sz * 0.02), 'OV', font=font('bold', sz * 0.55), fill=(255, 255, 255, 255), anchor='mm')
    d.rectangle([c - r * 0.75, c + r * 0.55, c + r * 0.75, c + r * 0.62], fill=(255, 255, 255, 255))
    return im

MARKS = dict(helix=mark_helix, vanta=mark_vanta, nexus=mark_nexus, okada=mark_okada)
BRANDS = {
    # name/sub: the wordmark block left of the mark; right/right_sub: the block right of it
    'helix': dict(mark='helix', colour=(0, 225, 255), name='HELIX', sub='TRANSIT GROUP', right='HX-3737', right_sub='MOVING THE CITY', code='HLX', unit='HX-3737', title='Helix Transit X40'),
    'vanta': dict(mark='vanta', colour=(255, 186, 48), name='VANTA', sub='LINES', right='V 3737', right_sub='FIRST CLASS ONLY', code='VNT', unit='V 3737', title='Vanta Lines X40'),
    'nexus': dict(mark='nexus', colour=(255, 116, 0), name='NEXUS RAIL', sub='CITY TRANSIT AUTHORITY', right='NX 3737', right_sub='PUBLIC SERVICE', code='NXR', unit='NX 3737', title='Nexus Rail X40'),
    'okada': dict(mark='okada', colour=(222, 28, 60), name='OKADA-VANCE', sub='MOBILITY DIVISION', right='OV-3737', right_sub='SINCE 2041', code='OV', unit='OV-3737', title='Okada-Vance X40'),
}

def text_img(txt, kind, size, fill, tracking=0):
    f = font(kind, size); w = sum(f.getlength(c) for c in txt) + tracking * (len(txt) - 1)
    im = Image.new('RGBA', (int(w) + 4, int(size * 1.3) + 2), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 2
    for c in txt:
        d.text((x, 0), c, font=f, fill=fill); x += f.getlength(c) + tracking
    return im

def inpaint_rect(img, box):
    """Fills `box` of RGBA image `img` with a gradient interpolated from the ring of texels just outside
    it, so cleared crests and lettering leave no visible rectangle even on a shaded panel."""
    x0, y0, x1, y1 = box; a = np.asarray(img).astype(np.float32); H, W = a.shape[:2]
    x0, y0, x1, y1 = max(x0, 2), max(y0, 2), min(x1, W - 3), min(y1, H - 3)
    if x1 <= x0 or y1 <= y0:
        return
    L, R = a[y0:y1, x0 - 2, :3], a[y0:y1, x1 + 1, :3]; Tp, Bt = a[y0 - 2, x0:x1, :3], a[y1 + 1, x0:x1, :3]
    h, w = y1 - y0, x1 - x0; u = (np.arange(w) + 0.5) / w; t = (np.arange(h) + 0.5) / h
    horiz = L[:, None, :] * (1 - u)[None, :, None] + R[:, None, :] * u[None, :, None]
    vert = Tp[None, :, :] * (1 - t)[:, None, None] + Bt[None, :, :] * t[:, None, None]
    wh = 1.0 / np.minimum(np.arange(w) + 1, w - np.arange(w))[None, :, None]; wv = 1.0 / np.minimum(np.arange(h) + 1, h - np.arange(h))[:, None, None]
    a[y0:y1, x0:x1, :3] = (horiz * wh + vert * wv) / (wh + wv)
    img.paste(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), 'RGBA'))

def apply(slots, lay, brand, glow_index=1):
    """Paints `brand` (a BRANDS entry) over the SJ branding of the atlas in `slots` (slot -> (PIL image,
    srgb); Emissive included), scaled from the 2048 px layout to this atlas. Lit brand texels get
    `glow_index` in lay['index_map'] (1 = the windows' light: on at night)."""
    base = slots['BaseColor'][0]; emis = slots['Emissive'][0]; idx = lay['index_map']
    W, H = base.size; s = W / 2048.0
    b = base.convert('RGBA'); e = emis.convert('RGBA'); d = ImageDraw.Draw(b); de = ImageDraw.Draw(e)
    col = tuple(brand['colour']) + (255,); white = (255, 255, 255, 255); mark = MARKS[brand['mark']]
    def sc(box): return tuple(int(round(v * s)) for v in box)
    def clear(key, _tone=None):
        inpaint_rect(b, sc(REGIONS[key])); de.rectangle(sc(REGIONS[key]), fill=(0, 0, 0, 0))
        x0, y0, x1, y1 = sc(REGIONS[key]); idx[y0:y1 + 1, x0:x1 + 1] = 0
    def put(im, cx, cy, glow=False):
        pos = (int(round(cx * s - im.width / 2)), int(round(cy * s - im.height / 2))); b.alpha_composite(im, pos)
        if glow:
            g = np.asarray(im); lit = g[..., 3] > 96
            tex = np.zeros_like(g); tex[lit] = (*brand['colour'], 255)
            e.alpha_composite(Image.fromarray(tex), pos)
            x0, y0 = max(pos[0], 0), max(pos[1], 0)
            sub = lit[y0 - pos[1]:, x0 - pos[0]:][:H - y0, :W - x0]
            idx[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]][sub] = glow_index
    clear('band', 'band')
    if s >= 0.5:
        put(mark(int(60 * s), col), 975, 289, glow=True)
        for cx, big, small in ((740, brand['name'], brand['sub']), (1210, brand['right'], brand['right_sub'])):
            put(text_img(big, 'bold', 30 * s, white, tracking=5 * s), cx, 281, glow=True)
            put(text_img(small, 'regular', 10 * s, col, tracking=3 * s), cx, 307)
        for key, cx in (('door_a', 413), ('door_b', 1548)):
            clear(key, 'door'); put(mark(int(32 * s), col), cx, 402, glow=True)
        clear('nose', 'nose'); put(mark(int(46 * s), col).rotate(-18, resample=Image.BICUBIC, expand=True), 1967, 719)
        clear('number', 'plate'); put(text_img(brand['unit'], 'mono', 16 * s, white, tracking=s), 1064, 701)
        plate = text_img(f'{brand["code"]} 94 74 440 3737-0 X40', 'bold', 12 * s, (12, 12, 12, 255), tracking=s)
        clear('plate', 'plate'); put(plate, 98, 434)
        clear('vtext', 'plate'); put(plate.rotate(-90, expand=True), 1297, 605)
    else:   # the small LOD atlas: the crests become a dot of brand colour, lettering is below texel size
        for key, tone_key in (('door_a', 'door'), ('door_b', 'door'), ('nose', 'nose')):
            clear(key, tone_key)
        put(mark(max(4, int(60 * s)), col), 975, 289)
    base.paste(b.convert(base.mode)); emis.paste(e.convert(emis.mode))
