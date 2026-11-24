#!/usr/bin/env python3
# Renders the benchmark charts of .github/README.md as static SVG, one light and one dark file per chart
# (the README picks one with <picture> / prefers-color-scheme). Edit the numbers below and rerun.
import os

S7150 = 'S7150 x4'
P100 = 'P100 x2'
DEPTHS = ['empty context', '4K context', '16K context']
PROMPT = {S7150: [419, 354, 229], P100: [437, 565, 377]}
DECODE = {S7150: [50.7, 46.0, 39.6], P100: [64.2, 64.0, 58.1]}
COMPUTE_RATIO = 81      # fp32 TFLOPS, 4 x 3.77 vs 2 x 9.3
BANDWIDTH_RATIO = 44    # GB/s, 4 x 160 vs 2 x 732

THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9',
                  base='#c3c2b7', s1='#2a78d6', s2='#eb6834'),
    'dark': dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a',
                 base='#383835', s1='#3987e5', s2='#d95926'),
}
FONT = 'system-ui, -apple-system, Segoe UI, Helvetica, Arial, sans-serif'
W, H = 720, 360
L, R, T, B = 56, 20, 78, 46   # plot margins


def bar(x, y, w, h, fill, r=4):
    # rounded data end (top), square at the baseline
    if h <= 0:
        return ''
    r = min(r, w / 2, h)
    return (f'<path d="M{x:.1f},{y + h:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + w - r:.1f} '
            f'Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y + h:.1f} Z" fill="{fill}"/>')


def frame(t, title, subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}" role="img" aria-label="{title}">',
            f'<rect width="{W}" height="{H}" rx="8" fill="{t["surface"]}"/>',
            f'<text x="{L - 36}" y="30" font-size="17" font-weight="600" fill="{t["ink"]}">{title}</text>',
            f'<text x="{L - 36}" y="52" font-size="13" fill="{t["ink2"]}">{subtitle}</text>']


def axis(t, vmax, step, fmt):
    out = []
    ph = H - T - B
    v = 0
    while v <= vmax + 1e-9:
        y = T + ph - v / vmax * ph
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{t["grid"] if v else t["base"]}" stroke-width="1"/>')
        out.append(f'<text x="{L - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}">{fmt(v)}</text>')
        v += step
    return out


def grouped(t, title, subtitle, data, vmax, step, fmt, vfmt):
    out = frame(t, title, subtitle)
    # legend, top right
    lx = W - R - 190
    for i, (name, col) in enumerate([(S7150, t['s1']), (P100, t['s2'])]):
        out.append(f'<rect x="{lx + i * 98}" y="20" width="12" height="12" rx="3" fill="{col}"/>')
        out.append(f'<text x="{lx + i * 98 + 18}" y="30" font-size="13" fill="{t["ink2"]}">{name}</text>')
    out += axis(t, vmax, step, fmt)
    ph = H - T - B
    gw = (W - L - R) / len(DEPTHS)
    bw = 64
    for g, label in enumerate(DEPTHS):
        cx = L + gw * g + gw / 2
        for i, (name, col) in enumerate([(S7150, t['s1']), (P100, t['s2'])]):
            v = data[name][g]
            x = cx - bw - 1 + i * (bw + 2)      # 2px surface gap between the pair
            h = v / vmax * ph
            out.append(bar(x, T + ph - h, bw, h, col))
            out.append(f'<text x="{x + bw / 2:.1f}" y="{T + ph - h - 7:.1f}" font-size="12" text-anchor="middle" '
                       f'fill="{t["ink2"]}">{vfmt(v)}</text>')
        out.append(f'<text x="{cx:.1f}" y="{H - B + 22}" font-size="12" text-anchor="middle" fill="{t["ink2"]}">{label}</text>')
    out.append('</svg>')
    return '\n'.join(out)


def ratio_chart(t):
    title = 'The S7150 rig as a share of the P100 pair'
    subtitle = 'Measured throughput ratio per test vs. the raw hardware ratio (dashed): above the line beats the spec sheet'
    out = frame(t, title, subtitle)
    vmax = 100
    out += axis(t, vmax, 20, lambda v: f'{v:.0f}%')
    ph = H - T - B
    groups = [('Prompt processing', [PROMPT[S7150][i] / PROMPT[P100][i] * 100 for i in range(3)], COMPUTE_RATIO,
               f'fp32 compute ratio {COMPUTE_RATIO}%'),
              ('Token generation', [DECODE[S7150][i] / DECODE[P100][i] * 100 for i in range(3)], BANDWIDTH_RATIO,
               f'memory bandwidth ratio {BANDWIDTH_RATIO}%')]
    short = ['empty', '4K', '16K']
    half = (W - L - R) / 2
    bw = 52
    for g, (gname, vals, ref, refname) in enumerate(groups):
        x0 = L + half * g
        xs = [x0 + half * (k + 0.5) / 3 - bw / 2 for k in range(3)]
        for k, v in enumerate(vals):
            h = v / vmax * ph
            out.append(bar(xs[k], T + ph - h, bw, h, t['s1']))
            out.append(f'<text x="{xs[k] + bw / 2:.1f}" y="{T + ph - h - 7:.1f}" font-size="12" text-anchor="middle" '
                       f'fill="{t["ink"]}" font-weight="600">{v:.0f}%</text>')
            out.append(f'<text x="{xs[k] + bw / 2:.1f}" y="{H - B + 18}" font-size="11" text-anchor="middle" fill="{t["muted"]}">{short[k]}</text>')
        yr = T + ph - ref / vmax * ph
        out.append(f'<line x1="{xs[0] - 12:.1f}" x2="{xs[2] + bw + 12:.1f}" y1="{yr:.1f}" y2="{yr:.1f}" stroke="{t["ink2"]}" '
                   f'stroke-width="2" stroke-dasharray="6 4"/>')
        # reference label sits on a surface chip so the bars behind it don't cut through the text
        lw = 7.4 * len(refname) + 12
        rx = xs[2] + bw + 12 - lw           # right-aligned with the end of the line
        out.append(f'<rect x="{rx:.1f}" y="{yr - 22:.1f}" width="{lw:.1f}" height="18" rx="3" fill="{t["surface"]}" opacity="0.92"/>')
        out.append(f'<text x="{rx + lw - 6:.1f}" y="{yr - 8:.1f}" font-size="12" text-anchor="end" fill="{t["ink2"]}">{refname}</text>')
        out.append(f'<text x="{x0 + half / 2:.1f}" y="{H - B + 36}" font-size="12" text-anchor="middle" fill="{t["ink2"]}">{gname}</text>')
    out.append('</svg>')
    return '\n'.join(out)


here = os.path.dirname(os.path.abspath(__file__))
for mode, t in THEMES.items():
    charts = {
        'prompt': grouped(t, 'Prompt processing', 'pp512, tokens/s (higher is better)', PROMPT, 600, 100, lambda v: f'{v:.0f}', lambda v: f'{v:.0f}'),
        'decode': grouped(t, 'Token generation', 'tg128, tokens/s (higher is better)', DECODE, 70, 10,
                          lambda v: f'{v:.0f}', lambda v: f'{v:.1f}'),
        'ratio': ratio_chart(t),
    }
    for name, svg in charts.items():
        with open(os.path.join(here, f'{name}-{mode}.svg'), 'w') as f:
            f.write(svg + '\n')
print('ok')
