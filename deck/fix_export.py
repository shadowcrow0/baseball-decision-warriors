"""修 PDF 匯出跑版：svg 給明確寬高、表頭字色放到 th、<b style=color> 改成 span+b、第 12 頁大數字縮小。"""
import re, sys, json
from pathlib import Path

slides = Path(sys.argv[1])
order = json.load(open(slides.parent / "deck.json"))["order"]
changed = []

def edit(name, fn):
    p = slides / f"{name}.html"
    old = p.read_text()
    new = fn(old)
    if new != old:
        p.write_text(new)
        changed.append(name)

# 1. method：svg 內容在 y 100–500，裁成 420 高、給寬高
def fix_method(s):
    s = s.replace('<svg viewBox="0 0 1664 600" ', '<svg width="1664" height="420" viewBox="0 0 1664 420" ', 1)
    s = s.replace(' style="width:100%;height:auto">', '>', 1)
    # 把 defs 之後的內容整個位移 -90
    i = s.index('</defs>') + len('</defs>')
    j = s.index('</svg>')
    s = s[:i] + '<g transform="translate(0 -90)">' + s[i:j] + '</g>' + s[j:]
    return s
edit("method", fix_method)

# 2. trend：svg 給寬高，左欄固定 1180px
def fix_trend(s):
    s = s.replace('<svg viewBox="0 0 1180 470" ', '<svg width="1180" height="470" viewBox="0 0 1180 470" ', 1)
    s = s.replace(' style="width:100%;height:auto">', '>', 1)
    s = s.replace('<div style="flex:2.2;display:flex;flex-direction:column;justify-content:center"><svg',
                  '<div style="flex:none;width:1180px;display:flex;flex-direction:column;justify-content:center"><svg', 1)
    return s
edit("trend", fix_trend)

# 3. 表頭：th 自己帶字色（tr 的 color 匯出時不會繼承）
def fix_th(s):
    return re.sub(r'<th style="(?![^"]*color:)', '<th style="color:#F8F8F5;', s)
for n in order:
    edit(n, fix_th)

# 4. <b style="color:X">…</b> → <span style="color:X"><b>…</b></span>
def fix_b(s):
    return re.sub(r'<b style="color:(#[0-9A-Fa-f]{6})">([^<]*)</b>', r'<span style="color:\1"><b>\2</b></span>', s)
for n in order:
    edit(n, fix_b)

# 5. 第 12 頁三個大數字 52px → 42px（卡片太窄會折行蓋到說明）
def fix_risk(s):
    return s.replace('<p style="font-size:52px;font-weight:700;color:#E0812B;line-height:1">', '<p style="font-size:42px;font-weight:700;color:#E0812B;line-height:1">')
edit("draft_risk", fix_risk)

print("changed:", " ".join(dict.fromkeys(changed)))
left = sum(len(re.findall(r'<b style="color', (slides / f"{n}.html").read_text())) for n in order)
print("remaining <b style=color>:", left)
