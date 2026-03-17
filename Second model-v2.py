import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from ultralytics import YOLO
import time

tc = 90
lt = 10
eg = tc - lt
gap = 2
minG, maxG = 10, 45
sat = 2.0 

rds = {
    "mainS": {"q": 0, "w": 1.0, "last": 0, "arr": 0},
    "mainL": {"q": 0, "w": 1.5, "last": 0, "arr": 0},
    "mainR": {"q": 0, "w": 0.8, "last": 0, "arr": 0},
    "sideS": {"q": 0, "w": 1.0, "last": 0, "arr": 0},
    "sideL": {"q": 0, "w": 1.5, "last": 0, "arr": 0},
    "sideR": {"q": 0, "w": 0.8, "last": 0, "arr": 0}
}

gMap = {"mainL": 20, "mainS": 25, "sideL": 20, "sideS": 25}
ph = 1
idx = 1
new = True
ts = 0
start = time.time()
b1 = np.zeros((480, 640, 3), dtype=np.uint8)
b2 = np.zeros((480, 640, 3), dtype=np.uint8)
net = YOLO('yolov8n.pt')

def calc():
    global gMap, new
    print(f"Update Cycle {idx}")
    pv = {}
    for k in rds:
        pv[k] = rds[k]["q"] + (rds[k]["w"] * rds[k]["arr"])
    
    mp = max(0.1, pv["mainL"] + pv["mainS"] + pv["mainR"])
    sp = max(0.1, pv["sideL"] + pv["sideS"] + pv["sideR"])
    tp = mp + sp
    
    mG = np.clip((mp / tp) * eg, minG*2, maxG*2)
    sG = eg - mG
    
    mlr = pv["mainL"] / mp
    mLg = np.clip(mlr * mG, minG, maxG)
    mSg = mG - mLg
    
    slr = pv["sideL"] / sp
    sLg = np.clip(slr * sG, minG, maxG)
    sSg = sG - sLg
    
    gMap = {"mainL": round(mLg, 1), "mainS": round(mSg, 1), 
            "sideL": round(sLg, 1), "sideS": round(sSg, 1)}
    new = False
    print(f"Plan: {gMap}")

fig = plt.figure(figsize=(14, 7))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0,0]); ax1.axis("off"); d1 = ax1.imshow(b1)
ax2 = fig.add_subplot(gs[1,0]); ax2.axis("off"); d2 = ax2.imshow(b2)
pnl = fig.add_subplot(gs[:,1]); pnl.axis("off"); pnl.set_xlim(0, 10); pnl.set_ylim(0, 10)

t1 = pnl.text(1, 8.5, "", fontsize=11, weight='bold')
t2 = pnl.text(1, 7.5, "", fontsize=11, weight='bold')
st = pnl.text(5, 5, "", ha='center', color='blue', weight='bold', fontsize=12)
qt = pnl.text(1, 1, "", fontsize=10, family='monospace', bbox=dict(facecolor='white', alpha=0.5))

def detect(img, tag, isGrnList):
    res = net(img, verbose=False, conf=0.1, classes=[67, 73])[0]
    w = img.shape[1]
    cnt = {f"{tag}L": 0, f"{tag}S": 0, f"{tag}R": 0}
    
    for b in res.boxes:
        cx = (b.xyxy[0][0] + b.xyxy[0][2]) / 2
        if cx < w/3: cnt[f"{tag}R"] += 1
        elif cx < 2*w/3: cnt[f"{tag}S"] += 1
        else: cnt[f"{tag}L"] += 1
        
    for k, v in cnt.items():
        rds[k]["arr"] = max(0, v - rds[k]["last"]) / gap
        # Update queue ONLY if that specific lane is NOT green
        if k not in isGrnList:
            rds[k]["q"] = max(rds[k]["q"], v)
        rds[k]["last"] = v
        
    if len(res.boxes) > 0: print(f"Sight {tag}: {len(res.boxes)}")
    return res.plot()

def run(f):
    global ph, idx, new, ts, start, b1, b2
    
    ok1, r1 = c1.read()
    ok2, r2 = c2.read()
    if not ok1 or not ok2: return d1, d2

    now = time.time()
    
    grn = {1: ["mainL"], 2: ["mainS", "mainR"], 3: ["sideL"], 4: ["sideS", "sideR"]}.get(ph, [])

    if now - ts >= gap:
        b1 = detect(r1, "main", grn)
        b2 = detect(r2, "side", grn)
        ts = now

    if new: calc()

    keys = {1: "mainL", 2: "mainS", 3: "sideL", 4: "sideS"}
    k = keys[ph]
    limit = gMap[k]
    dt = now - start
    rem = max(0, limit - dt)

    if dt >= limit:
        ph = ph + 1 if ph < 4 else 1
        start = now
        if ph == 1:
            new = True
            idx += 1

    if int(now) != int(now - 0.033):
        for r in grn:
            if rds[r]["q"] > 0:
                rds[r]["q"] = max(0, rds[r]["q"] - sat)
            if rds[r]["last"] == 0: rds[r]["q"] = 0

    show = (now - ts < 1.3)
    d1.set_data(cv2.cvtColor(b1 if show else r1, cv2.COLOR_BGR2RGB))
    d2.set_data(cv2.cvtColor(b2 if show else r2, cv2.COLOR_BGR2RGB))
    
    t1.set_text(f"NORTH | {k if ph<=2 else 'STOP'} | {rem:.1f}s")
    t2.set_text(f"WEST  | {k if ph>=3 else 'STOP'} | {rem:.1f}s")
    st.set_text(f"CYCLE {idx} | PHASE {ph}")
    
    inf = f"Queues:\n"
    inf += f"M -> L:{int(rds['mainL']['q'])} S:{int(rds['mainS']['q'])} R:{int(rds['mainR']['q'])}\n"
    inf += f"S -> L:{int(rds['sideL']['q'])} S:{int(rds['sideS']['q'])} R:{int(rds['sideR']['q'])}"
    qt.set_text(inf)

    return d1, d2, t1, t2, st, qt

if __name__ == "__main__":
    c1, c2 = cv2.VideoCapture(0), cv2.VideoCapture(1)
    ani = animation.FuncAnimation(fig, run, interval=30, blit=False)
    plt.show()
    c1.release(); c2.release(); cv2.destroyAllWindows()