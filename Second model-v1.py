import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from ultralytics import YOLO

# global variable initialize
lanes={
    "mainS":{"queue":deque(),"saturation":0.5,"weight":1.0},
    "mainL":{"queue":deque(),"saturation":0.35,"weight":1.5},
    "mainR":{"queue":deque(),"saturation":0.4,"weight":0.8},
    "sideS":{"queue":deque(),"saturation":0.5,"weight":1.0},
    "sideL":{"queue":deque(),"saturation":0.35,"weight":1.5},
    "sideR":{"queue":deque(),"saturation":0.4,"weight":0.8}
}

cycle=180
lost=15
mingl=15
mings=20
maxgl=70
maxgs=70
ts=5

currentp=1
phase=0

prev={
    "mainS":0,
    "mainL":0,
    "mainR":0,
    "sideS":0,
    "sideL":0,
    "sideR":0
}

arrivalrate={
    "mainS":deque([0,0,0], maxlen=3),
    "mainL":deque([0,0,0], maxlen=3),
    "mainR":deque([0,0,0], maxlen=3),
    "sideS":deque([0,0,0], maxlen=3),
    "sideL":deque([0,0,0], maxlen=3),
    "sideR":deque([0,0,0], maxlen=3)
}

greentime={
    "mainL":0,
    "mainS":0,
    "sideL":0,
    "sideS":0
}

model=YOLO('CarDetection.pt')


#Car detection
def detect(cam,dir):
    ret,frame=cam.read()
    if not ret:
        e1=np.zeros((480, 640, 3), dtype=np.uint8)
        e2={f"{dir}L":0, f"{dir}S":0, f"{dir}R":0}
        return e1,e2
    r=model(frame,verbose=False,conf=0.3)[0]
    h, w = frame.shape[:2]

    lane={
        f"{dir}S": 0,
        f"{dir}L": 0,
        f"{dir}R": 0
    }

    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
        cx = (x1 + x2) / 2
        if cx < w // 3: lane[f"{dir}R"] += 1
        elif cx < 2 * w // 3: lane[f"{dir}S"] += 1
        else: lane[f"{dir}L"] += 1
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    return frame,lane

#pressure and green calculation
newcycle=True

def green(main,side):
    global prev,greentime,newcycle
    cur={**main,**side} # **=拆开字典并合并
    arrival={}
    newc={}
    for l in lanes:
        new=max(0,cur[l]-prev[l])
        newc[l]=new
        arr=new/ts
        arrivalrate[l].append(arr)
        if arrivalrate[l]: arrival[l]=np.mean(arrivalrate[l])
        else: arrival[l]=0
        lanes[l]["queue"].extend([1]*newc[l])
        if len(lanes[l]["queue"]) > 200:
            lanes[l]["queue"] = deque(list(lanes[l]["queue"])[-200:])

    prev=cur.copy()

    
    if newcycle:
        p={}
        for l in lanes:
            p[l]=len(lanes[l]["queue"])+ts*arrival[l]
        mainp=(p["mainS"]*lanes["mainS"]["weight"]+
               p["mainL"]*lanes["mainL"]["weight"]+
               p["mainR"]*lanes["mainR"]["weight"])
        sidep=(p["sideS"]*lanes["sideS"]["weight"]+
               p["sideL"]*lanes["sideL"]["weight"]+
               p["sideR"]*lanes["sideR"]["weight"])
        totalp=mainp+sidep or 1

        #available green per side
        green=cycle-lost
        maing=(mainp/totalp)*green
        maing=(maing//ts)*ts
        maing=np.clip(maing,(mingl+mings),(maxgl+maxgs))
        sideg=green-maing

        #available green per phase
        mainp=mainp or 1
        sidep=sidep or 1
        mainlg=(p["mainL"]*lanes["mainL"]["weight"]/mainp)*maing
        mainlg=(mainlg//ts)*ts
        mainlg=np.clip(mainlg,mingl,maxgl)
        mainsg=maing-mainlg
        sidelg=(p["sideL"]*lanes["sideL"]["weight"]/sidep)*sideg
        sidelg=(sidelg//ts)*ts
        sidelg=np.clip(sidelg,mingl,maxgl)
        sidesg=sideg-sidelg

        greentime={
            "mainL":mainlg,
            "mainS":mainsg,
            "sideL":sidelg,
            "sideS":sidesg
        }
        newcycle=False
        print(f"New green distribution: North left{mainlg:.1f}s | North straight+right{mainsg:.1f}s | West left{sidelg:.1f}s | West straight+right{sidesg:.1f}s")

    cl=[]
    if currentp==1: cl=["mainL"]
    elif currentp==2: cl=["mainS","mainR"]
    elif currentp==3: cl=["sideL"]
    elif currentp==4: cl=["sideS","sideR"]

    for l in cl:
        dep=int(lanes[l]["saturation"]*ts)
        dep=min(dep,len(lanes[l]["queue"]))
        for i in range(dep):
            if lanes[l]["queue"]: lanes[l]["queue"].popleft()
    
    return greentime,arrival

fig = plt.figure(figsize=(16,8))
gs = fig.add_gridspec(2,2)

# Camera display
mainscreen = fig.add_subplot(gs[0,0])
mainscreen.set_title("North Direction Camera", fontsize=10)
mainscreen.axis("off")
display1 = mainscreen.imshow(np.zeros((480,640,3),np.uint8))

sidescreen = fig.add_subplot(gs[1,0])
sidescreen.set_title("West Direction Camera", fontsize=10)
sidescreen.axis("off")
display2 = sidescreen.imshow(np.zeros((480,640,3),np.uint8))

light = fig.add_subplot(gs[:,1])
light.set_xlim(0, 14)
light.set_ylim(0, 9)
light.axis("off")

ml = light.text(1.5, 7, "North direction left turn light:0.0s", fontsize=10)
ms = light.text(1.5, 6, "North direction straight & right turn light:0.0s", fontsize=10)
sl = light.text(8.5, 7, "West direction left turn light:0.0s", fontsize=10)
ss = light.text(8.5, 6, "West direction straight & right turn light:0.0s", fontsize=10)
pt = light.text(5, 8, "Current phase:North direct left turn | Current Cycle:1", fontsize=12)
qt = light.text(5, 1, "Queue length:North left 0/North Straight 0/North right 0 | West left 0/West Straight 0/West right 0", fontsize=8)

cyclecount=1

def update(frame):
    global currentp,phase,newcycle,cyclecount
    ind=frame//(cycle//ts)+1
    f1,mainc=detect(cam1,"main")
    f2,sidec=detect(cam2,"side")
    g,ar=green(mainc,sidec)
    f1 = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
    f2 = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
    display1.set_data(f1)
    display2.set_data(f2)
    phase+=5
    if currentp==1:
        t=greentime["mainL"]-phase
        ml.set_text(f"North direction left turn light:{t:.1f}s")
        pt.set_text(f"Current phase:North direct left turn | Current Cycle:{cyclecount}")
        if(phase>=g["mainL"]):
            currentp=2
            phase=0
    elif currentp==2:
        t=greentime["mainS"]-phase
        ms.set_text(f"North direction straight & right turn light:{t:.1f}s")
        pt.set_text(f"Current phase:North direct straight & right turn | Current Cycle:{cyclecount}")
        if(phase>=g["mainS"]):
            currentp=3
            phase=0
    elif currentp==3:
        t=greentime["sideL"]-phase
        sl.set_text(f"West direction left turn light:{t:.1f}s")
        pt.set_text(f"Current phase:West direct left turn | Current Cycle:{cyclecount}")
        if(phase>=g["sideL"]):
            currentp=4
            phase=0
    elif currentp==4:
        t=greentime["sideS"]-phase
        ss.set_text(f"West direction straight & right turn light:{t:.1f}s")
        pt.set_text(f"Current phase:West direct straight & right turn | Current Cycle:{cyclecount}")
        if(phase>=g["sideS"]):
            currentp=1
            phase=0
            newcycle=True
            cyclecount+=1
    
    qt.set_text(
        f"Queue length:North left {len(lanes['mainL']['queue'])}/North Straight {len(lanes['mainS']['queue'])}/North right {len(lanes['mainR']['queue'])} | West left {len(lanes['sideL']['queue'])}/West Straight {len(lanes['sideS']['queue'])}/West right {len(lanes['sideR']['queue'])}"
    )

    return display1,display2,ml,ms,sl,ss,pt,qt
    
if __name__ == "__main__":
    # 0=north, 1=west
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)

    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cam1.isOpened() or not cam2.isOpened():
        print("Error: Camera not connected")
        exit()

    ani = animation.FuncAnimation(
        fig, update, interval=5000, blit=True, cache_frame_data=False
    )
    plt.show()

    cam1.release()
    cam2.release()
    cv2.destroyAllWindows()