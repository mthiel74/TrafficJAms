"""IDM animation with cumulative Vehicle-Hours-Lost (VHL) counter."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DARK_BG = "#0f1117"; ROAD_COLOR = "#2a2d3a"; ROAD_EDGE = "#3d4055"
TEXT_COLOR = "#c0caf5"; ACCENT = "#f7768e"

def speed_to_color(v, vm):
    t = np.clip(v/vm,0,1)
    if t<0.5: return (1.0,t*2,0.15)
    return (1.0-(t-0.5)*2,1.0,0.15+(t-0.5)*0.3)

def run_idm(nv,L,T,dt,v0,nf):
    vl=5.0; spf=max(1,int(T/dt/nf)); sp=L/nv
    pos=np.array([i*sp for i in range(nv)],dtype=float)
    vel=np.ones(nv)*v0*0.95; vel[0]=0.0; vel[1]=v0*0.3
    s0,Th,a,b,d=2.0,1.0,1.0,1.5,4
    ph,vh,th=[pos.copy()],[vel.copy()],[0.0]; t=0.0
    for _ in range(nf-1):
        for __ in range(spf):
            gaps=np.empty(nv); dv_=np.empty(nv)
            for i in range(nv):
                l=(i+1)%nv; gaps[i]=(pos[l]-pos[i])%L-vl; dv_[i]=vel[i]-vel[l]
            gaps=np.maximum(gaps,0.1)
            ss=np.maximum(s0+vel*Th+vel*dv_/(2*np.sqrt(a*b)),s0)
            acc=a*(1-(vel/v0)**d-(ss/gaps)**2)
            vel=np.maximum(vel+acc*dt,0.0); pos=(pos+vel*dt)%L; t+=dt
        ph.append(pos.copy()); vh.append(vel.copy()); th.append(t)
    return np.array(ph),np.array(vh),np.array(th)

def main():
    nf=200; nv=25; L=250.0; vm=12.0
    print("Simulating IDM with VHL counter...")
    positions,velocities,times=run_idm(nv,L,80.0,0.02,vm,nf)

    # Compute VHL: hours lost = sum over vehicles of (1 - v/v0) * dt
    dt_frames = np.diff(times)
    vhl_cum = np.zeros(nf)
    for f in range(1, nf):
        # Each vehicle loses (v0 - v)/v0 fraction of travel time
        lost = np.sum(np.maximum(vm - velocities[f], 0)) / vm * dt_frames[f-1] / 3600
        vhl_cum[f] = vhl_cum[f-1] + lost

    Ro,Ri=1.0,0.78; Rm=(Ro+Ri)/2

    fig=plt.figure(figsize=(7,7.5),facecolor=DARK_BG)
    ax_ring=fig.add_axes([0.05,0.28,0.90,0.65])
    ax_vhl=fig.add_axes([0.12,0.06,0.80,0.17])
    ax_ring.set_facecolor(DARK_BG); ax_vhl.set_facecolor(DARK_BG)
    ax_ring.set_xlim(-1.5,1.5); ax_ring.set_ylim(-1.5,1.5)
    ax_ring.set_aspect("equal"); ax_ring.axis("off")

    # Road
    th=np.linspace(0,2*np.pi,200)
    ax_ring.fill_between(np.cos(th)*Ro,np.sin(th)*Ro,np.cos(th)*Ri*0.99,color=ROAD_COLOR,zorder=1)
    ax_ring.add_patch(plt.Circle((0,0),Ro,fill=False,color=ROAD_EDGE,lw=2.5,zorder=2))
    ax_ring.add_patch(plt.Circle((0,0),Ri,fill=False,color=ROAD_EDGE,lw=2.5,zorder=2))
    for i in range(60):
        a1=i*2*np.pi/60; t_=np.linspace(a1,a1+np.pi/60*0.6,10)
        ax_ring.plot(np.cos(t_)*Rm,np.sin(t_)*Rm,color="#4a4d60",lw=1,zorder=2)

    cars=[]
    for i in range(nv):
        b=FancyBboxPatch((0,0),0.055,0.025,boxstyle="round,pad=0.005",
                          facecolor="#9ece6a",edgecolor="none",zorder=5)
        ax_ring.add_patch(b); cars.append(b)

    ax_ring.text(0,1.35,"IDM: Vehicle-Hours Lost",ha="center",fontsize=15,
                 fontweight="bold",color=TEXT_COLOR,zorder=10)
    time_text=ax_ring.text(0,-1.35,"",ha="center",fontsize=10,
                           color=TEXT_COLOR,fontfamily="monospace",zorder=10)

    # VHL counter in centre
    vhl_text=ax_ring.text(0,0,"",ha="center",va="center",fontsize=22,
                          fontweight="bold",color=ACCENT,fontfamily="monospace",zorder=10)
    ax_ring.text(0,-0.15,"veh-hrs lost",ha="center",fontsize=8,color="#565a7a",zorder=10)

    # VHL cumulative plot
    ax_vhl.set_xlim(0,nf); ax_vhl.set_ylim(0,vhl_cum[-1]*1.1)
    ax_vhl.tick_params(colors="#565a7a",labelsize=6)
    for s in ax_vhl.spines.values(): s.set_color("#2a2d3a")
    ax_vhl.set_xlabel("Frame",color="#565a7a",fontsize=7)
    ax_vhl.set_ylabel("VHL",color="#565a7a",fontsize=7)
    ax_vhl.fill_between(range(nf),vhl_cum,color=ACCENT,alpha=0.15)
    vhl_line,=ax_vhl.plot([],[],color=ACCENT,lw=2)
    cursor=ax_vhl.axvline(x=0,color="#7aa2f7",lw=1.5)

    def update(frame):
        p=positions[frame]; v=velocities[frame]; t=times[frame]
        for i in range(nv):
            angle=2*np.pi*p[i]/L; cx,cy=np.cos(angle)*Rm,np.sin(angle)*Rm
            rd=np.degrees(angle)-90
            tr=(mtransforms.Affine2D().translate(-0.0275,-0.0125)
                .rotate_deg(rd).translate(cx,cy)+ax_ring.transData)
            cars[i].set_transform(tr)
            cars[i].set_facecolor(speed_to_color(v[i],vm))

        vhl_text.set_text(f"{vhl_cum[frame]:.3f}")
        vhl_line.set_data(range(frame+1),vhl_cum[:frame+1])
        cursor.set_xdata([frame,frame])
        time_text.set_text(f"t = {t:5.1f} s  |  v\u0304 = {v.mean():.1f} m/s")
        return cars+[vhl_text,vhl_line,cursor,time_text]

    anim=FuncAnimation(fig,update,frames=nf,interval=40,blit=False)
    sp=os.path.join(RESULTS_DIR,"anim_idm_vhl.gif")
    print(f"Saving {sp} ({nf} frames)...")
    anim.save(sp,writer="pillow",fps=25,dpi=90,savefig_kwargs={"facecolor":DARK_BG})
    plt.close(); print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")

if __name__=="__main__":
    os.makedirs(RESULTS_DIR,exist_ok=True); main()
