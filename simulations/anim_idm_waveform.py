"""IDM animation with audio-visualiser style waveform strip at the bottom."""

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
            gaps=np.empty(nv); dv=np.empty(nv)
            for i in range(nv):
                l=(i+1)%nv; gaps[i]=(pos[l]-pos[i])%L-vl; dv[i]=vel[i]-vel[l]
            gaps=np.maximum(gaps,0.1)
            ss=np.maximum(s0+vel*Th+vel*dv/(2*np.sqrt(a*b)),s0)
            acc=a*(1-(vel/v0)**d-(ss/gaps)**2)
            vel=np.maximum(vel+acc*dt,0.0); pos=(pos+vel*dt)%L; t+=dt
        ph.append(pos.copy()); vh.append(vel.copy()); th.append(t)
    return np.array(ph),np.array(vh),np.array(th)

def main():
    nf=200; nv=25; L=250.0; vm=12.0
    print("Simulating IDM with waveform...")
    positions,velocities,times=run_idm(nv,L,80.0,0.02,vm,nf)

    Ro,Ri=1.0,0.78; Rm=(Ro+Ri)/2

    fig=plt.figure(figsize=(7,8),facecolor=DARK_BG)
    ax_ring=fig.add_axes([0.05,0.30,0.90,0.65])
    ax_wave=fig.add_axes([0.08,0.06,0.86,0.18])
    ax_ring.set_facecolor(DARK_BG); ax_wave.set_facecolor(DARK_BG)
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

    ax_ring.text(0,1.35,"IDM: Traffic Heartbeat",ha="center",fontsize=15,
                 fontweight="bold",color=TEXT_COLOR,zorder=10)
    time_text=ax_ring.text(0,-1.35,"",ha="center",fontsize=10,
                           color=TEXT_COLOR,fontfamily="monospace",zorder=10)

    # Waveform
    ax_wave.set_xlim(0,nf); ax_wave.set_ylim(-0.5,vm+1)
    ax_wave.tick_params(colors="#565a7a",labelsize=6)
    for s in ax_wave.spines.values(): s.set_color("#2a2d3a")
    ax_wave.set_xlabel("Frame",color="#565a7a",fontsize=7)

    # Pre-compute per-vehicle speed spread
    mean_v=velocities.mean(axis=1)
    min_v=velocities.min(axis=1)
    max_v=velocities.max(axis=1)

    wave_line,=ax_wave.plot([],[],color=ACCENT,lw=1.5)
    wave_fill=None
    # Bar-style waveform: thin vertical bars for each frame
    wave_bars=[]
    for i in range(nf):
        b,=ax_wave.plot([i,i],[0,0],color=ACCENT,lw=1,alpha=0.4)
        wave_bars.append(b)

    cursor=ax_wave.axvline(x=0,color="#7aa2f7",lw=1.5,ls="-")

    def update(frame):
        p=positions[frame]; v=velocities[frame]; t=times[frame]
        for i in range(nv):
            angle=2*np.pi*p[i]/L; cx,cy=np.cos(angle)*Rm,np.sin(angle)*Rm
            rd=np.degrees(angle)-90
            tr=(mtransforms.Affine2D().translate(-0.0275,-0.0125)
                .rotate_deg(rd).translate(cx,cy)+ax_ring.transData)
            cars[i].set_transform(tr)
            cars[i].set_facecolor(speed_to_color(v[i],vm))

        # Update waveform
        wave_line.set_data(range(frame+1),mean_v[:frame+1])
        # Update bars up to current frame
        for i in range(min(frame+1,nf)):
            wave_bars[i].set_data([i,i],[min_v[i],max_v[i]])
            wave_bars[i].set_alpha(0.3)
        cursor.set_xdata([frame,frame])

        time_text.set_text(f"t = {t:5.1f} s  |  v\u0304 = {mean_v[frame]:.1f} m/s")
        return cars+[wave_line,cursor,time_text]+wave_bars[:frame+1]

    anim=FuncAnimation(fig,update,frames=nf,interval=40,blit=False)
    sp=os.path.join(RESULTS_DIR,"anim_idm_waveform.gif")
    print(f"Saving {sp} ({nf} frames)...")
    anim.save(sp,writer="pillow",fps=25,dpi=90,savefig_kwargs={"facecolor":DARK_BG})
    plt.close(); print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")

if __name__=="__main__":
    os.makedirs(RESULTS_DIR,exist_ok=True); main()
