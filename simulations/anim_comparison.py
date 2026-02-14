"""Side-by-side IDM vs Bando on identical ring roads."""

import os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import matplotlib.transforms as mtransforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

DARK_BG = "#0f1117"; ROAD_COLOR = "#2a2d3a"; ROAD_EDGE = "#3d4055"; TEXT_COLOR = "#c0caf5"

def speed_to_color(v, vm):
    t = np.clip(v / vm, 0, 1)
    if t < 0.5: return (1.0, t*2, 0.15)
    return (1.0-(t-0.5)*2, 1.0, 0.15+(t-0.5)*0.3)

def run_idm(nv, L, T, dt, v0, nf):
    vl=5.0; spf=max(1,int(T/dt/nf)); sp=L/nv
    pos=np.array([i*sp for i in range(nv)],dtype=float)
    vel=np.ones(nv)*v0*0.95; vel[0]=0.0; vel[1]=v0*0.3
    s0,Th,a,b,d = 2.0,1.0,1.0,1.5,4
    ph,vh,th = [pos.copy()],[vel.copy()],[0.0]; t=0.0
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

def run_bando(nv, L, T, dt, kappa, vm, sc, nf):
    from trafficjams.bando import optimal_velocity
    spf=max(1,int(T/dt/nf)); sp=L/nv
    pos=np.array([i*sp for i in range(nv)],dtype=float)
    vel=np.ones(nv)*optimal_velocity(sp,vm,sc)
    pos[0]=(pos[0]-sp*0.4)%L
    ph,vh,th = [pos.copy()],[vel.copy()],[0.0]; t=0.0
    for _ in range(nf-1):
        for __ in range(spf):
            gaps=np.empty(nv)
            for i in range(nv): gaps[i]=(pos[(i+1)%nv]-pos[i])%L
            vo=optimal_velocity(gaps,vm,sc); acc=kappa*(vo-vel)
            vel=np.maximum(vel+acc*dt,0.0); pos=(pos+vel*dt)%L; t+=dt
        ph.append(pos.copy()); vh.append(vel.copy()); th.append(t)
    return np.array(ph),np.array(vh),np.array(th)

def draw_ring(ax, R_outer, R_inner, n_veh, title):
    Rm = (R_outer+R_inner)/2
    th=np.linspace(0,2*np.pi,200)
    ax.fill_between(np.cos(th)*R_outer,np.sin(th)*R_outer,
                    np.cos(th)*R_inner*0.99,color=ROAD_COLOR,zorder=1)
    ax.add_patch(plt.Circle((0,0),R_outer,fill=False,color=ROAD_EDGE,lw=2,zorder=2))
    ax.add_patch(plt.Circle((0,0),R_inner,fill=False,color=ROAD_EDGE,lw=2,zorder=2))
    for i in range(40):
        a1=i*2*np.pi/40; t_=np.linspace(a1,a1+np.pi/40*0.5,10)
        ax.plot(np.cos(t_)*Rm,np.sin(t_)*Rm,color="#4a4d60",lw=0.8,zorder=2)
    ax.text(0,1.25,title,ha="center",fontsize=11,fontweight="bold",color=TEXT_COLOR)
    cars=[]
    for i in range(n_veh):
        b=FancyBboxPatch((0,0),0.07,0.03,boxstyle="round,pad=0.005",
                          facecolor="#9ece6a",edgecolor="none",zorder=5)
        ax.add_patch(b); cars.append(b)
    return cars, Rm

def main():
    nf=180; nv=22; vm=12.0
    L_idm=250.0; L_bando=528.0

    print("Simulating IDM vs Bando comparison...")
    p1,v1,t1 = run_idm(nv, L_idm, 80.0, 0.02, vm, nf)
    p2,v2,t2 = run_bando(nv, L_bando, 120.0, 0.02, 0.22, 14.0, 12.0, nf)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11,5.5),facecolor=DARK_BG)
    for ax in [ax1,ax2]:
        ax.set_facecolor(DARK_BG); ax.set_xlim(-1.4,1.4); ax.set_ylim(-1.4,1.4)
        ax.set_aspect("equal"); ax.axis("off")

    cars1,Rm1 = draw_ring(ax1, 1.0, 0.78, nv, "IDM")
    cars2,Rm2 = draw_ring(ax2, 1.0, 0.80, nv, "Bando OVM (\u03ba=0.22)")

    fig.text(0.5,0.96,"Model Comparison: Same Perturbation, Different Dynamics",
             ha="center",fontsize=13,fontweight="bold",color=TEXT_COLOR)
    time_text=fig.text(0.5,0.03,"",ha="center",fontsize=10,color=TEXT_COLOR,
                       fontfamily="monospace")

    def update(frame):
        for i in range(nv):
            for pos,vel,L,cars,Rm,vmax in [(p1,v1,L_idm,cars1,Rm1,vm),
                                            (p2,v2,L_bando,cars2,Rm2,14.0)]:
                angle=2*np.pi*pos[frame][i]/L
                cx,cy=np.cos(angle)*Rm,np.sin(angle)*Rm
                rd=np.degrees(angle)-90
                ax_ref = ax1 if cars is cars1 else ax2
                tr=(mtransforms.Affine2D().translate(-0.035,-0.015)
                    .rotate_deg(rd).translate(cx,cy)+ax_ref.transData)
                cars[i].set_transform(tr)
                cars[i].set_facecolor(speed_to_color(vel[frame][i],vmax))
        time_text.set_text(f"t = {t1[frame]:5.1f} s")
        return cars1+cars2+[time_text]

    anim=FuncAnimation(fig,update,frames=nf,interval=40,blit=False)
    plt.tight_layout(pad=1.5)
    sp=os.path.join(RESULTS_DIR,"anim_comparison.gif")
    print(f"Saving {sp} ({nf} frames)...")
    anim.save(sp,writer="pillow",fps=20,dpi=90,savefig_kwargs={"facecolor":DARK_BG})
    plt.close(); print(f"Done -> {os.path.getsize(sp)/1e6:.1f} MB")

if __name__=="__main__":
    os.makedirs(RESULTS_DIR,exist_ok=True); main()
