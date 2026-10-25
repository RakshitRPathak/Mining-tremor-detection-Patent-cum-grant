"""
Stage 3 of 3: 4-class model, annotated multi-node run and all figures.

Trains a 4-class Random Forest (Normal, Minor vibration, Major tremor, Structural
disturbance), runs a 40 s three-node scenario (shared tremor + local drilling) with a
per-window event log, tracks a faulty node's reliability score, and renders figures
from all three stages.

Requires Stages 1 and 2. Outputs: ../results/extended_analysis_results.json, ../figures/*.png
"""
import numpy as np, json, io, contextlib
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
with contextlib.redirect_stdout(io.StringIO()):
    import tremor_simulation as sim
from tremor_simulation import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import os; OUT='../figures/'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
C1='#1F3A5F'; C2='#C8553D'; C3='#6A994E'; C4='#E0A458'
R1=json.load(open('../results/tremor_simulation_results.json')); R2=json.load(open('../results/streaming_and_firmware_results.json'))
R={}
# ---------- (a) 4-class model aligned with IDF-B Step 4 ----------
MAP={'baseline':'Normal','footsteps':'Normal','machinery':'Minor vibration','drilling':'Minor vibration','handling':'Minor vibration','tremor':'Major tremor','rockburst':'Structural disturbance'}
LAB=['Normal','Minor vibration','Major tremor','Structural disturbance']
def mc(seed,n):
    rng=np.random.default_rng(seed); X=[];y=[]
    gens={**HAZ,**NON}
    for name,g in gens.items():
        k=n*(2 if name in HAZ else 1)
        for _ in range(k): X.append(feats(g(rng))); y.append(LAB.index(MAP[name]))
    return np.array(X),np.array(y)
Xa,ya=mc(21,400); Xb,yb=mc(22,200)
rf4=RandomForestClassifier(n_estimators=50,max_depth=10,random_state=0,n_jobs=-1).fit(Xa,ya)
pb=rf4.predict(Xb); cm=confusion_matrix(yb,pb)
rep=classification_report(yb,pb,target_names=LAB,output_dict=True)
R['mc']=dict(cm=cm.tolist(),acc=accuracy_score(yb,pb),macro_f1=f1_score(yb,pb,average='macro'),
             per={l:dict(p=rep[l]['precision'],r=rep[l]['recall'],f1=rep[l]['f1-score'],n=int(rep[l]['support'])) for l in LAB},
             n_train=len(ya),n_test=len(yb))
hz_true=np.isin(yb,[2,3]); hz_pred=np.isin(pb,[2,3])
R['mc']['hazard_collapsed']=dict(recall=float((hz_pred&hz_true).sum()/hz_true.sum()),fpr=float((hz_pred&~hz_true).sum()/(~hz_true).sum()))
print(json.dumps(R['mc'],indent=1))

def heat(ax,m,labs,title,fmt='d'):
    ax.imshow(m,cmap='Blues'); ax.set_xticks(range(len(labs))); ax.set_yticks(range(len(labs)))
    ax.set_xticklabels(labs,rotation=20,ha='right'); ax.set_yticklabels(labs); ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    for i in range(len(labs)):
        for j in range(len(labs)):
            v=m[i][j]; ax.text(j,i,format(v,fmt),ha='center',va='center',color='white' if v>np.max(m)*0.6 else 'black')
# Fig: multi-class confusion
fig,ax=plt.subplots(figsize=(5.2,4.2)); heat(ax,cm,['Normal','Minor vib.','Major tremor','Structural'],'4-class Random Forest, simulated test set (n=1,800)')
plt.tight_layout(); plt.savefig(OUT+'s_cm4.png',dpi=200); plt.close()
# Fig: binary confusion matrices (threshold, ML, hybrid)
fig,axs=plt.subplots(1,3,figsize=(9,3))
for ax,k,t in zip(axs,['thr','ml','hyb'],['Threshold only','Random Forest only','Hybrid (RF AND threshold)']):
    m=R1[k]; heat(ax,[[m['tn'],m['fp']],[m['fn'],m['tp']]],['Non-hazard','Hazard'],t)
plt.tight_layout(); plt.savefig(OUT+'s_cm2.png',dpi=200); plt.close()
# Fig: example waveforms
rng=np.random.default_rng(5)
fig,axs=plt.subplots(2,4,figsize=(10,4),sharex=True)
t=np.arange(N)/FS
for ax,(name,g) in zip(axs.flat,list({**HAZ,**NON}.items())+[('',None)]):
    if g is None: ax.axis('off'); continue
    a=g(rng); dyn=np.linalg.norm(a-a.mean(0),axis=1)
    ax.plot(t,dyn,color=C2 if name in HAZ else C1,lw=0.7); ax.axhline(TAU,ls='--',color='grey',lw=0.7)
    ax.set_title(f"{name} ({'hazard' if name in HAZ else 'non-hazard'})",fontsize=8.5)
for ax in axs[1]: ax.set_xlabel('Time (s)')
for ax in axs[:,0]: ax.set_ylabel('Dynamic accel. (g)')
plt.tight_layout(); plt.savefig(OUT+'s_waves.png',dpi=200); plt.close()
# Fig: detector comparison bars
fig,ax=plt.subplots(figsize=(7,3)); keys=[('thr_smv_only','Threshold (SMV)'),('thr','Threshold (dynamic)'),('ml','Random Forest'),('hyb','Hybrid')]
x=np.arange(len(keys)); w=0.26
for i,(met,lab,c) in enumerate([('accuracy','Accuracy',C1),('recall','Recall',C3),('fpr','False-alarm rate',C2)]):
    vals=[R1[k][met]*100 for k,_ in keys]; b=ax.bar(x+(i-1)*w,vals,w,label=lab,color=c)
    for xi,v in zip(x+(i-1)*w,vals): ax.text(xi,v+1,f'{v:.1f}',ha='center',fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([l for _,l in keys]); ax.set_ylabel('%'); ax.set_ylim(0,110); ax.legend(ncol=3,frameon=False,loc='upper left')
ax.set_title('Detector comparison on simulated test set (800 hazard / 1,000 non-hazard windows)')
plt.tight_layout(); plt.savefig(OUT+'s_detcmp.png',dpi=200); plt.close()
# Fig: threshold sweep
sw=R1['sweep']; tau=[s['tau'] for s in sw]
fig,ax=plt.subplots(figsize=(6.5,3.2))
ax.plot(tau,[s['thr']['recall']*100 for s in sw],'-o',color=C3,ms=3,label='Recall, threshold only')
ax.plot(tau,[s['hyb']['recall']*100 for s in sw],'--o',color=C3,ms=3,label='Recall, hybrid')
ax.plot(tau,[s['thr']['fpr']*100 for s in sw],'-s',color=C2,ms=3,label='False-alarm rate, threshold only')
ax.plot(tau,[s['hyb']['fpr']*100 for s in sw],'--s',color=C2,ms=3,label='False-alarm rate, hybrid')
ax.axvline(0.05,color='grey',ls=':',lw=0.8); ax.text(0.052,50,'operating\npoint 0.05 g',fontsize=7)
ax.set_xlabel('Threshold τ on dynamic acceleration envelope (g)'); ax.set_ylabel('%'); ax.legend(frameon=False,fontsize=7); ax.set_title('Threshold sweep (simulated)')
plt.tight_layout(); plt.savefig(OUT+'s_sweep.png',dpi=200); plt.close()
# Fig: feature importance
imp=sorted(zip(FEATS,sim.rf.feature_importances_),key=lambda x:x[1])
fig,ax=plt.subplots(figsize=(5.5,3.6)); ax.barh([f for f,_ in imp],[v for _,v in imp],color=C1); ax.set_xlabel('Gini importance'); ax.set_title('Random Forest feature importance (binary model)')
plt.tight_layout(); plt.savefig(OUT+'s_imp.png',dpi=200); plt.close()
R['importance']=[(f,float(v)) for f,v in sorted(imp,key=lambda x:-x[1])]
# Fig: detection heatmap freq x amplitude (Doc 3)
rows=R2['doc3']; freqs=[2,5,10,20]; amps=[0.05,0.1,0.2]; M=np.zeros((3,4))
for r in rows:
    if r['stimulus'].startswith('Tremor'):
        f=int(r['stimulus'].split()[1]); a=float(r['stimulus'].split(', ')[1].split()[0])
        M[amps.index(a),freqs.index(f)]=r['detected']/r['trials']*100
fig,ax=plt.subplots(figsize=(4.8,2.8)); im=ax.imshow(M,cmap='RdYlGn',vmin=0,vmax=100,origin='lower')
ax.set_xticks(range(4)); ax.set_xticklabels([f'{f} Hz' for f in freqs]); ax.set_yticks(range(3)); ax.set_yticklabels([f'{a} g' for a in amps])
for i in range(3):
    for j in range(4): ax.text(j,i,f'{M[i,j]:.0f}%',ha='center',va='center',fontsize=8)
ax.set_title('Simulated tremor detection rate (50 trials per cell)'); plt.colorbar(im,ax=ax,fraction=0.03)
plt.tight_layout(); plt.savefig(OUT+'s_heat.png',dpi=200); plt.close()
# Fig: latency histogram
L=np.array(R2['doc9']['lat_all'])
fig,ax=plt.subplots(figsize=(6,2.8)); ax.hist(L,bins=np.arange(0,11.5,0.5),color=C1,edgecolor='white')
for v,lab in [(np.median(L),'median'),(np.percentile(L,95),'95th pct')]: ax.axvline(v,color=C2,ls='--',lw=0.9); ax.text(v+0.1,ax.get_ylim()[1]*0.8,f'{lab}\n{v:.2f} s',fontsize=7,color=C2)
ax.set_xlabel('Detection latency from stimulus onset (s)'); ax.set_ylabel('Detected events'); ax.set_title(f'Simulated detection latency (n={len(L)} detected hazard trials)')
plt.tight_layout(); plt.savefig(OUT+'s_lat.png',dpi=200); plt.close()
R['lat']=dict(n=int(len(L)),mean=float(L.mean()),median=float(np.median(L)),p90=float(np.percentile(L,90)),p95=float(np.percentile(L,95)),max=float(L.max()),share_le2=float((L<=2).mean()))
# Fig: firmware comparison
f=R2['doc7_8']
fig,axs=plt.subplots(1,3,figsize=(9,2.8))
axs[0].bar(['Current','Designed'],[f['current']['delivery_rate']*100,f['designed']['delivery_rate']*100],color=[C2,C3]); axs[0].set_title('Events delivered (%)')
axs[1].bar(['Current','Designed'],[f['current']['mean_sensing_downtime_min'],f['designed']['mean_sensing_downtime_min']],color=[C2,C3]); axs[1].set_title('Sensing downtime (min/day)')
axs[2].bar(['Current','Designed'],[f['current']['days_ending_halted_or_hung'],f['designed']['days_ending_halted_or_hung']],color=[C2,C3]); axs[2].set_title('Days ending offline (of 200)')
for ax in axs:
    for p in ax.patches: ax.text(p.get_x()+p.get_width()/2,p.get_height(),f'{p.get_height():.1f}',ha='center',va='bottom',fontsize=7)
plt.suptitle('Firmware fault-injection simulation, 200 days each (assumed fault rates)',fontsize=9); plt.tight_layout(); plt.savefig(OUT+'s_fw.png',dpi=200); plt.close()

# ---------- (c) multi-node event log ----------
rng=np.random.default_rng(42); T=40; n=int(T*FS)
def stream():
    a=np.zeros((n,3)); a[:,2]=1.0; a+=rng.normal(0,NOISE,a.shape); a+=rng.normal(0,0.01,3); return a
def burst(a,t0,dur,f,amp,d):
    i0=int(t0*FS); k=int(dur*FS); tt=np.arange(k)/FS; e=np.minimum(1,np.minimum(tt,dur-tt)/0.3)
    a[i0:i0+k]+=np.outer(amp*e*np.sin(2*np.pi*f*tt),d)
def drill(a,t0,dur):
    i0=int(t0*FS); k=int(dur*FS); tt=np.arange(k)/FS; ramp=np.minimum(1,np.minimum(tt,dur-tt)/0.15)
    a[i0:i0+k]+=np.outer(0.35*ramp*np.sin(2*np.pi*22*tt)*(1+0.3*rng.normal(size=k)),rand_dir(rng))
nodes=[stream() for _ in range(3)]; d=rand_dir(rng)
for a,att,delay in zip(nodes,[1.0,0.7,0.5],[0,0.08,0.15]): burst(a,12+delay,6,5,0.2*att,d)
drill(nodes[1],28,2.0); drill(nodes[1],31,1.5)
WIN=int(3*FS); HOP=int(1.5*FS)
dec={}
for ni,a in enumerate(nodes):
    for s in range(0,n-WIN+1,HOP):
        w=a[s:s+WIN]; e=env(w); run=best=0
        for v in e>TAU: run=run+1 if v else 0; best=max(best,run)
        p=rf.predict_proba([feats(w)])[0,1]; thr=e.max()>=TAU and best/FS>=TDUR
        dec[(ni,(s+WIN)/FS)]=(float(p),bool(thr),float(e.max()))
times=sorted(set(t for _,t in dec))
log=[]; confirmed=[]
for t_ in times:
    cand=[ni for ni in range(3) if dec[(ni,t_)][0]>=0.5 and dec[(ni,t_)][1]]
    for ni in range(3):
        p,thr,pk=dec[(ni,t_)]
        if p>=0.5 or thr:
            log.append(dict(t=t_,node=f'N{ni+1}',peak=pk,p=p,ml='Hazard' if p>=0.5 else 'Non-hazard',thr='Pass' if thr else 'Fail',
                            local='Candidate' if (p>=0.5 and thr) else 'Rejected'))
    if len(cand)>=2: confirmed.append((t_,cand))
R['mn_log']=log; R['mn_confirmed']=[(t,[f'N{c+1}' for c in cs]) for t,cs in confirmed]
fig,axs=plt.subplots(3,1,figsize=(8,4.6),sharex=True)
tt=np.arange(n)/FS
for ni,(ax,a) in enumerate(zip(axs,nodes)):
    ax.plot(tt,np.sqrt(np.convolve(np.linalg.norm(a-a.mean(0),axis=1)**2,np.ones(10)/10,'same')),color=C1,lw=0.6)
    ax.axhline(TAU,ls='--',color='grey',lw=0.6); ax.set_ylabel(f'N{ni+1} (g)'); ax.set_ylim(0,0.35)
    for (nj,t_),(p,thr,_) in dec.items():
        if nj==ni and p>=0.5 and thr: ax.plot(t_,0.3,'v',color=C2,ms=5)
        elif nj==ni and thr: ax.plot(t_,0.3,'x',color='grey',ms=5)
for t_,cs in confirmed: 
    for ax in axs: ax.axvspan(t_-0.2,t_+0.2,color=C3,alpha=0.25)
axs[0].set_title('Simulated 3-node run: shared 5 Hz tremor at 12 s; local drilling at N2 only at 28–33 s\n▼ local hazard candidate   × threshold-only trigger (rejected by ML)   green band = 2-of-3 confirmation',fontsize=8)
axs[2].set_xlabel('Time (s)'); plt.tight_layout(); plt.savefig(OUT+'s_mn.png',dpi=200); plt.close()
# reliability score decay for faulty node
rng2=np.random.default_rng(7); score=1.0; sc=[]
def faulty(r):
    a=baseline(r); L_=int(r.uniform(0.3,1.2)*FS); i=r.integers(0,N-L_); tt_=np.arange(L_)/FS
    a[i:i+L_]+=np.outer(r.uniform(0.1,0.6)*np.sign(np.sin(2*np.pi*r.uniform(4,15)*tt_)),rand_dir(r)); return a
for k in range(40):
    fn=rf.predict_proba([feats(faulty(rng2))])[0,1]>=0.5
    if fn: score=max(0,score-0.05)
    else: score=min(1,score+0.01)
    sc.append(score)
fig,ax=plt.subplots(figsize=(5.5,2.5)); ax.plot(range(1,41),sc,'-o',ms=3,color=C2); ax.axhline(0.5,ls='--',color='grey',lw=0.7)
ax.set_xlabel('Window number'); ax.set_ylabel('Reliability score'); ax.set_title('Faulty node reliability score (simulated)'); plt.tight_layout(); plt.savefig(OUT+'s_rel.png',dpi=200); plt.close()
json.dump(R,open('../results/extended_analysis_results.json','w'),indent=1,default=float)
print(json.dumps(R['mn_confirmed'])); print(R['lat'])
for r in log: print(r)
