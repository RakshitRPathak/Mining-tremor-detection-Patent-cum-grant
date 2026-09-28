"""
Stage 1 of 3: binary hazard detection on synthetic 3-axis accelerometer windows.

Generates labelled 3 s windows (100 Hz) for tremor, rockburst, drilling, machinery,
handling, footsteps and quiet baseline; extracts 14 features; trains a Random Forest;
compares threshold-only, ML-only and hybrid (ML AND threshold) detectors; sweeps the
threshold; and simulates 2-of-3 multi-node consensus, node dropout and a faulty node.

Output: ../results/tremor_simulation_results.json
All signals are synthetic. Results validate algorithm logic, not hardware performance.
"""
import numpy as np
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import json, time

FS=100; WIN=3.0; N=int(FS*WIN)
NOISE=0.006  # g rms, approx MPU6050 accel noise at ~50 Hz bandwidth plus mounting

def base(rng):
    t=np.arange(N)/FS
    a=np.zeros((N,3)); a[:,2]=1.0
    a+=rng.normal(0,NOISE,(N,3))
    a+=rng.normal(0,0.01,3)  # per-window offset / tilt
    return t,a

def rand_dir(rng):
    v=rng.normal(size=3); return v/np.linalg.norm(v)

def tremor(rng, scale=1.0):
    t,a=base(rng)
    f=rng.uniform(2,20); amp=rng.uniform(0.02,0.8)*scale
    dur=rng.uniform(0.5,2.5); t0=rng.uniform(0,WIN-dur)
    m=(t>=t0)&(t<t0+dur); tt=t[m]-t0
    env=np.sin(np.pi*tt/dur)*np.exp(-tt*rng.uniform(0,2))
    sig=amp*env*np.sin(2*np.pi*f*tt+rng.uniform(0,6.28))
    a[m]+=np.outer(sig,rand_dir(rng)); return a
def rockburst(rng, scale=1.0):
    t,a=base(rng)
    t0=rng.uniform(0.2,WIN-1.5); amp=rng.uniform(0.3,2.5)*scale
    m=t>=t0; tt=t[m]-t0
    f=rng.uniform(5,30)
    sig=amp*np.exp(-tt*rng.uniform(3,8))*np.sin(2*np.pi*f*tt)
    a[m]+=np.outer(sig,rand_dir(rng)); return a
def machinery(rng):
    t,a=base(rng)
    f=rng.uniform(8,45); amp=rng.uniform(0.02,0.35)
    sig=amp*(np.sin(2*np.pi*f*t)+0.3*np.sin(4*np.pi*f*t))
    a+=np.outer(sig,rand_dir(rng)); return a
def handling(rng):
    t,a=base(rng)
    for _ in range(rng.integers(1,4)):
        i=rng.integers(0,N-5); amp=rng.uniform(0.2,2.0)
        k=rng.integers(1,4)
        a[i:i+k]+=amp*rand_dir(rng)*rng.choice([-1,1])
    return a
def footsteps(rng):
    t,a=base(rng)
    f=rng.uniform(1.5,2.5); amp=rng.uniform(0.02,0.2)
    for ts in np.arange(rng.uniform(0,0.5),WIN,1/f):
        i=int(ts*FS)
        if i<N-4: a[i:i+4]+=amp*np.array([0.2,0.2,1])*np.hanning(4)[:,None]
    return a
def baseline(rng):
    return base(rng)[1]

HAZ={'tremor':tremor,'rockburst':rockburst}
def drilling(rng):
    t,a=base(rng)
    f=rng.uniform(8,40); amp=rng.uniform(0.05,0.6)
    dur=rng.uniform(0.5,2.5); t0=rng.uniform(0,WIN-dur); m=(t>=t0)&(t<t0+dur); tt=t[m]-t0
    ramp=np.minimum(1,np.minimum(tt,dur-tt)/0.15)
    a[m]+=np.outer(amp*ramp*np.sin(2*np.pi*f*tt)*(1+0.3*rng.normal(size=m.sum())),rand_dir(rng)); return a
NON={'drilling':drilling,'machinery':machinery,'handling':handling,'footsteps':footsteps,'baseline':baseline}

def env(a):
    dyn=a-a.mean(0); mag=np.linalg.norm(dyn,axis=1)
    return np.sqrt(np.convolve(mag**2,np.ones(10)/10,'same'))
def feats(a):
    smv=np.linalg.norm(a,axis=1); d=smv-np.median(smv); e=env(a)
    ad=np.abs(d)
    spec=np.abs(np.fft.rfft(d*np.hanning(N))); fr=np.fft.rfftfreq(N,1/FS)
    spec[0]=0
    domf=fr[np.argmax(spec)]; cent=(fr*spec).sum()/(spec.sum()+1e-9)
    low=spec[(fr>=1)&(fr<=20)].sum()/(spec.sum()+1e-9)
    above=(ad>0.05).mean()
    # longest run above 0.05 g (s)
    run=best=0
    for v in ad>0.05:
        run=run+1 if v else 0; best=max(best,run)
    zcr=(np.diff(np.sign(d))!=0).mean()
    dyn=np.linalg.norm(a-a.mean(0),axis=1)
    spec=np.abs(np.fft.rfft((dyn-dyn.mean())*np.hanning(N))); spec[0]=0
    domf=fr[np.argmax(spec)]; cent=(fr*spec).sum()/(spec.sum()+1e-9); low=spec[(fr>=1)&(fr<=20)].sum()/(spec.sum()+1e-9)
    ab=e>0.03; run=best=0
    for v in ab: run=run+1 if v else 0; best=max(best,run)
    return [d.std(),ad.max(),e.max(),e.mean(),*a.std(0),domf,cent,low,ab.mean(),best/FS,zcr,kurtosis(e)]
FEATS=['smv_std','smv_peak','dyn_env_peak','dyn_env_mean','std_x','std_y','std_z','dom_freq','spec_centroid','low_band_ratio','frac_above_0.05g','longest_run_s','zcr','kurtosis']

def peak_smv(a):
    smv=np.linalg.norm(a,axis=1); return np.abs(smv-np.median(smv)).max()
def peak(a): return env(a).max()
def longest(a,thr=0.03):
    ad=env(a)>thr
    run=best=0
    for v in ad: run=run+1 if v else 0; best=max(best,run)
    return best/FS

def dataset(n_per, seed):
    rng=np.random.default_rng(seed); X=[];y=[];c=[];raw=[]
    for name,g in {**HAZ,**NON}.items():
        lab=1 if name in HAZ else 0
        for _ in range(n_per*(2 if lab else 1)):
            a=g(rng); X.append(feats(a)); y.append(lab); c.append(name); raw.append((peak(a),longest(a),peak_smv(a)))
    return np.array(X),np.array(y),np.array(c),np.array(raw)

def m(y,p):
    cm=confusion_matrix(y,p,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
    return dict(accuracy=accuracy_score(y,p),precision=precision_score(y,p,zero_division=0),recall=recall_score(y,p),
                f1=f1_score(y,p),fpr=fp/(fp+tn),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

R={}
Xtr,ytr,ctr,rtr=dataset(400,1)   # 800 per hazard class, 400 per non-hazard class = 3200
Xte,yte,cte,rte=dataset(200,2)   # independent seed
rf=RandomForestClassifier(n_estimators=50,max_depth=10,random_state=0,n_jobs=-1)
cv=cross_validate(rf,Xtr,ytr,cv=StratifiedKFold(5,shuffle=True,random_state=0),scoring=['accuracy','precision','recall','f1'])
R['cv']={k:(float(v.mean()),float(v.std())) for k,v in cv.items() if k.startswith('test_')}
rf.fit(Xtr,ytr); prob=rf.predict_proba(Xte)[:,1]; pml=(prob>=0.5).astype(int)
R['n_train']=len(ytr); R['n_test']=len(yte)
R['ml']=m(yte,pml)
TAU=0.05; TDUR=0.3
pth=((rte[:,0]>=TAU)&(rte[:,1]>=TDUR)).astype(int)
R['thr']=m(yte,pth)
from numpy import array
R['thr_smv_only']=m(yte,((rte[:,2]>=0.15)).astype(int))
phy=(pml&pth); R['hyb']=m(yte,phy)
# per class
R['per_class']={}
for cl in np.unique(cte):
    k=cte==cl
    R['per_class'][cl]={'n':int(k.sum()),'thr':float(pth[k].mean()),'ml':float(pml[k].mean()),'hyb':float(phy[k].mean())}
# sweep
R['sweep']=[]
for tau in np.round(np.arange(0.02,0.21,0.02),2):
    pt=((rte[:,0]>=tau)&(rte[:,1]>=TDUR)).astype(int); ph=pml&pt
    R['sweep'].append(dict(tau=float(tau),thr=m(yte,pt),hyb=m(yte,ph)))
R['importance']=sorted(zip(FEATS,rf.feature_importances_.round(3).tolist()),key=lambda x:-x[1])[:6]
nodes=sum(e.tree_.node_count for e in rf.estimators_)
R['model_nodes']=int(nodes); R['est_bytes']=int(nodes*16)
t0=time.perf_counter(); 
for i in range(200): rf.predict(Xte[i:i+1])
R['py_infer_ms']=(time.perf_counter()-t0)/200*1000

# multi-node simulation: 3 nodes, k=2
def node_decide(a):
    p=rf.predict_proba([feats(a)])[0,1]>=0.5
    return int(p and peak(a)>=TAU and longest(a)>=TDUR)
rng=np.random.default_rng(3)
def shared(g, att):
    # same event, attenuated per node, independent noise
    s=np.random.SeedSequence(int(rng.integers(1e9)))
    out=[]
    for i,at in enumerate(att):
        r=np.random.default_rng(s.spawn(1)[0]) if False else np.random.default_rng(int(rng.integers(1e9)))
        out.append(g(r,scale=at))
    return out
T=300
mn={}
# 4.1 shared hazard
conf=0; single=0
for _ in range(T):
    g=[tremor,rockburst][rng.integers(0,2)]
    att=rng.uniform(0.4,1.0,3)
    seed=int(rng.integers(1e9))
    ds=[node_decide(g(np.random.default_rng(seed+i*0),scale=a)) for i,a in enumerate(att)]  # same event params, scaled
    conf+=sum(ds)>=2; single+=ds[0]
mn['shared_confirm']=conf/T; mn['single_node_detect']=single/T
# 4.2 local disturbance on one node (non-hazard local events + local tremor-like knock)
fc=0; fs=0
for _ in range(T):
    g=[machinery,handling,footsteps][rng.integers(0,3)]
    ds=[node_decide(g(rng)),node_decide(baseline(rng)),node_decide(baseline(rng))]
    fc+=sum(ds)>=2; fs+=ds[0]
mn['local_false_confirm']=fc/T; mn['local_single_false']=fs/T
# 4.3 dropout: 2 remaining nodes, need 2
conf=0
for _ in range(T):
    g=[tremor,rockburst][rng.integers(0,2)]; seed=int(rng.integers(1e9))
    ds=[node_decide(g(np.random.default_rng(seed),scale=a)) for a in rng.uniform(0.4,1.0,2)]
    conf+=sum(ds)>=2
mn['dropout_confirm_2of2']=conf/T
# 4.4 faulty node: loose node random spikes every window
def faulty(r):
    a=baseline(r)
    L=int(r.uniform(0.3,1.2)*FS); i=r.integers(0,N-L); tt=np.arange(L)/FS
    a[i:i+L]+=np.outer(r.uniform(0.1,0.6)*np.sign(np.sin(2*np.pi*r.uniform(4,15)*tt)),rand_dir(r))
    return a
fc=0; ff=0; score=1.0; scores=[]
for _ in range(T):
    ds=[node_decide(faulty(rng)),node_decide(baseline(rng)),node_decide(baseline(rng))]
    ff+=ds[0]; fc+=sum(ds)>=2
    if ds[0] and sum(ds[1:])==0: score=max(0.0,score-0.05)
    else: score=min(1.0,score+0.01)
    scores.append(score)
mn['faulty_node_trigger_rate']=ff/T; mn['faulty_false_confirm']=fc/T; mn['faulty_score_final']=score
mn['faulty_windows_to_below_0.5']=int(next((i+1 for i,s in enumerate(scores) if s<0.5),-1))
mn['trials']=T
R['multinode']=mn
json.dump(R,open('../results/tremor_simulation_results.json','w'),indent=1,default=float)
print(json.dumps(R,indent=1,default=float))
