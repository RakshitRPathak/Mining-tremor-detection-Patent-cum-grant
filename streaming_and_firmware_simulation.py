"""
Stage 2 of 3: streaming detection and firmware fault-injection simulation.

Runs the Stage 1 model as a sliding-window detector (3 s window, 1.5 s hop, cooldown)
over continuous simulated streams for a tremor frequency x amplitude matrix, drop-mass
impacts and non-hazard sources; records detection rates, latency and duplicate alerts.
Then simulates 200 days of current vs designed firmware under injected Wi-Fi outages,
sensor dropouts, power cuts and hangs (assumed fault rates).

Requires Stage 1. Output: ../results/streaming_and_firmware_results.json
"""
import numpy as np, json, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import tremor_simulation as sim   # reuses trained RF, feature extraction, thresholds from Annex A
from tremor_simulation import rf, feats, env, FS, NOISE, rand_dir, TAU, TDUR
WINS=int(3*FS); HOP=int(1.5*FS); COOLDOWN=3.0
rng=np.random.default_rng(11)

def stream(T):
    a=np.zeros((int(T*FS),3)); a[:,2]=1.0
    a+=rng.normal(0,NOISE,a.shape); a+=rng.normal(0,0.01,3); return a
def add_burst(a,t0,dur,f,amp):
    i0=int(t0*FS); n=int(dur*FS); tt=np.arange(n)/FS
    env_=np.minimum(1,np.minimum(tt,dur-tt)/0.3)
    a[i0:i0+n]+=np.outer(amp*env_*np.sin(2*np.pi*f*tt),rand_dir(rng))
def add_drop(a,t0,amp):
    i0=int(t0*FS); n=int(1.5*FS); tt=np.arange(n)/FS; f=rng.uniform(10,30)
    a[i0:i0+n]+=np.outer(amp*np.exp(-tt*5)*np.sin(2*np.pi*f*tt),rand_dir(rng))
def add_machinery(a,f,amp):
    t=np.arange(len(a))/FS; a+=np.outer(amp*(np.sin(2*np.pi*f*t)+0.3*np.sin(4*np.pi*f*t)),rand_dir(rng))
def add_handling(a,T):
    for ts in rng.uniform(0,T-1,rng.integers(5,15)):
        i=int(ts*FS); a[i:i+rng.integers(1,4)]+=rng.uniform(0.2,2.0)*rand_dir(rng)
def add_footsteps(a,T):
    f=rng.uniform(1.5,2.5); amp=rng.uniform(0.02,0.2)
    for ts in np.arange(0,T,1/f):
        i=int(ts*FS)
        if i<len(a)-4: a[i:i+4]+=amp*np.array([0.2,0.2,1])*np.hanning(4)[:,None]

def run_stream(a, cooldown=COOLDOWN):
    """Returns alert times (s). Window decision = RF AND dynamic-envelope threshold; alert suppressed during cooldown."""
    alerts=[]; last=-1e9
    for s in range(0,len(a)-WINS+1,HOP):
        w=a[s:s+WINS]; e=env(w); run=best=0
        for v in e>TAU: run=run+1 if v else 0; best=max(best,run)
        ml=rf.predict_proba([feats(w)])[0,1]>=0.5
        if ml and e.max()>=TAU and best/FS>=TDUR:
            t=(s+WINS)/FS
            if t-last>=cooldown: alerts.append(t); last=t
    return alerts

R={'doc3':[], 'doc9':{}}
TR=50
# ---- Document 3 stimulus matrix ----
rows=[]
for f in [2,5,10,20]:
    for amp in [0.05,0.1,0.2]:
        rows.append((f'Tremor {f} Hz, {amp} g','haz',dict(f=f,amp=amp)))
rows+= [('Drop mass 30 cm (modelled 1.0 g peak)','haz',dict(drop=1.0)),
        ('Drop mass 10 cm (modelled 0.35 g peak)','haz',dict(drop=0.35)),
        ('Drilling bursts 60 s','non',dict(drill=True)),('Machinery 60 s','non',dict(mach=True)),('Handling 60 s','non',dict(hand=True)),
        ('Footsteps 60 s','non',dict(foot=True)),('Quiet baseline 600 s','non',dict(base=True))]
lat_all=[]; dup=[]
for name,kind,p in rows:
    det=0; lats=[]; fa=0; alerts_n=0; dur_total=0
    ntr = 5 if 'base' in p else TR
    for _ in range(ntr):
        if kind=='haz':
            T=20; a=stream(T); t0=rng.uniform(5,10)
            if 'f' in p:
                d=rng.uniform(3,10); add_burst(a,t0,d,p['f'],p['amp'])
            else:
                d=1.5; add_drop(a,t0,p['drop'])
            al=run_stream(a); hits=[t for t in al if t>=t0 and t<=t0+d+3]
            if hits: det+=1; lats.append(hits[0]-t0); dup.append(len(hits)-1)
            fa+=len(al)-len(hits)
        else:
            T=600 if 'base' in p else 60; a=stream(T)
            if 'mach' in p: add_machinery(a,rng.uniform(8,45),rng.uniform(0.02,0.35))
            if 'hand' in p: add_handling(a,T)
            if 'foot' in p: add_footsteps(a,T)
            if 'drill' in p:
                for ts in np.arange(rng.uniform(0,5),T-4,rng.uniform(6,12)):
                    d=rng.uniform(0.5,2.5); f=rng.uniform(8,40); amp=rng.uniform(0.05,0.6)
                    i0=int(ts*FS); n=int(d*FS); tt=np.arange(n)/FS; ramp=np.minimum(1,np.minimum(tt,d-tt)/0.15)
                    a[i0:i0+n]+=np.outer(amp*ramp*np.sin(2*np.pi*f*tt)*(1+0.3*rng.normal(size=n)),rand_dir(rng))
            al=run_stream(a); fa+=len(al); alerts_n+=len(al); dur_total+=T
    row=dict(stimulus=name,trials=ntr,lats=[float(x) for x in lats])
    if kind=='haz':
        row.update(detected=det,missed=ntr-det,false_triggers=fa,median_latency=float(np.median(lats)) if lats else None,
                   max_latency=float(np.max(lats)) if lats else None); lat_all+=lats
    else:
        row.update(streams_with_false_alert=None,false_alerts=fa,false_alerts_per_hour=fa/(dur_total/3600))
    R['doc3'].append(row); print(row)
R['doc9']['lat_all']=[float(x) for x in lat_all]; R['doc9']['dup3']=[int(x) for x in dup]
R['doc9']['latency_median']=float(np.median(lat_all)); R['doc9']['latency_p95']=float(np.percentile(lat_all,95)); R['doc9']['latency_max']=float(np.max(lat_all))
R['doc9']['mean_duplicate_alerts_3s_cooldown']=float(np.mean(dup)); R['doc9']['share_events_with_duplicates_3s']=float(np.mean(np.array(dup)>0))
# duplicate check with longer cooldown on long tremors
d2=[]
for _ in range(50):
    a=stream(30); t0=8; d=rng.uniform(6,10); add_burst(a,t0,d,rng.choice([2,5,10]),0.2)
    al=run_stream(a,cooldown=15); d2.append(max(0,len([t for t in al if t0<=t<=t0+d+3])-1))
R['doc9']['mean_duplicate_alerts_15s_cooldown']=float(np.mean(d2))
print(R['doc9'])

# ---- Documents 7 & 8: firmware logic model with fault injection (discrete time, 0.1 s ticks, 24 h) ----
def fw_sim(design, seed, hours=24):
    r=np.random.default_rng(seed); dt=0.1; T=int(hours*3600/dt)
    # fault schedule
    wifi_down=np.zeros(T,bool); imu_down=np.zeros(T,bool); power_cut=set(); hang=set()
    for _ in range(r.poisson(12)):   # ~12 Wi-Fi outages/day, 10-300 s
        s=r.integers(0,T); wifi_down[s:s+int(r.uniform(10,300)/dt)]=True
    for _ in range(r.poisson(3)):    # ~3 I2C/sensor dropouts/day, 2-30 s
        s=r.integers(0,T); imu_down[s:s+int(r.uniform(2,30)/dt)]=True
    for _ in range(r.poisson(1)): power_cut.add(int(r.integers(0,T)))
    for _ in range(r.poisson(1)): hang.add(int(r.integers(0,T)))
    events=sorted(r.integers(0,T,r.poisson(40)))   # ~40 hazard detections/day to transmit
    ev=set(events)
    alive=True; hung=False; hung_t=0; sensing=True; queue=[]; delivered={}; lost=0; halted=False
    boot_until=0; sense_down_time=0; recover_times=[]; imu_fail_start=None
    for k in range(T):
        if k in power_cut:
            if not design: queue=[]   # RAM only
            boot_until=k+int(5/dt)       # reboot ~5 s (+recalibration)
        if k in hang:
            hung=True; hung_t=k
        if hung:
            if design and k-hung_t>=int(8/dt): hung=False; boot_until=k+int(5/dt)   # watchdog 8 s
            elif not design: pass   # stays hung until manual reset
        booting=k<boot_until
        imu_ok=not imu_down[k]
        if not design and imu_down[k] and booting: halted=True   # boot with IMU absent -> infinite halt
        if not design and imu_down[k] and not booting: pass      # stale data, no detection possible
        can_sense=imu_ok and not booting and not hung and not halted
        if imu_down[k] and imu_fail_start is None: imu_fail_start=k
        if design and imu_fail_start is not None and imu_ok and not booting:
            # retry every 5 s -> recovers at next retry tick
            if (k-imu_fail_start)%int(5/dt)==0: recover_times.append((k-imu_fail_start)*dt); imu_fail_start=None
        if not design and imu_fail_start is not None and imu_ok: imu_fail_start=None
        if not can_sense: sense_down_time+=dt
        if k in ev:
            if can_sense:
                if design or not wifi_down[k]: queue.append(k)
                else: lost+=1   # current firmware: event during outage dropped (Blynk.logEvent fails)
            else: lost+=1
        if queue and not wifi_down[k] and not booting and not hung and not halted:
            for q in queue: delivered[q]=(k-q)*dt
            queue=[]
    return dict(events=len(events),delivered=len(delivered),lost=lost+len(queue),
                max_delay=max(delivered.values()) if delivered else 0,
                median_delay=float(np.median(list(delivered.values()))) if delivered else 0,
                sensing_downtime_s=sense_down_time, imu_recover_max=max(recover_times) if recover_times else None,
                halted=halted, hung_left=hung)
res={'current':[], 'designed':[]}
for s in range(200):
    res['current'].append(fw_sim(False,s)); res['designed'].append(fw_sim(True,s))
summ={}
for k,v in res.items():
    ev=sum(x['events'] for x in v); dl=sum(x['delivered'] for x in v)
    summ[k]=dict(days=len(v),events=ev,delivered=dl,delivery_rate=dl/ev,
                 median_delay_s=float(np.median([x['median_delay'] for x in v if x['delivered']])),
                 worst_delay_s=float(max(x['max_delay'] for x in v)),
                 mean_sensing_downtime_min=float(np.mean([x['sensing_downtime_s'] for x in v])/60),
                 days_ending_halted_or_hung=int(sum(x['halted'] or x['hung_left'] for x in v)),
                 imu_recover_max_s=max([x['imu_recover_max'] for x in v if x['imu_recover_max'] is not None], default=None))
R['doc7_8']=summ; print(json.dumps(summ,indent=1))
json.dump(R,open('../results/streaming_and_firmware_results.json','w'),indent=1,default=float)
