"""Reassemble from cache with stages + per-species G (CPU)."""
import sys, json, os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'provenance'))
import reassemble as RA
from pathlib import Path
CACHE=os.environ.get('RS_CACHE',os.path.join(os.path.dirname(os.path.abspath(__file__)),'cache'))
REX=os.environ.get('RS_REACTIONS',os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),'experiments','qm_mlip_solvation','scripts','reactions_tecrdb_std.json'))
out=sys.argv[1]; extra=sys.argv[2:]
SOLV=os.environ.get('RS_SOLV','cosmo'); VIA=os.environ.get('RS_VIA') or ('primary' if SOLV=='cosmo' else 'solv_also'); ANC=os.environ.get('RS_ANCH','on')
args=RA.parse_args(['--cache',CACHE,'--reactions',REX,'--anchors',ANC,'--solv-model',SOLV,'--cache-via',VIA,'--out',out]+sum([['--env',e] for e in extra],[]))
RA.configure_environment(args)
import metag.energetics.uma as uma, metag.pipeline as P
from metag.energetics import species_cache
from metag.energetics.conformers import spin_multiplicity
uma.DEV='cpu'
table,_=RA.load_species_table([CACHE],dict(P._IMPLICIT_SETTINGS),VIA,species_cache.canonical)
from rdkit import Chem as _Ch
HBF=P._ACID_HB_FILTER
table_hb=RA.load_species_table([CACHE],dict(P._IMPLICIT_SETTINGS,acid_hb_filter=P._ACID_HB_KEY),VIA,species_cache.canonical)[0] if HBF else {}
def _elig(smi,q):
    if not HBF or int(q)!=0: return False
    g=P._acid_groups(_Ch.AddHs(_Ch.MolFromSmiles(smi))); return len(set(g[0].values()))>=2
used={}
def cached(pu,q,smi,*a,**k):
    key=(species_cache.canonical(smi),int(q),spin_multiplicity(smi,q))
    T=table_hb if _elig(smi,q) else table
    if key not in T:
        if f'{smi} q{int(q)}' in REARR: raise P.SpeciesRearranged('sp', smi, 1)
        raise RA.CacheMiss(f'{smi} q{q}')
    used[(smi,q)]=T[key][0]; return T[key]
import glob as _gl
REARR=set()
for _f in _gl.glob(os.path.join(os.path.dirname(CACHE),'done*','*.json')):
    try:
        _r=json.load(open(_f))
        if 'SpeciesRearranged' in _r.get('error',''): REARR.add(_r['item'])
    except Exception: pass
P.implicit_G=cached
wg=float(json.load(open(os.environ.get('RS_WATER') or os.path.join(os.path.dirname(os.path.abspath(__file__)),'provenance','')+('water_ref_G.json' if SOLV=='cosmo' else f'water_ref_G_{SOLV}.json')))['G'])
P.water_ref_G=lambda pu,log=None: wg
rx=json.load(open(REX)); res={}; miss={}
for rid,r in rx.items():
    used.clear()
    rin=dict(r,species={k:tuple(v) for k,v in r['species'].items()})
    try: s=P.score_reaction(None,rin,log=lambda *x:None,key=rid)
    except RA.CacheMiss as e: miss[rid]=str(e); continue
    if s is None or s.get('dG') is None: miss[rid]='none'; continue
    import statistics
    exp=statistics.median(r['exp'])
    res[rid]=dict(dG=s['dG'],dG_raw=s['dG_raw'],exp=exp,err=s['dG']-exp,stages=s.get('stages'),cls=s['sigma_breakdown'].get('class'),
      anchor=s['anchor'],routes=s['routes'],sp=s['species_scored'],G={f'{k[0]}|{k[1]}':v for k,v in used.items()},note=r.get('note',''),nh=r.get('n_Hplus'))
json.dump(dict(res=res,miss=miss),open(out,'w'),indent=0)
import numpy as np
e=np.array([v['err'] for v in res.values()]); print(len(res),'miss',len(miss),'MAE %.2f med %.2f bias %.2f'%(np.abs(e).mean(),np.median(np.abs(e)),e.mean()))
