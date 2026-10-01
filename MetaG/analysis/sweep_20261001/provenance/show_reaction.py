import json,sys
d=json.load(open(sys.argv[1]))['res']
rx=json.load(open('/homes/rzhu/ModelSEED_FAISS/thermodynamic_calc/experiments/qm_mlip_solvation/scripts/reactions_tecrdb_std.json'))
for rid in sys.argv[2:]:
    v=d[rid]; print('=====',rid,v['note'][14:80],'exp %.1f dG %.1f err %.1f'%(v['exp'],v['dG'],v['err']))
    print('  stages',v['stages']); print('  routes',{k:x for k,x in v['routes'].items() if x and k not in('errors','warnings','reverted')})
    print('  INPUT:'); [print('    %+d q%+d %s'%(c,q,s)) for n,(c,q,s) in rx[rid]['species'].items()]; print('    nH+',rx[rid].get('n_Hplus'))
    print('  SCORED:'); [print('    %+d q%+d %s'%(c,q,s)) for n,(c,q,s) in v['sp'].items()]
