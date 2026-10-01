import json,sys,numpy as np,collections
A=json.load(open(sys.argv[1]))['res']; B=json.load(open(sys.argv[2]))['res']
com=sorted(set(A)&set(B)); print('common',len(com))
ea=np.array([A[r]['err'] for r in com]); eb=np.array([B[r]['err'] for r in com])
print('A MAE %.2f med %.2f bias %.2f | B MAE %.2f med %.2f bias %.2f | tail20 %d vs %d'%(abs(ea).mean(),np.median(abs(ea)),ea.mean(),abs(eb).mean(),np.median(abs(eb)),eb.mean(),(abs(ea)>20).sum(),(abs(eb)>20).sum()))
def rc(x):
    return '+'.join(k[:5] for k in ['cofactor_ring','truncated','ph0'] if x['routes'].get(k)) or 'full'
for key in ('cls','route'):
  g=collections.defaultdict(list)
  for r in com: g[A[r]['cls'] if key=='cls' else rc(A[r])].append((A[r]['err'],B[r]['err']))
  for k,v in sorted(g.items(),key=lambda x:-len(x[1])):
    v=np.array(v); print('  %-26s n=%3d  A %5.1f (%+5.1f)  B %5.1f (%+5.1f)'%(k,len(v),abs(v[:,0]).mean(),v[:,0].mean(),abs(v[:,1]).mean(),v[:,1].mean()))
if len(sys.argv)>3:
  d=sorted(com,key=lambda r:abs(B[r]['err'])-abs(A[r]['err']))
  print('most improved:'); [print('  %s %6.1f -> %6.1f %s'%(r,A[r]['err'],B[r]['err'],A[r]['note'][14:60])) for r in d[:15]]
  print('most worsened:'); [print('  %s %6.1f -> %6.1f %s'%(r,A[r]['err'],B[r]['err'],A[r]['note'][14:60])) for r in d[-15:]]
