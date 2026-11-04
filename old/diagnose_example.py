import numpy as np, re
from pathlib import Path
import masses
root=Path(r'C:\Users\Mitchell\Desktop\CMA_examples')
ns={}
src=(root/'main.py').read_text().split('sym_sort =')[0]
src=src[src.index('def normalize'):]
def block_diag(*blocks):
    out=np.zeros((sum(b.shape[0] for b in blocks),sum(b.shape[1] for b in blocks)))
    i=j=0
    for b in blocks:
        out[i:i+b.shape[0],j:j+b.shape[1]]=b;i+=b.shape[0];j+=b.shape[1]
    return out
exec(src,dict(np=np,norm=np.linalg.norm,block_diag=block_diag),ns)
P=ns['Proj']
lines=(root/'zmat').read_text().split('cart begin')[1].split('cart end')[0].strip().splitlines()
x=np.array([[float(t) for t in l.split()[1:]] for l in lines]).ravel()
m=np.repeat([masses.get_mass(l.split()[0])/5.48579909065e-4 for l in lines],3)
inds=[list(map(lambda v:int(v)-1,l.split())) for l in (root/'zmat').read_text().split('ZMAT begin')[1].split('ZMAT end')[0].splitlines() if l.strip()]
def q(x):
    xyz=x.reshape(-1,3); vals=[]
    for ix in inds:
        if len(ix)==2: vals.append(np.linalg.norm(xyz[ix[0]]-xyz[ix[1]]))
        else:
            a=xyz[ix[0]]-xyz[ix[1]];b=xyz[ix[2]]-xyz[ix[1]]
            vals.append(np.arccos(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b)))
    return np.array(vals)@P
h=1e-4; eye=np.eye(18)*h
B=np.array([(q(x+d)-q(x-d))/(2*h) for d in eye]).T
B2=np.empty((12,18,18))
for i in range(18):
    for j in range(18): B2[:,i,j]=(q(x+eye[i]+eye[j])-q(x+eye[i]-eye[j])-q(x-eye[i]+eye[j])+q(x-eye[i]-eye[j]))/(4*h*h)
F=np.loadtxt(root/'fc_int_b.dat',skiprows=1).reshape(12,12)
steps=.01/np.max(P,axis=0)
g=np.zeros(12)
log=(root/'CMA.out').read_text(encoding='utf-8').split('Level B Frequencies:')[0]
for i,ep,em in re.findall(r'\[(\d+),\1\] Δ\+ = ([-\d.]+)\s+\[\d+,\d+\] Δ- = ([-\d.]+)',log):
    g[int(i)]=(float(ep)-float(em))/(2*steps[int(i)])
H=B.T@F@B
corr=np.einsum('r,rij->ij',g,B2)
# restrict Cartesian Hessian to vibrational tangent space
U=np.linalg.svd((B/np.sqrt(m)).T,full_matrices=False)[0]
def freq(H):
    w=np.linalg.eigvalsh(U.T@(H/np.sqrt(m[:,None]*m[None,:]))@U)
    return np.sign(w)*np.sqrt(abs(w))*219474.6313708
print('Masses amu',m[::3]*5.48579909065e-4)
print('Steps',steps)
print('Internal gradient',g)
print('Original',freq(H))
print('Corrected',freq(H+corr))
print('Target',[93.23,93.34,264.37,264.53,278.28,278.44,292.56,303.78,398.91,485.58,591.18,591.20])
pairs={}
for i,j,sgn,val in re.findall(r'\[(\d+),(\d+)\] Δ([+-]) = ([-\d.]+)',log): pairs[int(i),int(j),sgn]=float(val)
Fr=np.zeros_like(F)
for i,j,s in pairs:
    if s!='+': continue
    if i==j: v=(pairs[i,i,'+']+pairs[i,i,'-'])/steps[i]**2
    else: v=(pairs[i,j,'+']+pairs[i,j,'-']-pairs[i,i,'+']-pairs[i,i,'-']-pairs[j,j,'+']-pairs[j,j,'-'])/(2*steps[i]*steps[j])
    Fr[i,j]=Fr[j,i]=v
print('Maximum F reconstruction difference from rounded logged energies',np.max(abs(Fr-F)))
print('Reconstructed frequency max difference',np.max(abs(freq(B.T@Fr@B)-freq(H))))
print('Projected B rank',np.linalg.matrix_rank(B))
order=[(i,j) for i,j,s in pairs if s=='+']
all_en={}
max_res=0.
patterns={'CCSD(T)':r'!CCSD\(T\) total energy\s+(-?\d+\.\d+)','CCSD':r'CCSD total energy\s+(-?\d+\.\d+)','MP2':r'MP2 total energy:\s+(-?\d+\.\d+)','RHF':r'!RHF STATE 1.1 Energy\s+(-?\d+\.\d+)'}
for n in range(1,62):
    folder=root/'DispsB'/str(n)
    txt=(folder/'output.dat').read_text()
    all_en[n]={k:float(re.search(p,txt)[1]) for k,p in patterns.items()}
    geom=(folder/'input.dat').read_text().split('geometry = {')[1].split('}')[0].strip().splitlines()
    xd=np.array([[float(t) for t in l.split()[1:]] for l in geom]).ravel()
    desired=np.zeros(12)
    if n>1:
        i,j=order[(n-2)//2];desired[i]=steps[i];desired[j]=steps[j]
        desired*=1 if n%2==0 else -1
    max_res=max(max_res,np.max(abs(q(xd)-q(x)-desired)))
print('Largest input coordinate displacement residual',max_res)
for method in patterns:
    e0=all_en[1][method]; vals={}
    for k,(i,j) in enumerate(order):
        vals[i,j,'+']=all_en[2+2*k][method]-e0;vals[i,j,'-']=all_en[3+2*k][method]-e0
    FF=np.zeros((12,12))
    for i,j in order:
        if i==j: v=(vals[i,i,'+']+vals[i,i,'-'])/steps[i]**2
        else: v=(vals[i,j,'+']+vals[i,j,'-']-vals[i,i,'+']-vals[i,i,'-']-vals[j,j,'+']-vals[j,j,'-'])/(2*steps[i]*steps[j])
        FF[i,j]=FF[j,i]=v
    print(method,'frequencies',freq(B.T@FF@B))
    if method=='CCSD(T)': print('Full precision F discrepancy',np.max(abs(FF-F)))
bench=root/'Benchmarker'/'CMA_Benchmarker'/'4_G3_99'/'67_pcl5'
Hb=np.loadtxt(bench/'CCSD_T_TZ'/'fc.dat',skiprows=1).reshape(18,18)
Pc=np.linalg.pinv(B)@B
print('Benchmark projected Cartesian frequencies',freq(Pc.T@Hb@Pc))
Ai=np.linalg.pinv(B)
Fb=Ai.T@Hb@Ai
print('Symmetric stretch block current',F[np.ix_([0,2],[0,2])])
print('Symmetric stretch block benchmark',Fb[np.ix_([0,2],[0,2])])
