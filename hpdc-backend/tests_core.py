import numpy as np, json
from app.step_analyzer import analyze_mesh
from app.process_model import predict

def box(x0,y0,z0,x1,y1,z1,flip=False):
    v=np.array([[x,y,z] for x in(x0,x1) for y in(y0,y1) for z in(z0,z1)],float)
    q=[(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]
    f=[]
    for a,b,c,d in q: f+=[(a,b,c),(a,c,d)]
    f=np.array(f)
    return v,(f[:,::-1] if flip else f)

def merge(m1,m2):
    v=np.vstack([m1[0],m2[0]]); f=np.vstack([m1[1],m2[1]+len(m1[0])]); return v,f

# fix box winding by testing signed volume, then build 3 mm shell (open-top not modelled; closed hollow)
def outward(v,f):
    t=v[f]; sv=np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()
    return f if sv>0 else f[:,::-1]
for wall in (3.0, 1.8):
    o=box(0,0,0,100,80,40); o=(o[0],outward(*o))
    i=box(wall,wall,wall,100-wall,80-wall,40-wall); i=(i[0],outward(*i)[:,::-1])
    v,f=merge(o,i)
    g=analyze_mesh(v,f,face_count=24,bbox=(100,80,40),name="test_housing").to_dict()
    print(json.dumps(g)); print(json.dumps(predict(g),indent=1))
