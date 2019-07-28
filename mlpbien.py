import numpy as np

class layer:
    def __init__(self,n_capa,n_ent,n_neur,f):
        self.n_capa = n_capa
        self.f = f
        self.n_ent = n_ent
        self.n_neur = n_neur
        self.w_init()
        self.b_init()
        self.d_l  = np.zeros(n_neur)

    def w_init(self):
        wpn = list() #Pesos por neurona
        for i in range(self.n_neur): #Por cada neurona n pesos diferentes por las entradas
            wpn.append(np.random.random(self.n_ent))
        self.wpn = wpn
    
    def b_init(self):
        bpn = list()
        for i in range(self.n_neur):
            bpn.append(np.random.random())
        self.bpn = bpn

    def foward(self,inputs,p=False):
        inputs = inputs.ravel()
        self.inps = inputs
        ins = inputs #una vector horizontal con las caracteristicas de entrada

        netas = list() #En esta lista se van a guardar todas las salidas 
                       #netas osease antes de la activacion
        actvs = list() #En esta lista se van a guardar todas las 
                       #activaciones de las netas 
                       #Osea las salidas de las neuronas

        nets = np.zeros(self.n_neur)
        out  = np.zeros(self.n_neur)

        if p:
            print("Capa %d:" % self.n_capa)
        for i in range(self.n_neur): 
            #Para cada neurona
            if p:
                print("\nNeurona %d:" % i)
                print("Entrada: %s\n" % str(ins.shape), ins)
                print("Pesos: %s\n" % str(self.wpn[i].shape), self.wpn[i])
                print("bias: ", self.bpn[i])
            netas.append((np.sum(ins*self.wpn[i])+self.bpn[i]))
            if p:
                print(netas)
                print("Salida neta: %s\n" %str(netas[-1].shape), netas[-1])
            actvs.append(self.f[0](netas[-1]))
            if p:
                print("Activacion: %s\n" %str(actvs[-1].shape), actvs[-1])
            nets[i] = netas[-1]
            out[i] = actvs[-1]

        if p:
            print("\nSalida final: %s\n" %str(out.shape), out)
        self.out = out
        self.nets = nets
        self.netas = netas
        self.actvs = actvs
        return out

class MLP:
    def __init__(self, topo, fs,ff):
        self.topo = topo
        self.fs = fs
        self.nIns = topo[0]
        self.nOuts = topo[-1]
        self.layers(topo)
        self.ff = ff

    def layers(self,topo):
        lyrs = []
        for i in range(1,len(topo)):
            l = layer(i,topo[i-1],topo[i],self.fs[i])
            lyrs.append(l)
        self.lyrs = lyrs

    def foward(self,x, p=False):
        ls = self.lyrs
        inp = x
        for i in range(len(ls)):
            l = ls[i]
            inp = l.foward(inp,p=p)
        return inp


    def fit(self,x,d,epo,alpha,p=False):
        lyrs = self.lyrs

        if p:
            print(x)
            print(d)
        for j in range(epo):

            r = np.arange(x.shape[0])
            np.random.shuffle(r)

            print("\n\nEpo %d:" % j)

            for k in r:
                yob = self.foward(x[k],p=p)
                err = y[k] - self.ff(yob)
                if True:
                    print("d:%d ob:%d" % (y[k],self.ff(yob)))
                    print("Err:")
                    print(y[k] - yob)
                
                #Por cada capa hacia atras:
                for l in reversed(range(len(lyrs))):
                    if l == len(lyrs)-1:
                        ol = lyrs[l]
                        for i,ys in enumerate(yob):
                            ol.d_l[i] = ol.f[1](ys)*err[i]
                        ol.p_wpn = ol.wpn.copy()
                        #Modificando pesos:
                        for i in range(ol.n_neur):
                            for ii in range(ol.n_ent):
                                ol.wpn[i][ii] +=  alpha*ol.d_l[i]*ol.inps[ii]
                        if p:
                            print("Nuevos pesos de capa de salida:\n" , ol.wpn)

                    else:
                        cl = lyrs[l]
                        pl = lyrs[l+1]
                        cl.p_wpn = cl.wpn.copy()
                        #Calculo de deltas
                        for i,ys in enumerate(cl.out):
                            cl.d_l[i] = cl.f[1](ys)*(pl.p_wpn[0][i]*pl.d_l[0])
                            print("delta_h[%d]" % i ,cl.d_l[i])
       
                        for i in range(cl.n_neur):
                            for ii in range(cl.n_ent):
                                cl.wpn[i][ii] += (alpha*cl.inps[ii]*cl.d_l[i])
                        print("Pesos de hl:\n" , cl.wpn)




uni = (lambda x : x,
       lambda x : np.ones(x.shape,np.float))

rou = lambda x : np.round(x)

sigm = (lambda x : 1/(1+np.exp(-x)),
        lambda x : x * (1 -x))


if False:
    ins = np.random.random((6,1))
    ds = np.hstack([ins,ins+1])

    topo = [1,5,2]
    fs = [0,sigm,sigm]


    z = MLP(topo,fs)
    z.fit(ins,ins,10,0.01,True)

import sys
args = sys.argv
epos = int(args[1])
alpha = float(args[2])

x = np.array([[0.,1.],
              [1.,0.],
              [1.,1.],
              [0.,0.]])
y = np.array([[1.],
              [1.],
              [0.],
              [0.]])


print("Conjunto de entrenamiento:\n  x:   y:")
for X,Y in zip(x,y):
    print(X,Y)

print("Taza de aprendizaje:\n" , alpha)

print("Numero de epocas:",epos)

topo = [2,2,1]
fs = [None,sigm,sigm]
z = MLP(topo,fs,rou)
z.fit(x,y,epos,alpha)

for ii,i in enumerate(x):
    print(rou(z.foward(i)), y[ii])

