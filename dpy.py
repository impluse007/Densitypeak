import numpy as np
import os
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt

def calcdisdc(X,pencent):
    lx=len(X)
    dist=np.zeros((lx,lx))
    for i in range(lx):
        for j in range(lx):
            dist[i,j]=np.sqrt(np.sum(np.square(X[i]-X[j])))
    pos=int(np.ceil(0.5*pencent*lx*(lx-1)))
    print pos,len(dist)
    dc=np.sort(np.concatenate(dist))[pos]
    print dc
    return dist,dc

def calrho(dist,dc):
    len_rho=len(dist)
    rho=np.zeros(len_rho)
    for i in range(len_rho):
        rho[i]=np.sum(np.exp(-np.square(dist[i,:]/dc)))
    return rho

def caldelta(rho,dist):
    len_delta=len(rho)
    delta=np.ones(len_delta)*np.inf
    q=np.arange(len_delta)
    for i in range(len_delta):
        for j in range(len_delta):
            if (rho[j]>rho[i])&(dist[i,j]<delta[i]):
               delta[i]=dist[i,j]
               q[i]=j
    indexmax=np.argmax(delta)
    delta[indexmax]=dist[indexmax,:].max()
    return delta,q

def calcenters(gamma):
    x = np.flipud(np.argsort(gamma))
    y = np.flipud(np.sort(gamma))
    gamma_mean=gamma.mean()
    centers=[x[0],x[1]]
    for i in range(2, len(y) - 1):
        # if y[i] - y[i + 1] < (y[i - 1] - y[i]) / 2.:
        #     break
        if y[i]-gamma_mean<y[i-1]-y[i]:
            break
        centers.append(x[i])
    return centers



def plot(rho,delta,gamma,points,clusters):
    for cluster in clusters:
        plt.scatter(points[cluster][:,0],points[cluster][:,1],color=np.random.rand(3))



if __name__=='__main__':
    n_samples = 900
    random_state = 5
    centers = [[-0.6, -0.4], [-0.3, 0.5], [0.5, -0.4]]
    center_box = [-1, 1]
    X, y = make_blobs(n_features=2, n_samples=n_samples,
                      cluster_std=0.15, center_box=center_box,
                      centers=centers, random_state=random_state)

    m = np.array(((1, 1), (1, 3)))
    X = X.dot(m)
    pencent=0.015
    dist,dc=calcdisdc(X,pencent)
    rho=calrho(dist,dc)
    delta,q=caldelta(rho,dist)
    gamma=rho*delta
    centers=calcenters(gamma)
    clusters = np.array(centers).reshape(-1, 1).tolist()
    qc = np.copy(q)
    dp=list(X)
    for i in np.flipud(np.argsort(rho)):
       if i not in centers:
            if qc[i] not in centers:
                qc[i] = qc[qc[i]]
                clusters[centers.index(qc[i])].append(i)
                dp[i]=centers.index(qc[i])

    plot(rho,delta,gamma,X,clusters)
    plt.show()

   
