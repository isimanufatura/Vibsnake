import numpy as np

'''
def sum_theta(BETA,j):
    theta_p = 0
    for p in range(j+1):
        theta_p += BETA[p]
    return theta_p

def static_def(links,g, M,BETA,L,Lc):
    alpha_list = np.array([])
    for i in range(links):
        print("i",i)
        if i == 0:
            sum1 = 0
            print("Soma zero")
        else:
            for j in range(links):
                print('j',j)
                theta_p = sum_theta(BETA,j)
                print('theta',theta_p)
                a = L[j] * np.cos(theta_p)
                print('a',a)
                sum1 += a
                print('sum',sum1)
                
        sum2 = M[i]*g*sum1 + M[i]*g*Lc[i]*np.cos(BETA[i])
        print("sum2:\n", sum2)
        
        alpha = -1/K[i]*sum2
        print('Alpha:\n',alpha)
        alpha_list = np.append(alpha_list,alpha)
    
    return alpha_list  
    
g = 9.81
L = [2,4,6]
Lc = [1,2,3]
BETA = [0,1,2]
M = [1,1,1]
K = [1,2,3]
links = 2

alpha_list = static_def(links,g, M,BETA,L,Lc)

for i, ai in enumerate(alpha_list):
    print(f"Elo {i+1}:")
    print(ai)

'''
    
'''
Comentários:
A ideia foi boa, porém é necessário realizar algumas alterações como tirar a função
sum_theta, que pode ser substituida por soma cumulativa (np.cumsum)
'''


def static_deflections(n_links, g, M, BETA, L, Lc, K):
    """
    Compute static deflections alpha_st for each joint i in an n-link arm.
    n_links : int
        Number of links (and joints).
    g : float
        Gravitational acceleration.
    M : array_like of length n_links
        Link masses m[0],…,m[n_links-1].
    BETA : array_like of length n_links
        Nominal joint angles beta[0],…,beta[n_links-1].
    L : array_like of length n_links
        Link lengths ℓ[0],…,ℓ[n_links-1].
    Lc : array_like of length n_links
        Centre-of-mass offsets ℓ_c[0],…,ℓ_c[n_links-1].
    K : array_like of length n_links
        Joint stiffnesses k[0],…,k[n_links-1].
    """
    alpha_st = np.zeros(n_links)

    # Precompute cumulative sums of betas for link orientations θ_j
    thetas = np.cumsum(BETA)  # θ[j] = β[0]+...+β[j]

    for i in range(n_links):
        torque = 0.0
        print("i:\n",i)

        # Sum torque contributions from each link j ≥ i
        for j in range(i, n_links):
            # lever arm from joint i to CoM of link j:
            #   sum of full link-lengths ℓ[i]..ℓ[j-1], plus offset ℓ_c[j]
            print('j:\n',j)
            lever = np.sum(L[i:j]) + Lc[j]
            print("lever:\n",lever)
            a = M[j] * g * lever * np.cos(thetas[j])
            print('a: \n', a)
            torque += M[j] * g * lever * np.cos(thetas[j])
            print('Torque_j:\n',torque)
            
            
        print('Torque_i:\n',torque)

        # static deflection α_i,st = – (total torque) / k_i
        alpha_st[i] = - torque / K[i]

    return alpha_st

g = 9.81
L = [2,4,6]
Lc = [1,2,3]
BETA = [0,1,2]
M = [1,1,1]
K = [1,2,3]
links = 2

alpha_st = static_deflections(links, g, M, BETA, L, Lc, K)
for i, ai in enumerate(alpha_st):
    print(f"Elo {i+1}:")
    print(ai)