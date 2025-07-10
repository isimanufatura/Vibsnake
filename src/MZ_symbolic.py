import sympy as sp
import numpy as np
from itertools import accumulate

'''
# Example cumulative sum of symbolic
from sympy import symbols
from itertools import accumulate

# Define symbolic variables
x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
sym_list = [x1, x2, x3, x4]

# Compute cumulative sum
cumsum = list(accumulate(sym_list))

# Output
for i, expr in enumerate(cumsum):
    print(f"Cumulative sum at index {i}: {expr}")

------------------------------------------------------
# Example of sum of two symbolic lists, element-wise
from sympy import symbols

# Define symbolic variables
a1, a2, a3 = symbols('a1 a2 a3')
b1, b2, b3 = symbols('b1 b2 b3')

# Create two lists
A = [a1, a2, a3]
B = [b1, b2, b3]

# Element-wise sum
C = [ai + bi for ai, bi in zip(A, B)]

print(C)    
'''

g = sp.symbols('g')

l1, l2, l3 = sp.symbols('l1 l2 l3')
l = [l1, l2, l3 ]

lc1, lc2, lc3 = sp.symbols('lc1 lc2 lc3')
lc = [lc1, lc2, lc3]

α1, α2, α3 = sp.symbols('α1 α2 α3')
α = [α1, α2, α3]

β1, β2, β3 = sp.symbols('β1 β2 β3')
β = [β1, β2, β3]

Ia, I2, I3 = sp.symbols('Ia I2 I3')
I = [Ia, I2, I3]

m1, m2, m3 = sp.symbols('m1 m2 m3')
m = [m1, m2, m3]

k1, k2, k3 = sp.symbols('k1 k2 k3')
k = [k1, k2, k3]

links = 2
'''
# Cálculo velocidades (eq. (22) e (23))

def velocity(links, l, lc, α, β):
    α_s = list(accumulate(α))
    β_s = list(accumulate(β))
    θ = [ai + bi for ai, bi in zip(α_s, β_s)]
    xs1 = 0
    ys1 = 0
    xs = [0]
    ys = [0]
    vs = [0]
    for i in range(links-1):
        xs1 += - l[i] * α_s[i] * sp.sin(θ[i])
        xi = xs1 - lc[i+1] * α_s[i+1] * sp.sin(θ[i+1])
        xs.append(xi)
        ys1 += l[i] * α_s[i] * sp.cos(θ[i])
        yi = ys1 + lc[i+1] * α_s[i+1] * sp.cos(θ[i+1])
        ys.append(yi)
        vi = xi**2 + yi**2
        vs.append(vi)
    return xs,ys,vs

xs,ys,vs = velocity(links, l, lc, α, β)

# DEBUGGING PRINT

# for i,xi in enumerate(xs):
#     print(f"Termo {i} de xs:\n {xi}")
#     #sp.pretty_print(xi)
# for i,yi in enumerate(ys):
#     print(f"Termo {i} de ys:\n {yi}")
#     #sp.pretty_print(yi)
# for i,vi in enumerate(vs):
#     print(f"Termo {i} de vs:\n {vi}")
#     #sp.pretty_print(vi)
    
# (a) First expand all powers/products of sin/cos:
vs_expanded = [sp.expand_trig(vi) for vi in vs]

# (b) Then apply trig‐simplification:
vs_simplified = [sp.trigsimp(vi) for vi in vs_expanded]

# Print results:
# for i, v in enumerate(vs_simplified):
#     print(f"v_{i} simplified = {v}")
    
# Substituindo a versão simplificada na saída da
# função
vs = vs_simplified

# ============================================================
# CHECK - OK =================================================
# ============================================================

# Cálculo energia cinética
# (Eq. (16),(17),(18),(24))

def kinetic_energy(links, vs, α, I, m):
    α_s = list(accumulate(α))
    E = []
    for i in range(links):
        e1 = 1/2*m[i]*vs[i]
        e2 = 1/2*I[i]*α_s[i]**2
        ei = e1 + e2
        E.append(ei)
    return E

E = kinetic_energy(links, vs, α, I, m)

# for i,ei in enumerate(E):
#     print(f"Termo {i} de E:\n {ei}")
    
# (a) First expand all powers/products of sin/cos:
E_expanded = [sp.expand_trig(ei) for ei in E]

# (b) Then apply trig‐simplification:
E_simplified = [sp.trigsimp(ei) for ei in E_expanded]

# Print results:
# for i, e in enumerate(E_simplified):
#     print(f"v_{i} simplified = {e}")

# Substituindo a versão simplificada na saída da
# função
E = E_simplified

# ============================================================
# CHECK - OK =================================================
# ============================================================
'''

# Cálculo da deflexão estática

def static_deflection(links, g, m, l, lc, β):
    # Reverse the lists so we loop from the end-effector backward
    m_l  = m[0:links]
    l_l  = l[0:links]
    lc_l = lc[0:links]
    β_l  = list(accumulate(β))
    β_l = β_l[0:links]
    k_l  = k[0:links]
    
    m_r  = m_l[::-1]
    l_r  = l_l[::-1]
    lc_r = lc_l[::-1]
    β_s = β_l[::-1]
    k_r  = k_l[::-1]
    
    α_st = []
    for i in range(links):
        if i == 0:
            num = - (m_r[i]*g*lc_r[i]) * sp.cos(β_s[i])
            αi = num / k_r[i] 
        else:
            num -= (m_r[i]*g*lc_r[i] + m_r[i-1]*g*l_r[i]) * sp.cos(β_s[i])
            αi = num / k_r[i] 
        α_st.append(αi)
        
    α_st.reverse()
    return(α_st)

α_st = static_deflection(links, g, m, l, lc, β)

for i,αi in enumerate(α_st):
    print(f"Termo {i} de E:\n {αi}")
    
# ============================================================
# CHECK - Possível fonte de erro =============================
# ============================================================
        
# Cálculo da energia potecial

# Redefinindo a deflexão estática para facilitar o debugging
# e checar a aplicação das condições de contorno

αst1, αst2, αst3 = sp.symbols('αst1 αst2 αst3')
αst = [αst1, αst2, αst3]

def potential_energy(links, g, m, l, lc, β, α, αst,k):
    α_s = list(accumulate(α))
    β_s = list(accumulate(β))
    θ = [ai + bi for ai, bi in zip(α_s, β_s)]
    vk = []
    vg = []
    vgxi = 0
    for i in range(links):
        vki = 1/2 * k[i] * (α[i] + αst[i])
        if i == 0:
            vgi = m[i]*g * lc[i]*sp.sin(θ[i])
        else:
            vgxi += l[i-1]*sp.sin(θ[i-1])
            vgi = m[i]*g * (vgxi + lc[i]*sp.sin(θ[i]))
        vk.append(vki)
        vg.append(vgi) 
    return vk,vg

vk,vg = potential_energy(links, g, m, l, lc, β, α, αst, k)

for i,vi in enumerate(vk):
    print(f"Termo {i} de Vk:\n {vi}")
    
for i,vi in enumerate(vg):
    print(f"Termo {i} de Vg:\n {vi}")

# ============================================================
# CHECK - OK =================================================
# ============================================================









# Cálculo energia potencial
# (Eq. (3),(4),(5),(6) e (14))


