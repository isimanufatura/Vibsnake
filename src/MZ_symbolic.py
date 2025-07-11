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

αd1, αd2, αd3 = sp.symbols('α.1 α.2 α.3')
αd = [αd1, αd2, αd3]

β1, β2, β3 = sp.symbols('β1 β2 β3')
β = [β1, β2, β3]

Ia, I2, I3 = sp.symbols('Ia I2 I3')
I = [Ia, I2, I3]

m1, m2, m3 = sp.symbols('m1 m2 m3')
m = [m1, m2, m3]

k1, k2, k3 = sp.symbols('k1 k2 k3')
k = [k1, k2, k3]

links = 2

# Cálculo velocidades (eq. (22) e (23))

def velocity(links, l, lc, α, αd, β):
    α_s = list(accumulate(α))
    αd_s = list(accumulate(αd))
    β_s = list(accumulate(β))
    θ = [ai + bi for ai, bi in zip(α_s, β_s)]
    xs1 = 0
    ys1 = 0
    xs = [0]
    ys = [0]
    vs = [0]
    for i in range(links-1):
        xs1 += - l[i] * αd_s[i] * sp.sin(θ[i])
        xi = xs1 - lc[i+1] * αd_s[i+1] * sp.sin(θ[i+1])
        xs.append(xi)
        ys1 += l[i] * αd_s[i] * sp.cos(θ[i])
        yi = ys1 + lc[i+1] * αd_s[i+1] * sp.cos(θ[i+1])
        ys.append(yi)
        vi = xi**2 + yi**2
        vs.append(vi)
    return xs,ys,vs

xs,ys,vs = velocity(links, l, lc,α, αd, β)

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
#     print(f"v_{i} simplified = \n{v}\n")
    
# Substituindo a versão simplificada na saída da
# função
vs = vs_simplified

# ============================================================
# CHECK - OK - OK2 ===========================================
# ============================================================

# Cálculo energia cinética
# (Eq. (16),(17),(18),(24))

# vs1, vs2 = sp.symbols('vs1**2 vs2**2')
# vs = [vs1, vs2]

def kinetic_energy(links, vs, αd, I, m):
    α_s = list(accumulate(αd))
    E = []
    for i in range(links):
        e1 = 1/2*m[i]*vs[i]
        e2 = 1/2*I[i]*α_s[i]**2
        ei = e1 + e2
        E.append(ei)
    return E

E = kinetic_energy(links, vs, αd, I, m)

# for i,ei in enumerate(E):
#     print(f"Termo {i} de E:\n {ei}")
    
# (a) First expand all powers/products of sin/cos:
E_expanded = [sp.expand_trig(ei) for ei in E]

# (b) Then apply trig‐simplification:
E_simplified = [sp.trigsimp(ei) for ei in E_expanded]

# Print results:
# for i, e in enumerate(E_simplified):
#     print(f"E_{i} simplified = \n{e}\n")

# Substituindo a versão simplificada na saída da
# função
E = sum(E_simplified)
print(f"A energia cinética total E é: \n{E}\n")

E = sp.expand(E)
E = sp.collect(E, [αd1, αd2])
print(f"A energia cinética total E agrupada é: \n{E}\n")
# Corresponde exatamente à equação (24)

# ============================================================
# CHECK - OK - OK2 ===========================================
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
    V = []
    vgxi = 0
    for i in range(links):
        vki = 1/2 * k[i] * (α[i] + αst[i])**2 # → OK
        if i == 0:
            vgi = m[i]*g * lc[i]*sp.sin(θ[i])
        else:
            vgxi += l[i-1]*sp.sin(θ[i-1])
            vgi = m[i]*g*vgxi + m[i]*g*lc[i]*sp.sin(θ[i])
        vk.append(vki)
        vg.append(vgi) 
        V.append(vki+vgi)
    return vk,vg,V

vk,vg,V = potential_energy(links, g, m, l, lc, β, α, αst, k)

# Prints
# for i,vi in enumerate(vk):
#     print(f"Termo {i} de Vk:\n {vi} \n")
#     
# for i,vi in enumerate(vg):
#     print(f"Termo {i} de Vg:\n {vi} \n")
    
V_tot = sum(V)
V_tot = sp.expand(V_tot)

print(f"Energia potencial total V: \n {V_tot} \n")

def create_subs_ang_V(links, α, β):
    α_s = list(accumulate(α))
    β_s = list(accumulate(β))
    θ = [ai + bi for ai, bi in zip(α_s, β_s)]
    subs = {}
    for i in range(links):
        subs[sp.sin(θ[i])] = sp.sin(β_s[i]) + α_s[i] * sp.cos(β_s[i])
    return subs

subs = create_subs_ang_V(links, α, β)

V_tot = V_tot.subs(subs)
print(f"Energia potencial total V")
print(f"com substituição das Eq. (7) e (8): \n {V_tot}\n")

# ------------------- IGUAL À EQUAÇÃO (9) --------------------



# ============================================================
# CHECK - OK =================================================
# ============================================================

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

# Prints
# for i,αi in enumerate(α_st):
#     print(f"Termo {i} de α_st:\n {αi}\n")
    
# Substituindo as deflexões estáticas ao quadrado

def zero_αst_squared(links, αst):
    αst_2 = [xi**2 for xi in αst]
    subs_zero_αst_squared = {}
    for i in range(links):
        subs_zero_αst_squared[αst_2[i]] = 0
    return subs_zero_αst_squared

subs_zero_αst_squared = zero_αst_squared(links, αst)
V_tot = V_tot.subs(subs_zero_αst_squared)
print(f"Energia potencial total V com αst² zerados: \n {V_tot} \n")

# Substituindo as deflexões estáticas

def αst_substitution(links, αst, α_st):
    subs_αst = {}
    for i in range(links):
        subs_αst[αst[i]] = α_st[i]
    return subs_αst

subs_αst = αst_substitution(links, αst, α_st)
V_tot = V_tot.subs(subs_αst)
print(f"Energia potencial total V com αst calculados substituidos: \n {V_tot} \n")

V_tot = sp.simplify(V_tot)
print(f"Energia potencial total V simplificada: \n {V_tot} \n")

# ------------------- IGUAL À EQUAÇÃO (14) -------------------
    
# ============================================================
# CHECK - Possível fonte de erro =============================
# ============================================================

V = V_tot

# ============================================================
# CHECK - OK =================================================
# ============================================================

L = E - V
# print(f"A função de Lagrange é:\n{L}\n")

L = sp.collect(L, [αd1, αd2])

print(f"A função de Lagrange é:\n{l}\n")

# ------------------- IGUAL À EQUAÇÃO (25) -------------------

αdd = [sp.symbols(f'α..{i+1}') for i in range(len(α))]  # second derivatives

def euler_lagrange_eqs(L, α, αd, αdd):
    eqs = []
    for i in range(len(α)):
        dL_dαd = sp.diff(L, αd[i])  # ∂L/∂α̇ᵢ
        dL_dα  = sp.diff(L, α[i])   # ∂L/∂αᵢ

        # Total derivative of ∂L/∂α̇ᵢ w.r.t. time
        ddt_dL_dαd = 0
        for j in range(len(α)):
            ddt_dL_dαd += sp.diff(dL_dαd, α[j])  * αd[j]   # ∂/∂αⱼ * α̇ⱼ
            ddt_dL_dαd += sp.diff(dL_dαd, αd[j]) * αdd[j]  # ∂/∂α̇ⱼ * α̈ⱼ

        eq = sp.simplify(ddt_dL_dαd - dL_dα)
        eqs.append(eq)
    return eqs

EL_eqs = euler_lagrange_eqs(L, α[:links], αd[:links], αdd)

E = sp.collect(E, [αd1, αd2, αdd[0], αdd[1]])

for i, eq in enumerate(EL_eqs):
    print(f"Euler–Lagrange equation {i+1}:")
    print(eq)
    
# ------------ IGUAIS ÀS EQUAÇÕES (26) E (27) ------------
