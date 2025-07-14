import sympy as sp
import numpy as np
from itertools import accumulate
from scipy.linalg import eig

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

# l1, l2, l3, l4, l5, l6 = sp.symbols('l1 l2 l3 l4 l5 l6')
# l = [l1, l2, l3, l4, l5, l6]
# 
# lc1, lc2, lc3, lc4, lc5, lc6 = sp.symbols('lc1 lc2 lc3 lc4 lc5 lc6')
# lc = [lc1, lc2, lc3, lc4, lc5, lc6]
# 
# α1, α2, α3, α4, α5, α6 = sp.symbols('α1 α2 α3 α4 α5 α6')
# α = [α1, α2, α3, α4, α5, α6]
# 
# αd1, αd2, αd3, αd4, αd5, αd6 = sp.symbols('α.1 α.2 α.3 α.4 α.5 α.6')
# αd = [αd1, αd2, αd3, αd4, αd5, αd6]
# 
# β1, β2, β3, β4, β5, β6 = sp.symbols('β1 β2 β3 β4 β5 β6')
# β = [β1, β2, β3, β4, β5, β6]
# 
# Ia, I2, I3, I4, I5, I6 = sp.symbols('Ia I2 I3 I4 I5 I6')
# I = [Ia, I2, I3, I4, I5, I6]
# 
# m1, m2, m3, m4, m5, m6 = sp.symbols('m1 m2 m3 m4 m5 m6')
# m = [m1, m2, m3, m4, m5, m6]
# 
# k1, k2, k3, k4, k5, k6 = sp.symbols('k1 k2 k3 k4 k5 k6')
# k = [k1, k2, k3, k4, k5, k6]

links = 5

l = [sp.symbols(f"l{i+1}") for i in range(links)]
lc = [sp.symbols(f"lc{i+1}") for i in range(links)]
α = [sp.symbols(f"α{i+1}") for i in range(links)]
αd = [sp.symbols(f"αd{i+1}") for i in range(links)]
αdd = [sp.symbols(f'α..{i+1}') for i in range(links)]
β = [sp.symbols(f'β{i+1}') for i in range(links)]  
I = [sp.symbols(f'I{i+1}') for i in range(links)]  
m = [sp.symbols(f'm{i+1}') for i in range(links)]  
k = [sp.symbols(f'k{i+1}') for i in range(links)]  

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
# vs_expanded = [sp.expand_trig(vi) for vi in vs]
# 
# # (b) Then apply trig‐simplification:
# vs_simplified = [sp.trigsimp(vi) for vi in vs_expanded]
# 
# # Print results:
# # for i, v in enumerate(vs_simplified):
# #     print(f"v_{i} simplified = \n{v}\n")
#     
# # Substituindo a versão simplificada na saída da
# # função
# vs = vs_simplified

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
#E_expanded = [sp.expand_trig(ei) for ei in E]
#
## (b) Then apply trig‐simplification:
#E_simplified = [sp.trigsimp(ei) for ei in E_expanded]
#
## Print results:
## for i, e in enumerate(E_simplified):
##     print(f"E_{i} simplified = \n{e}\n")
#
## Substituindo a versão simplificada na saída da
## função
#E = sum(E_simplified)

E = sum(E)

# print(f"A energia cinética total E é: \n{E}\n")

# E = sp.expand(E)
# E = sp.collect(E, [αd1, αd2])
# print(f"A energia cinética total E agrupada é: \n{E}\n")
# Corresponde exatamente à equação (24)

# ============================================================
# CHECK - OK - OK2 ===========================================
# ============================================================

      
# Cálculo da energia potecial

# Redefinindo a deflexão estática para facilitar o debugging
# e checar a aplicação das condições de contorno

αst1, αst2, αst3, αst4, αst5, αst6  = sp.symbols('αst1 αst2 αst3 αst4 αst5 αst6')
αst = [αst1, αst2, αst3, αst4, αst5, αst6]

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

#print(f"Energia potencial total V: \n {V_tot} \n")

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
#print(f"Energia potencial total V")
#print(f"com substituição das Eq. (7) e (8): \n {V_tot}\n")

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
# print(f"Energia potencial total V com αst² zerados: \n {V_tot} \n")

# Substituindo as deflexões estáticas

def αst_substitution(links, αst, α_st):
    subs_αst = {}
    for i in range(links):
        subs_αst[αst[i]] = α_st[i]
    return subs_αst

subs_αst = αst_substitution(links, αst, α_st)
V_tot = V_tot.subs(subs_αst)
#print(f"Energia potencial total V com αst calculados substituidos: \n {V_tot} \n")

V_tot = sp.simplify(V_tot)
# print(f"Energia potencial total V simplificada: \n {V_tot} \n")

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

# L = sp.collect(L, [αd1, αd2])

# print(f"A função de Lagrange é:\n{l}\n")

# ------------------- IGUAL À EQUAÇÃO (25) -------------------

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

        # eq = sp.simplify(ddt_dL_dαd - dL_dα)
        eq = ddt_dL_dαd - dL_dα
        eqs.append(eq)
    return eqs

EL_eqs = euler_lagrange_eqs(L, α[:links], αd[:links], αdd)

# E = sp.collect(E, [αd1, αd2, αdd[0], αdd[1]])

# for i, eq in enumerate(EL_eqs):
#     print(f"Euler–Lagrange equation {i+1}:")
#     print(eq)
    
# ------------ IGUAIS ÀS EQUAÇÕES (26) E (27) ------------

# Zerando α e αd das equações de Euler-Lagrange
def zero_EL(α,αd,EL_eqs):
    α_zero = {v: 0 for v in α}
    αd_zero = {v: 0 for v in αd}
    subs = {**α_zero, **αd_zero}
    EL_eqs_subs = [eq.subs(subs) for eq in EL_eqs]
    return EL_eqs_subs

M_eqs = zero_EL(α,αd,EL_eqs)

# for i, eq in enumerate(M_eqs):
#     print(f"\nM matrix equations {i+1}:\n")
#     print(eq)

M, _ = sp.linear_eq_to_matrix(M_eqs, αdd[:links])
# NOTA: o alpha foi completamente zerado, portanto é possível
# utilizar na Eq.(40)

K = sp.diag(*k[:links])

# ------------ IGUAIS ÀS EQUAÇÕES (32) E (38) ------------

# Substituindo os valores obtidos do CAD
k1 = 6.70e4 # N*m/rad
k2 = 3.10e4 # N*m/rad
k3 = 1.60e4 # N*m/rad
k4 = 0.47e4 # N*m/rad
k5 = 0.47e4 # N*m/rad

I1 = 237510.956e-4  #kg*m² #calculado em relação a A
Is2 = 43253.680e-4 #kg*m²
Is3 = 30574.85e-4  #kg*m²
Is4 = 1145e-4      #kg*m²
Is5 = 6328e-4      #kg*m²

m1 = 6458.27e-3 # kg
m2 = 4574.25e-3 # kg
m3 = 4018e-3    # kg
m4 = 584e-3     # kg
m5 = 2868e-3    # kg

l1 = 251.00e-3 #m
l2 = 215.95e-3 #m
l3 = 252.77e-3 #m
l4 = 90.00e-3  #m
l5 = 100.00e-3 #m

lc1 = 155.59e-3 #m
lc2 = 137.80e-3 #m
lc3 = 137.01e-3 #m
lc4 = 42.26e-3  #m
lc5 = 17.07e-3  #m

β1 = 0
β2 = 0
β3 = 0
β4 = 0
β5 = 0
β6 = 0

k_values = [k1, k2, k3, k4, k5]  # N*m/rad

I_values = [I1, Is2, Is3, Is4, Is5]  # kg*m²

m_values = [m1, m2, m3, m4, m5]  # kg

l_values = [l1, l2, l3, l4, l5]

lc_values = [lc1, lc2, lc3, lc4, lc5]

β_values = [β1, β2, β3, β4, β5]

float_subs = {}
float_subs.update({li: val for li, val in zip(l, l_values)})
float_subs.update({lci: val for lci, val in zip(lc, lc_values)})
float_subs.update({mi: val for mi, val in zip(m, m_values)})
float_subs.update({ki: val for ki, val in zip(k, k_values)})
float_subs.update({Ii: val for Ii, val in zip(I, I_values)})
float_subs.update({βi: val for βi, val in zip(β, β_values)})

M = M.subs(float_subs).evalf()
K = K.subs(float_subs).evalf()

# --------------------------------------------------------

# Convert SymPy matrices to NumPy arrays
M_np = np.array(M.tolist(), dtype=np.float64)
K_np = np.array(K.tolist(), dtype=np.float64)

print(f"A matriz de massa M é: \n{M_np}\n")
print(f"A matriz de rigidez K é: \n{K_np}\n")

# Solve the generalized eigenvalue problem Kx = λMx
eigvals, eigvecs = eig(K_np, M_np)

# Get only real parts (eigenvalues may be complex due to round-off)
eigvals_real = np.real(eigvals)

# Compute natural frequencies in rad/s
frequencies_rad = np.sqrt(eigvals_real)
print("Natural frequencies f (rad/s):")
for i, f in enumerate(frequencies_rad):
    print(f"f_{i+1} = {f:.8f} Hz")

# Compute natural frequencies in Hz
frequencies_Hz = frequencies_rad / (2 * np.pi)

# Sort frequencies
frequencies_Hz.sort()

# Print results
print("Natural frequencies f (Hz):")
for i, f in enumerate(frequencies_Hz):
    print(f"f_{i+1} = {f:.8f} Hz")