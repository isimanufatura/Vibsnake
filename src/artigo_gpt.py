import numpy as np

# Functions as before
def static_deformations(m1, m2, g, lc1, l1, lc2, k1, k2, beta1, beta2):
    alpha2_st = - (m2 * g * lc2 * np.cos(beta1 + beta2)) / k2
    alpha1_st = - ((m1 * g * lc1 + m2 * g * l1) * np.cos(beta1) +
                   m2 * g * lc2 * np.cos(beta1 + beta2)) / k1
    return alpha1_st, alpha2_st

def potential_energy(alpha1, alpha2, alpha1_st, alpha2_st,
                     k1, k2, m1, m2, g, lc1, l1, lc2, beta1, beta2):
    Vk1 = 0.5 * k1 * (alpha1 + alpha1_st)**2
    Vk2 = 0.5 * k2 * (alpha2 + alpha2_st)**2
    Vg1 = m1 * g * lc1 * np.sin(beta1 + alpha1)
    Vg2 = m2 * g * (l1 * np.sin(beta1 + alpha1) +
                    lc2 * np.sin(beta1 + beta2 + alpha1 + alpha2))
    return Vk1 + Vk2 + Vg1 + Vg2

def kinetic_energy(alpha1_dot, alpha2_dot,
                   IA1, IS2, m2, l1, lc2, beta2, alpha2):
    q1dot = alpha1_dot
    q2dot = alpha2_dot
    vS2_sq = (
        l1**2 * q1dot**2
        + lc2**2 * (q1dot + q2dot)**2
        + 2 * l1 * lc2 * q1dot * (q1dot + q2dot) * np.cos(beta2 + alpha2)
    )
    E1 = 0.5 * IA1 * q1dot**2
    E2 = 0.5 * m2 * vS2_sq + 0.5 * IS2 * (q1dot + q2dot)**2
    return E1 + E2

# Example parameters
m1, m2 = 40.0, 50.58
g = 9.80665
lc1, l1, lc2 = 0.315, 0.63, 0.1
k1, k2 = 555000.0, 138000.0
IA1, IS2 = 0.55, 0.0009
beta1, beta2 = 2*np.pi/3, np.pi/4
alpha1, alpha2 = 0.01, -0.005
alpha1_dot, alpha2_dot = 0.2, -0.1

# Compute
alpha1_st, alpha2_st = static_deformations(m1, m2, g, lc1, l1, lc2, k1, k2, beta1, beta2)
V = potential_energy(alpha1, alpha2, alpha1_st, alpha2_st,
                     k1, k2, m1, m2, g, lc1, l1, lc2, beta1, beta2)
E = kinetic_energy(alpha1_dot, alpha2_dot, IA1, IS2, m2, l1, lc2, beta2, alpha2)

print(f"Static deformations: α1_st = {alpha1_st:.6f} rad, α2_st = {alpha2_st:.6f} rad")
print(f"Potential energy V = {V:.4f} J")
print(f"Kinetic energy E = {E:.4f} J")

f = (1 / (2 * np.pi)) * np.sqrt(V / E)
print(f"Frequency f = {f:.4f} Hz")