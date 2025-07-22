import MZ_func as mz

#modificando para test de commit

# Definir os número de links
links = 5

# Incluir valores de projeto CAD

# Rigidez em N*m/rad
k1,k2,k3,k4,k5 = (6.70e4, 3.10e4, 1.60e4, 0.47e4, 0.47e4)

# Momentos de inércia de massa kg*m²
Ia,Is2,Is3,Is4,Is5 = (237510.956e-4, 43253.680e-4, 30574.85e-4,
                      1145e-4, 6328e-4)

# Massas dos links em kg
m1,m2,m3,m4,m5 = (6458.27e-3, 4574.25e-3, 4018e-3, 584e-3, 2868e-3)    

# Comprimentos dos links em m
l1,l2,l3,l4,l5 = (251.00e-3, 215.95e-3, 252.77e-3, 90.00e-3,
                  100.00e-3)

# Distância dos CMs para o link respectivo
lc1,lc2,lc3,lc4,lc5 = (155.59e-3, 137.80e-3, 137.01e-3,
                       42.26e-3, 17.07e-3)

# Ângulos de posição das juntas
β1,β2,β3,β4,β5 = (0,0,0,0,0)

k_values = [k1, k2, k3, k4, k5]  # N*m/rad
I_values = [Ia, Is2, Is3, Is4, Is5]  # kg*m²
m_values = [m1, m2, m3, m4, m5]  # kg
l_values = [l1, l2, l3, l4, l5] # m
lc_values = [lc1, lc2, lc3, lc4, lc5] # m
β_values = [β1, β2, β3, β4, β5] # rad

frequencies_rad, frequencies_Hz = mz.freq_calculation(links,m_values,k_values,I_values,
                                                      l_values,lc_values,β_values)

# Print dos resultados
print("Natural frequencies f (rad/s):")
for i, f in enumerate(frequencies_rad):
    print(f"f_{i+1} = {f:.8f} Hz")

# Print dos resultados
print("Natural frequencies f (Hz):")
for i, f in enumerate(frequencies_Hz):
    print(f"f_{i+1} = {f:.8f} Hz")
