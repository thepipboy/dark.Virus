import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ========== AIRBORNE TRANSMISSION PARAMETERS ==========
total_population = 10000        # Total population
initial_infected = 5            # Initial infected individuals
recovery_days = 14              # Average recovery period (days)
R0 = 4.5                        # Basic reproduction number (COVID-19 Delta variant)
intervention_day = 30           # Day interventions start (e.g. masks/distancing)

# Airborne-specific factors (based on Nature [DOI:10.1038/s41598-021-84698-5])
aerosol_decay_rate = 0.63       # Virus half-life in air (hours^-1)
relative_humidity = 0.6         # RH impact on transmission
ventilation_rate = 4.0          # Air changes per hour (ACH)

# ========== MATHEMATICAL MODEL ==========
def airborne_sir_model(y, t, params):
    S, I, R = y
    N = S + I + R
    
    # Dynamic R0 based on interventions and environmental factors
    current_R0 = params['R0'] * np.exp(-params['intervention_strength'] * (t > params['intervention_day']))
    
    # Airborne transmission modifier (ventilation + humidity effects)
    env_factor = np.exp(-params['aerosol_decay_rate']/params['ventilation_rate']) * (0.8 + 0.2*np.sin(t/10))
    beta = current_R0 * env_factor / params['recovery_days']
    
    # ODE Equations
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - (1/params['recovery_days']) * I
    dRdt = (1/params['recovery_days']) * I
    
    return [dSdt, dIdt, dRdt]

# ========== SIMULATION SETUP ==========
params = {
    'R0': R0,
    'recovery_days': recovery_days,
    'intervention_day': intervention_day,
    'intervention_strength': 0.7,  # 70% reduction in transmission
    'aerosol_decay_rate': aerosol_decay_rate,
    'ventilation_rate': ventilation_rate
}

# Initial conditions [S0, I0, R0]
y0 = [total_population - initial_infected, initial_infected, 0]

# Time vector (days)
t = np.linspace(0, 180, 180)

# Solve ODE
solution = odeint(airborne_sir_model, y0, t, args=(params,))
S, I, R = solution.T

# ========== VISUALIZATION ==========
plt.figure(figsize=(12, 8))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.axvline(intervention_day, color='k', linestyle='--', label='Interventions Start')

# Environmental factors plot
plt.twinx()
plt.plot(t, (0.8 + 0.2*np.sin(t/10)), 'm--', alpha=0.5, label='Airborne Transmissibility')

plt.title('Airborne Virus Spread Dynamics with Interventions')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()

# ========== KEY OUTPUT METRICS ==========
peak_day = t[np.argmax(I)]
peak_infections = np.max(I)
print(f"Peak Infections: {peak_infections:.0f} people on day {peak_day}")
print(f"Total Infected: {np.max(R):.0f} people ({np.max(R)/total_population*100:.1f}% of population)")
