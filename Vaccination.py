import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# =================== PARAMETERS ===================
total_population = 10000
initial_infected = 5
recovery_days = 14
R0 = 4.5
intervention_day = 30
vaccine_start_day = 60  # Vaccination campaign start

# Vaccination parameters (based on COVID-19 mRNA vaccines)
vaccine_params = {
    'daily_capacity': 200,          # Max vaccines/day
    'efficacy': 0.95,               # Infection prevention
    'efficacy_severe': 0.98,        # Severe disease prevention
    'dose_interval': 21,            # Days between doses
    'waning_start': 180,            # When immunity begins waning
    'waning_rate': 0.001            # Daily protection loss
}

# Airborne parameters (unchanged)
aerosol_decay_rate = 0.63
ventilation_rate = 4.0

# =================== VACCINATION MODEL ===================
def vaccination_model(t, params, S, V1, V2):
    """Calculates daily vaccination flows"""
    # No vaccination before start day
    if t < params['vaccine_start_day']:
        return 0, 0, 0
    
    # Calculate available susceptibles
    available_S = max(S, 0)
    
    # First dose vaccinations
    first_doses = min(params['daily_capacity'], available_S)
    
    # Second doses (only for those who had first dose ≥ interval days ago)
    second_doses = 0
    if t >= params['vaccine_start_day'] + params['dose_interval']:
        # Estimate eligible population (simplified)
        eligible = V1 * (1 - params['waning_rate'])**(t - params['vaccine_start_day'])
        second_doses = min(params['daily_capacity'] * 0.8, eligible)
    
    # Waning immunity (for fully vaccinated)
    waned = 0
    if t > params['waning_start']:
        waned = V2 * params['waning_rate']
    
    return first_doses, second_doses, waned

# =================== MODIFIED SIRV MODEL ===================
def sirv_model(y, t, params):
    S, I, R, V1, V2, VW = y  # V1: partial vax, V2: full vax, VW: waned
    N = S + I + R + V1 + V2 + VW
    
    # Dynamic R0 with interventions
    current_R0 = params['R0'] * np.exp(-params['intervention_strength'] * (t > params['intervention_day']))
    
    # Environmental factor
    env_factor = np.exp(-params['aerosol_decay_rate']/params['ventilation_rate']) * (0.8 + 0.2*np.sin(t/10))
    beta = current_R0 * env_factor / params['recovery_days']
    
    # Vaccination flows
    first_doses, second_doses, waned = vaccination_model(t, params, S, V1, V2)
    
    # Differential equations
    dSdt = -beta * S * I / N - first_doses + waned
    
    # Infection terms with different susceptibility
    infection_flow = beta * I / N
    dIdt = (infection_flow * S + 
            infection_flow * V1 * (1 - params['vaccine_efficacy']*0.5) +  # Partial protection
            infection_flow * VW * (1 - params['vaccine_efficacy']*0.7)) - (1/params['recovery_days']) * I# Waned protection
            
    
    dRdt = (1/params['recovery_days']) * I
    
    # Vaccination compartments
    dV1dt = first_doses - second_doses - infection_flow * V1 * (1 - params['vaccine_efficacy']*0.5)
    dV2dt = second_doses - (waned if t > params['waning_start'] else 0)
    dVWdt = (waned if t > params['waning_start'] else 0) - infection_flow * VW * (1 - params['vaccine_efficacy']*0.7)
    
    return [dSdt, dIdt, dRdt, dV1dt, dV2dt, dVWdt]

# =================== SIMULATION SETUP ===================
params = {
    'R0': R0,
    'recovery_days': recovery_days,
    'intervention_day': intervention_day,
    'intervention_strength': 0.7,
    'aerosol_decay_rate': aerosol_decay_rate,
    'ventilation_rate': ventilation_rate,
    'vaccine_start_day': vaccine_start_day,
    **vaccine_params  # Merge vaccine parameters
}

# Initial conditions [S0, I0, R0, V1_0, V2_0, VW_0]
y0 = [total_population - initial_infected, initial_infected, 0, 0, 0, 0]

# Time vector (days)
t = np.linspace(0, 300, 300)

# Solve ODE
solution = odeint(sirv_model, y0, t, args=(params,))
S, I, R, V1, V2, VW = solution.T

# =================== VISUALIZATION ===================
plt.figure(figsize=(14, 8))

# Epidemic curves
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.plot(t, V1+V2+VW, 'c', label='Vaccinated (total)')
plt.plot(t, V2, 'm', label='Fully Vaccinated')

# Key events
plt.axvline(intervention_day, color='k', linestyle='--', label='Interventions Start')
plt.axvline(vaccine_start_day, color='purple', linestyle='--', label='Vaccination Start')
plt.axvline(vaccine_start_day + vaccine_params['dose_interval'], color='purple', linestyle=':', label='Second Doses Start')

plt.title('Airborne Virus Spread with Vaccination Dynamics')
plt.xlabel('Days')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()

# =================== VACCINATION METRICS ===================
peak_before_vax = np.max(I[t < vaccine_start_day])
peak_after_vax = np.max(I[t >= vaccine_start_day])
vax_coverage = (V2[-1] + VW[-1]) / total_population

print(f"Peak before vaccination: {peak_before_vax:.0f} infections")
print(f"Peak after vaccination: {peak_after_vax:.0f} infections")
print(f"Final vaccination coverage: {vax_coverage*100:.1f}%")
print(f"Cases prevented: {peak_before_vax - peak_after_vax:.0f} ({((peak_before_vax - peak_after_vax)/peak_before_vax*100):.1f}% reduction)")