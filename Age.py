import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# ====================== PARAMETERS ======================
total_population = 10000
initial_infected = 10

# Age group definitions (Young: 0-19, Adults: 20-59, Elderly: 60+)
age_groups = ['Young', 'Adults', 'Elderly']
age_distribution = np.array([0.25, 0.55, 0.20])  # Proportion in each age group
pop_per_group = total_population * age_distribution

# Age-specific biological parameters
susceptibility = np.array([0.7, 1.0, 1.2])      # Relative susceptibility to infection
severity = np.array([0.01, 0.05, 0.20])         # Probability of severe disease (hospitalization)
mortality = np.array([0.0001, 0.005, 0.05])     # Infection fatality rate

# Disease parameters
recovery_days = np.array([10, 14, 21])          # Recovery time by age
R0 = 4.5                                        # Basic reproduction number
intervention_day = 30                           # NPIs start
vaccine_start_day = 60                          # Vaccination campaign start

# Vaccination parameters (by age group)
vaccine_priority = {
    'Elderly': 1,   # Highest priority (1 = first)
    'Adults': 2,
    'Young': 3
}

vaccine_params = {
    'daily_capacity': 300,          # Total vaccines/day
    'efficacy': np.array([0.90, 0.95, 0.85]),  # Infection prevention by age
    'efficacy_severe': np.array([0.95, 0.98, 0.92]),  # Severe disease prevention
    'dose_interval': 21,            # Days between doses
    'waning_start': 180,            # When immunity begins waning
    'waning_rate': 0.001            # Daily protection loss
}

# Contact matrix (who interacts with whom)
# Rows: infector age group, Columns: infectee age group
matrix = np.array([
    [1.0, 2.0, 3.0],   # Young: mostly interact with other young
    [4.0, 5.0, 6.0],   # Adults: interact with all groups
    [7.0, 8.0, 9.0]    # Elderly: mostly interact with other elderly
])

# Normalize contact matrix
matrix = matrix / matrix.sum(axis=1, keepdims=True)

# Airborne parameters
aerosol_decay_rate = 0.63
ventilation_rate = 4.0

# ====================== VACCINATION MODEL ======================
def vaccination_model(t, params, S, V1, V2, age_group_idx):
    """Calculates daily vaccination flows with age prioritization"""
    # No vaccination before start day
    if t < params['vaccine_start_day']:
        return 0, 0, 0
    
    # Determine priority for today
    priority_order = sorted(vaccine_priority.items(), key=lambda x: x[1])
    current_priority = priority_order[0][0]
    
    # Check if we've completed the current priority group
    group_idx = age_groups.index(current_priority)
    if S[group_idx] <= 0.01 * pop_per_group[group_idx]:  # Less than 1% susceptible left
        # Move to next priority
        if len(priority_order) > 1:
            current_priority = priority_order[1][0]
            group_idx = age_groups.index(current_priority)
    
    # Calculate available susceptibles in priority group
    available_S = max(S[group_idx], 0)
    
    # First dose vaccinations (allocated to current priority group)
    first_doses = min(params['daily_capacity'], available_S)
    
    # Second doses (for all groups, proportional to V1)
    second_doses = np.zeros(len(age_groups))
    if t >= params['vaccine_start_day'] + params['dose_interval']:
        for i in range(len(age_groups)):
            # Estimate eligible population for second dose
            eligible = V1[i] * (1 - params['waning_rate'])**(t - params['vaccine_start_day'])
            second_doses[i] = min(params['daily_capacity'] * 0.3, eligible)
    
    # Waning immunity (for fully vaccinated)
    waned = np.zeros(len(age_groups))
    if t > params['waning_start']:
        for i in range(len(age_groups)):
            waned[i] = V2[i] * params['waning_rate']
    
    # Only the priority group gets first doses
    first_doses_arr = np.zeros(len(age_groups))
    first_doses_arr[group_idx] = first_doses
    
    return first_doses_arr, second_doses, waned

# ====================== AGE-STRATIFIED SIRV MODEL ======================
def age_stratified_sirv_model(y, t, params):
    # Unpack state variables - each age group has [S, I, R, V1, V2, VW]
    state = y.reshape(len(age_groups), 6)
    S = state[:, 0]
    I = state[:, 1]
    R = state[:, 2]
    V1 = state[:, 3]
    V2 = state[:, 4]
    VW = state[:, 5]
    
    N = S + I + R + V1 + V2 + VW  # Total in each age group
    
    # Dynamic R0 with interventions
    current_R0 = params['R0'] * np.exp(-params['intervention_strength'] * (t > params['intervention_day']))
    
    # Environmental factor
    env_factor = np.exp(-params['aerosol_decay_rate']/params['ventilation_rate']) * (0.8 + 0.2*np.sin(t/10))
    
    # Force of infection matrix
    lambda_matrix = np.zeros((len(age_groups), len(age_groups)))
    for i in range(len(age_groups)):  # Infectee group
        for j in range(len(age_groups)):  # Infector group
            # Transmission = contact * susceptibility * infectiousness
            lambda_matrix[i, j] = (matrix[j, i] * params['susceptibility'][i] * 
                                  current_R0 * env_factor / params['recovery_days'][j])
    
    # Total force of infection on each group
    lambda_total = np.zeros(len(age_groups))
    for i in range(len(age_groups)):
        lambda_total[i] = np.sum(lambda_matrix[i, :] * I)
    
    # Vaccination flows
    first_doses, second_doses, waned = vaccination_model(t, params, S, V1, V2, age_groups)
    
    # Initialize derivatives
    dSdt = np.zeros(len(age_groups))
    dIdt = np.zeros(len(age_groups))
    dRdt = np.zeros(len(age_groups))
    dV1dt = np.zeros(len(age_groups))
    dV2dt = np.zeros(len(age_groups))
    dVWdt = np.zeros(len(age_groups))
    
    # Differential equations for each age group
    for i in range(len(age_groups)):
        # Infection terms with different susceptibility
        infection_force = lambda_total[i] * S[i] / N[i]
        v1_infection = lambda_total[i] * V1[i] / N[i] * (1 - params['vaccine_efficacy'][i] * 0.6)
        vw_infection = lambda_total[i] * VW[i] / N[i] * (1 - params['vaccine_efficacy'][i] * 0.8)
        
        # Susceptible compartment
        dSdt[i] = -infection_force - first_doses[i] + waned[i]
        
        # Infected compartment
        dIdt[i] = (infection_force + v1_infection + vw_infection) - (1/params['recovery_days'][i]) * I[i]
        
        # Recovered compartment
        dRdt[i] = (1/params['recovery_days'][i]) * I[i]
        
        # Vaccination compartments
        dV1dt[i] = first_doses[i] - second_doses[i] - v1_infection
        dV2dt[i] = second_doses[i] - (waned[i] if t > params['waning_start'] else 0)
        dVWdt[i] = (waned[i] if t > params['waning_start'] else 0) - vw_infection
    
    # Track severe cases and deaths (for analysis)
    severe_cases = np.sum(dIdt * params['severity'])
    deaths = np.sum(dIdt * params['mortality'])
    
    # Return derivatives as a flat array
    return np.concatenate((dSdt, dIdt, dRdt, dV1dt, dV2dt, dVWdt))

# ====================== SIMULATION SETUP ======================
params = {
    'R0': R0,
    'recovery_days': recovery_days,
    'intervention_day': intervention_day,
    'intervention_strength': 0.7,
    'aerosol_decay_rate': aerosol_decay_rate,
    'ventilation_rate': ventilation_rate,
    'vaccine_start_day': vaccine_start_day,
    'susceptibility': susceptibility,
    'severity': severity,
    'mortality': mortality,
    'vaccine_efficacy': vaccine_params['efficacy'],
    'waning_start': vaccine_params['waning_start'],
    'waning_rate': vaccine_params['waning_rate'],
    'dose_interval': vaccine_params['dose_interval']
}

# Initial conditions (each age group: [S, I, R, V1, V2, VW])
# Initial infections distributed proportionally
initial_state = []
for i, pop in enumerate(pop_per_group):
    initial_infected_group = max(1, int(initial_infected * age_distribution[i]))
    initial_state.extend([
        pop - initial_infected_group,  # S
        initial_infected_group,        # I
        0,                             # R
        0,                             # V1
        0,                             # V2
        0                              # VW
    ])

# Time vector (days)
t = np.linspace(0, 300, 300)

# Solve ODE
solution = odeint(age_stratified_sirv_model, initial_state, t, args=(params,))

# Reshape solution to [time, age_group, compartment]
solution = solution.reshape(len(t), len(age_groups), 6)

# Extract important metrics
S = solution[:, :, 0]
I = solution[:, :, 1]
R = solution[:, :, 2]
V1 = solution[:, :, 3]
V2 = solution[:, :, 4]
VW = solution[:, :, 5]

# Calculate cumulative severe cases and deaths
cumulative_severe = np.zeros((len(t), len(age_groups)))
cumulative_deaths = np.zeros((len(t), len(age_groups)))
for i in range(1, len(t)):
    dI = I[i] - I[i-1] + R[i] - R[i-1]  # New infections
    cumulative_severe[i] = cumulative_severe[i-1] + dI * severity
    cumulative_deaths[i] = cumulative_deaths[i-1] + dI * mortality

# ====================== VISUALIZATION ======================
plt.figure(figsize=(16, 12))

# Plot 1: Infections by Age Group
plt.subplot(2, 2, 1)
for i, group in enumerate(age_groups):
    plt.plot(t, I[:, i], label=f'{group} Infections')
plt.axvline(intervention_day, color='k', linestyle='--', label='Interventions Start')
plt.axvline(vaccine_start_day, color='purple', linestyle='--', label='Vaccination Start')
plt.title('Infected Individuals by Age Group')
plt.xlabel('Days')
plt.ylabel('Active Infections')
plt.legend()
plt.grid(True)

# Plot 2: Vaccination Coverage by Age Group
plt.subplot(2, 2, 2)
for i, group in enumerate(age_groups):
    total_vaccinated = V1[:, i] + V2[:, i] + VW[:, i]
    plt.plot(t, total_vaccinated / pop_per_group[i] * 100, label=f'{group} Coverage')
plt.axvline(vaccine_start_day, color='purple', linestyle='--', label='Vaccination Start')
plt.title('Vaccination Coverage by Age Group')
plt.xlabel('Days')
plt.ylabel('Percentage Vaccinated')
plt.legend()
plt.grid(True)
plt.ylim(0, 100)

# Plot 3: Cumulative Severe Cases
plt.subplot(2, 2, 3)
for i, group in enumerate(age_groups):
    plt.plot(t, cumulative_severe[:, i], label=f'{group} Severe Cases')
plt.title('Cumulative Severe Cases (Hospitalizations)')
plt.xlabel('Days')
plt.ylabel('Cases')
plt.legend()
plt.grid(True)

# Plot 4: Cumulative Deaths
plt.subplot(2, 2, 4)
for i, group in enumerate(age_groups):
    plt.plot(t, cumulative_deaths[:, i], label=f'{group} Deaths')
plt.title('Cumulative Deaths by Age Group')
plt.xlabel('Days')
plt.ylabel('Deaths')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('age_stratified_model.png', dpi=300)
plt.show()

# ====================== ANALYSIS OUTPUT ======================
def print_stats(metric, values):
    """Print formatted statistics for each age group"""
    print(f"\n{metric}:")
    for i, group in enumerate(age_groups):
        print(f"  - {group}: {values[i]:.0f} ({values[i]/pop_per_group[i]*100:.1f}% of group)")

# Final statistics
final_infected = I[-1] + R[-1]  # Ever infected
final_vaccinated = V1[-1] + V2[-1] + VW[-1]

print("="*60)
print("EPIDEMIC AND VACCINATION SUMMARY")
print("="*60)
print_stats("Total Infections", final_infected)
print_stats("Total Vaccinated", final_vaccinated)
print_stats("Severe Cases", cumulative_severe[-1])
print_stats("Deaths", cumulative_deaths[-1])

# Effectiveness metrics
elderly_vax_coverage = final_vaccinated[2] / pop_per_group[2]
elderly_deaths = cumulative_deaths[-1, 2]
expected_elderly_deaths = final_infected[2] * mortality[2]  # Without vaccine protection
lives_saved = expected_elderly_deaths - elderly_deaths

print("\nVACCINE IMPACT ON HIGH-RISK GROUP:")
print(f"  - Elderly vaccination coverage: {elderly_vax_coverage*100:.1f}%")
print(f"  - Expected deaths without vaccine: {expected_elderly_deaths:.1f}")
print(f"  - Actual deaths: {elderly_deaths:.1f}")
print(f"  - Estimated lives saved: {lives_saved:.1f} ({lives_saved/expected_elderly_deaths*100:.1f}% reduction)")

# Peak infection comparison
peak_infections = np.max(I, axis=0)
print("\nPEAK INFECTIONS BY AGE GROUP:")
for i, group in enumerate(age_groups):
    print(f"  - {group}: {peak_infections[i]:.0f} active infections")
