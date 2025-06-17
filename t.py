# Add booster campaign after 240 days
if t > 240: 
    booster_eligible = VW  # Simplified
    booster_doses = min(150, booster_eligible)
# New variant at day 120 with higher R0
if t > 120:
    params['R0'] = 7.0

if t > 120:  # New variant emerges
    params['R0'] = 7.0
    params['vaccine_efficacy'] *= 0.7  # Reduced efficacy

if t > 240:  # Booster campaign
    booster_priority = {'Elderly': 1, 'Adults': 2}
    # Allocate boosters to VW compartment

if 100 < t < 180:  # School closures
    contact_matrix[0, :] *= 0.3  # Reduce young contacts