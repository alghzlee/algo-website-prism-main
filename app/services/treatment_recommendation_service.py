def format_treatment_action(action):
    """
    Format treatment action dosages into human-readable strings.
    
    Args:
        action: Array with [iv_dose, vasopressor_dose]
    
    Returns:
        Dictionary with formatted IV and vasopressor doses
    """
    iv_dose = float(action[0])
    vasopressor_dose = float(action[1])
    return {
        "iv_dose": f"{round(iv_dose, 3)} ml dose of iv fluid",
        "vasopressor_dose": f"{round(vasopressor_dose, 2)} ug/kg/min dose of vasopressor"
    }


# Backward-compatible aliases
physicianAction = format_treatment_action
aiRecommendation = format_treatment_action
