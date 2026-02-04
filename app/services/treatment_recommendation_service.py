def format_treatment_action(action):
    iv_dose = float(action[0])
    vasopressor_dose = float(action[1])
    return {
        "iv_dose": f"{round(iv_dose, 3)} ml dose of iv fluid",
        "vasopressor_dose": f"{round(vasopressor_dose, 3)} ug/kg/min dose of vasopressor"
    }

physicianAction = format_treatment_action
aiRecommendation = format_treatment_action
