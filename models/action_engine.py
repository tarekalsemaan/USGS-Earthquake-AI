"""
Model 6 — Action
Moteur de décision pour les recommandations opérationnelles.
"""


def action_engine(damage_level, tsunami):

    if damage_level == "Sévère":
        action = "Intervention prioritaire"
        priority = "Haute"
        reason = "Niveau de dommages sévère."

    elif damage_level == "Modéré":
        action = "Évaluation prioritaire"
        priority = "Moyenne"
        reason = "Niveau de dommages modéré."

    elif damage_level == "Faible":
        action = "Surveillance"
        priority = "Faible"
        reason = "Niveau de dommages faible."

    else:
        action = "Action indéterminée"
        priority = "Indéterminée"
        reason = "Niveau de dommages inconnu."

    if tsunami == 1:
        warning = "Avertissement tsunami"
    else:
        warning = "Aucun avertissement tsunami"

    return action, priority, warning, reason