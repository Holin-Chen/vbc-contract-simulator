"""Human-readable labels for HCAHPS measure IDs."""

HCAHPS_LABELS = {
    # Nurse communication
    "H_COMP_1_A_P":          "Nurses communicated well — Always (%)",
    "H_COMP_1_U_P":          "Nurses communicated well — Usually (%)",
    "H_COMP_1_SN_P":         "Nurses communicated well — Sometimes/Never (%)",
    "H_COMP_1_LINEAR_SCORE": "Nurse communication — linear score",
    "H_COMP_1_STAR_RATING":  "Nurse communication — star rating",
    # Doctor communication
    "H_COMP_2_A_P":          "Doctors communicated well — Always (%)",
    "H_COMP_2_U_P":          "Doctors communicated well — Usually (%)",
    "H_COMP_2_SN_P":         "Doctors communicated well — Sometimes/Never (%)",
    "H_COMP_2_LINEAR_SCORE": "Doctor communication — linear score",
    "H_COMP_2_STAR_RATING":  "Doctor communication — star rating",
    # Medicine communication
    "H_COMP_5_A_P":          "Staff explained medicines — Always (%)",
    "H_COMP_5_U_P":          "Staff explained medicines — Usually (%)",
    "H_COMP_5_SN_P":         "Staff explained medicines — Sometimes/Never (%)",
    "H_COMP_5_LINEAR_SCORE": "Medicine communication — linear score",
    "H_COMP_5_STAR_RATING":  "Medicine communication — star rating",
    # Discharge information
    "H_COMP_6_Y_P":          "Given discharge recovery info — Yes (%)",
    "H_COMP_6_N_P":          "Given discharge recovery info — No (%)",
    "H_COMP_6_LINEAR_SCORE": "Discharge information — linear score",
    "H_COMP_6_STAR_RATING":  "Discharge information — star rating",
    # Cleanliness
    "H_CLEAN_HSP_A_P":       "Room & bathroom always clean (%)",
    "H_CLEAN_HSP_U_P":       "Room & bathroom usually clean (%)",
    "H_CLEAN_HSP_SN_P":      "Room & bathroom sometimes/never clean (%)",
    "H_CLEAN_LINEAR_SCORE":  "Cleanliness — linear score",
    "H_CLEAN_STAR_RATING":   "Cleanliness — star rating",
    # Quietness
    "H_QUIET_HSP_A_P":       "Area around room always quiet at night (%)",
    "H_QUIET_HSP_U_P":       "Area around room usually quiet at night (%)",
    "H_QUIET_HSP_SN_P":      "Area around room sometimes/never quiet (%)",
    "H_QUIET_LINEAR_SCORE":  "Quietness — linear score",
    "H_QUIET_STAR_RATING":   "Quietness — star rating",
    # Overall hospital rating
    "H_HSP_RATING_9_10":          "Overall rating 9–10 out of 10 (%)",
    "H_HSP_RATING_7_8":           "Overall rating 7–8 out of 10 (%)",
    "H_HSP_RATING_0_6":           "Overall rating 0–6 out of 10 (%)",
    "H_HSP_RATING_LINEAR_SCORE":  "Overall hospital rating — linear score",
    "H_HSP_RATING_STAR_RATING":   "Overall hospital rating — star rating",
    # Recommend hospital
    "H_RECMND_DY":           "Would definitely recommend hospital (%)",
    "H_RECMND_PY":           "Would probably recommend hospital (%)",
    "H_RECMND_DN":           "Would not recommend hospital (%)",
    "H_RECMND_LINEAR_SCORE": "Recommend hospital — linear score",
    "H_RECMND_STAR_RATING":  "Recommend hospital — star rating",
    # Nurse detail
    "H_NURSE_LISTEN_A_P":    "Nurses always listened carefully (%)",
    "H_NURSE_LISTEN_U_P":    "Nurses usually listened carefully (%)",
    "H_NURSE_LISTEN_SN_P":   "Nurses sometimes/never listened (%)",
    "H_NURSE_EXPLAIN_A_P":   "Nurses always explained clearly (%)",
    "H_NURSE_EXPLAIN_U_P":   "Nurses usually explained clearly (%)",
    "H_NURSE_EXPLAIN_SN_P":  "Nurses sometimes/never explained clearly (%)",
    "H_NURSE_RESPECT_A_P":   "Nurses always treated with respect (%)",
    "H_NURSE_RESPECT_U_P":   "Nurses usually treated with respect (%)",
    "H_NURSE_RESPECT_SN_P":  "Nurses sometimes/never treated with respect (%)",
    # Doctor detail
    "H_DOCTOR_LISTEN_A_P":   "Doctors always listened carefully (%)",
    "H_DOCTOR_LISTEN_U_P":   "Doctors usually listened carefully (%)",
    "H_DOCTOR_LISTEN_SN_P":  "Doctors sometimes/never listened (%)",
    "H_DOCTOR_EXPLAIN_A_P":  "Doctors always explained clearly (%)",
    "H_DOCTOR_EXPLAIN_U_P":  "Doctors usually explained clearly (%)",
    "H_DOCTOR_EXPLAIN_SN_P": "Doctors sometimes/never explained clearly (%)",
    "H_DOCTOR_RESPECT_A_P":  "Doctors always treated with respect (%)",
    "H_DOCTOR_RESPECT_U_P":  "Doctors usually treated with respect (%)",
    "H_DOCTOR_RESPECT_SN_P": "Doctors sometimes/never treated with respect (%)",
    # Medication side effects
    "H_SIDE_EFFECTS_A_P":    "Staff always discussed medication side effects (%)",
    "H_SIDE_EFFECTS_U_P":    "Staff usually discussed medication side effects (%)",
    "H_SIDE_EFFECTS_SN_P":   "Staff sometimes/never discussed side effects (%)",
    # Medication purpose
    "H_MED_FOR_A_P":         "Staff always explained new medication purpose (%)",
    "H_MED_FOR_U_P":         "Staff usually explained new medication purpose (%)",
    "H_MED_FOR_SN_P":        "Staff sometimes/never explained medication purpose (%)",
    # Discharge help / symptoms
    "H_DISCH_HELP_Y_P":      "Discussed post-discharge help needs — Yes (%)",
    "H_DISCH_HELP_N_P":      "Discussed post-discharge help needs — No (%)",
    "H_SYMPTOMS_Y_P":        "Received written info on symptoms to watch — Yes (%)",
    "H_SYMPTOMS_N_P":        "Received written info on symptoms to watch — No (%)",
    # Summary
    "H_STAR_RATING":         "HCAHPS summary star rating",
}


FEATURE_LABELS = {
    "MSPB_Ratio":               "MSPB Ratio (episode cost vs national median)",
    "Total_Performance_Score":  "VBP Total Performance Score",
    "Hospital_overall_rating":  "Overall Star Rating",
    "Composite_Readmission":    "Composite Readmission Ratio",
    "Domain_Clinical_Score":    "VBP Domain: Clinical Outcomes",
    "Domain_Safety_Score":      "VBP Domain: Safety",
    "Domain_Engagement_Score":  "VBP Domain: Person & Community Engagement",
    "Domain_Efficiency_Score":  "VBP Domain: Efficiency & Cost Reduction",
}


def label(measure_id: str) -> str:
    """Return a human-readable label for a raw HCAHPS measure ID."""
    return HCAHPS_LABELS.get(measure_id, measure_id.replace("_", " ").title())


def feature_label(col: str) -> str:
    """Return a human-readable label for any feature column name."""
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    if col.startswith("HCAHPS_"):
        return label(col[len("HCAHPS_"):])
    if col.startswith("Readmission_"):
        return "Readmission: " + col[len("Readmission_"):].replace("_", " ")
    if col.startswith("State_"):
        return "State: " + col[len("State_"):]
    if col.startswith("Hospital Type_"):
        return "Hospital Type: " + col[len("Hospital Type_"):]
    if col.startswith("Hospital Ownership_"):
        return "Ownership: " + col[len("Hospital Ownership_"):]
    return col.replace("_", " ").title()
