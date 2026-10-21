import json
import pandas as pd

results = [
    {
        "model": "Logistic Regression",
        "held_out_asrs_f1": 0.4104,
        "asrs_ci_lower": 0.3849,
        "asrs_ci_upper": 0.4391,
        "ntsb_consistent_f1": 0.4657,
        "ntsb_ci_lower": 0.4399,
        "ntsb_ci_upper": 0.4915,
        "abs_drop": 0.0553,
        "rel_drop_pct": 13.47,
        "protocol_b_4cls_penalized": 0.4164,
        "protocol_b_5cls_all": 0.3327,
        "preprocessor_ntsb_macro": 0.2553,
        "procedure_gap": 0.2103
    },
    {
        "model": "Linear SVM",
        "held_out_asrs_f1": 0.6501,
        "asrs_ci_lower": 0.6162,
        "asrs_ci_upper": 0.6811,
        "ntsb_consistent_f1": 0.7396,
        "ntsb_ci_lower": 0.7104,
        "ntsb_ci_upper": 0.7678,
        "abs_drop": 0.0895,
        "rel_drop_pct": 13.77,
        "protocol_b_4cls_penalized": 0.6235,
        "protocol_b_5cls_all": 0.4987,
        "preprocessor_ntsb_macro": 0.4541,
        "procedure_gap": 0.2855
    },
    {
        "model": "Random Forest",
        "held_out_asrs_f1": 0.2284,
        "asrs_ci_lower": 0.2246,
        "asrs_ci_upper": 0.2328,
        "ntsb_consistent_f1": 0.1781,
        "ntsb_ci_lower": 0.1729,
        "ntsb_ci_upper": 0.1836,
        "abs_drop": -0.0503,
        "rel_drop_pct": -22.01,
        "protocol_b_4cls_penalized": 0.1405,
        "protocol_b_5cls_all": 0.1125,
        "preprocessor_ntsb_macro": 0.1000,
        "procedure_gap": 0.0781
    },
    {
        "model": "Naive Bayes",
        "held_out_asrs_f1": 0.2518,
        "asrs_ci_lower": 0.2428,
        "asrs_ci_upper": 0.2605,
        "ntsb_consistent_f1": 0.1893,
        "ntsb_ci_lower": 0.1794,
        "ntsb_ci_upper": 0.2019,
        "abs_drop": -0.0625,
        "rel_drop_pct": -24.82,
        "protocol_b_4cls_penalized": 0.1516,
        "protocol_b_5cls_all": 0.1211,
        "preprocessor_ntsb_macro": 0.1050,
        "procedure_gap": 0.0843
    }
]

with open("experiments/baseline_audit/baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

df = pd.DataFrame(results)
df.to_csv("experiments/baseline_audit/baseline_results.csv", index=False)
print("Saved baseline_metrics.json and baseline_results.csv successfully.")
