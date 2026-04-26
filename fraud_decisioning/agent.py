from google.adk.agents import Agent
import pandas as pd
import numpy as np


transaction_data = None
baseline_results = []
optimised_results = []


def generate_transaction_data(sample_size: int = 500) -> dict:
    """
    Generate synthetic fraud transaction data.
    """
    global transaction_data

    np.random.seed(42)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:05d}" for i in range(1, sample_size + 1)],
        "account_age_days": np.random.randint(1, 3000, sample_size),
        "amount": np.round(np.random.gamma(shape=2.2, scale=900, size=sample_size), 2),
        "txn_count_24h": np.random.poisson(lam=2.2, size=sample_size) + 1,
        "new_payee": np.random.choice([0, 1], sample_size, p=[0.72, 0.28]),
        "new_device": np.random.choice([0, 1], sample_size, p=[0.80, 0.20]),
        "previous_fraud_alert": np.random.choice([0, 1], sample_size, p=[0.88, 0.12]),
    })

    fraud_score = (
        0.30 * (df["amount"] > 3500).astype(int)
        + 0.25 * (df["new_payee"] == 1).astype(int)
        + 0.20 * (df["new_device"] == 1).astype(int)
        + 0.20 * (df["txn_count_24h"] >= 5).astype(int)
        + 0.25 * (df["previous_fraud_alert"] == 1).astype(int)
        + 0.15 * (df["account_age_days"] < 60).astype(int)
    )

    fraud_probability = np.clip(fraud_score, 0.02, 0.95)
    df["is_fraud"] = np.random.binomial(1, fraud_probability)

    transaction_data = df

    return {
        "status": "success",
        "message": f"Generated {sample_size} synthetic transactions.",
        "fraud_rate": round(float(df["is_fraud"].mean()), 3),
        "first_5_rows": df.head(5).to_dict(orient="records"),
    }


def profile_fraud_patterns() -> dict:
    """
    Profile fraud vs genuine transaction behaviour.
    """
    global transaction_data

    if transaction_data is None:
        return {"status": "error", "message": "Please generate data first."}

    df = transaction_data
    fraud = df[df["is_fraud"] == 1]
    genuine = df[df["is_fraud"] == 0]

    return {
        "status": "success",
        "summary": {
            "total_transactions": len(df),
            "fraud_cases": len(fraud),
            "genuine_cases": len(genuine),
            "avg_fraud_amount": round(float(fraud["amount"].mean()), 2),
            "avg_genuine_amount": round(float(genuine["amount"].mean()), 2),
            "fraud_new_payee_rate": round(float(fraud["new_payee"].mean()), 3),
            "genuine_new_payee_rate": round(float(genuine["new_payee"].mean()), 3),
            "fraud_new_device_rate": round(float(fraud["new_device"].mean()), 3),
            "genuine_new_device_rate": round(float(genuine["new_device"].mean()), 3),
            "fraud_previous_alert_rate": round(float(fraud["previous_fraud_alert"].mean()), 3),
            "genuine_previous_alert_rate": round(float(genuine["previous_fraud_alert"].mean()), 3),
        },
        "interpretation": (
            "Fraud is expected to concentrate around higher amounts, new payees, "
            "new devices, high transaction velocity, and previous fraud alerts."
        ),
    }


def _evaluate_rule(df, rule_name, triggered):
    tp = int(((triggered == True) & (df["is_fraud"] == 1)).sum())
    fp = int(((triggered == True) & (df["is_fraud"] == 0)).sum())
    tn = int(((triggered == False) & (df["is_fraud"] == 0)).sum())
    fn = int(((triggered == False) & (df["is_fraud"] == 1)).sum())

    detection_rate = tp / (tp + fn) if (tp + fn) else 0
    fpr = fp / (fp + tn) if (fp + tn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0

    return {
        "rule_name": rule_name,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "fraud_detection_rate": round(detection_rate, 3),
        "false_positive_rate": round(fpr, 3),
        "precision": round(precision, 3),
        "total_triggered": int(triggered.sum()),
    }


def evaluate_baseline_rules() -> dict:
    """
    Evaluate baseline fraud detection rules.
    """
    global transaction_data, baseline_results

    if transaction_data is None:
        return {"status": "error", "message": "Please generate data first."}

    df = transaction_data

    rules = {
        "R1_high_amount_new_payee": (
            (df["amount"] >= 2500)
            & (df["new_payee"] == 1)
        ),
        "R2_velocity_new_payee": (
            (df["txn_count_24h"] >= 5)
            & (df["new_payee"] == 1)
        ),
        "R3_large_amount_previous_alert": (
            (df["amount"] >= 3000)
            & (df["previous_fraud_alert"] == 1)
        ),
    }

    baseline_results = [
        _evaluate_rule(df, rule_name, triggered)
        for rule_name, triggered in rules.items()
    ]

    return {
        "status": "success",
        "message": "Baseline rules evaluated.",
        "baseline_results": baseline_results,
    }


def optimise_rules(target_fpr: float = 0.15) -> dict:
    """
    Optimise fraud rules to reduce false positive rate while preserving detection.
    """
    global transaction_data, optimised_results

    if transaction_data is None:
        return {"status": "error", "message": "Please generate data first."}

    df = transaction_data

    candidate_rules = {
        "R1_optimised_high_amount_new_payee_new_device": (
            (df["amount"] >= 3500)
            & (df["new_payee"] == 1)
            & (df["new_device"] == 1)
        ),
        "R2_optimised_velocity_new_payee_alert": (
            (df["txn_count_24h"] >= 5)
            & (df["new_payee"] == 1)
            & (
                (df["new_device"] == 1)
                | (df["previous_fraud_alert"] == 1)
            )
        ),
        "R3_optimised_large_amount_alert_or_device": (
            (df["amount"] >= 4000)
            & (
                (df["previous_fraud_alert"] == 1)
                | (df["new_device"] == 1)
            )
        ),
    }

    optimised_results = [
        _evaluate_rule(df, rule_name, triggered)
        for rule_name, triggered in candidate_rules.items()
    ]

    recommendations = []

    for result in optimised_results:
        if result["false_positive_rate"] <= target_fpr:
            action = "Keep"
        else:
            action = "Further tighten threshold or add stronger risk condition"

        recommendations.append({
            "rule_name": result["rule_name"],
            "fpr": result["false_positive_rate"],
            "detection_rate": result["fraud_detection_rate"],
            "recommended_action": action,
        })

    return {
        "status": "success",
        "target_fpr": target_fpr,
        "message": "Optimised rules evaluated.",
        "optimised_results": optimised_results,
        "recommendations": recommendations,
    }


def generate_decisioning_report() -> dict:
    """
    Generate final report comparing baseline and optimised rules.
    """
    global baseline_results, optimised_results

    if not baseline_results:
        return {"status": "error", "message": "Please evaluate baseline rules first."}

    if not optimised_results:
        return {"status": "error", "message": "Please optimise rules first."}

    best_optimised = sorted(
        optimised_results,
        key=lambda x: (x["false_positive_rate"], -x["fraud_detection_rate"], -x["precision"])
    )[0]

    report = {
        "executive_summary": (
            "This proof-of-concept demonstrates an adaptive fraud decisioning workflow. "
            "The system generates synthetic transaction data, evaluates baseline fraud rules, "
            "optimises rules to reduce false positives, and compares performance using detection rate, "
            "false positive rate and precision."
        ),
        "baseline_results": baseline_results,
        "optimised_results": optimised_results,
        "best_optimised_rule": best_optimised,
        "key_learning": (
            "Adding stronger risk conditions such as new device, previous fraud alert, and higher amount "
            "thresholds can reduce false positives, but may also reduce fraud detection coverage. "
            "The main decisioning challenge is balancing fraud capture against customer impact."
        ),
        "limitations": [
            "Synthetic data only",
            "Simplified deterministic rules",
            "Not production deployed",
            "Would require real fraud labels, governance, monitoring and validation before production use",
        ],
        "next_steps": [
            "Add rule versioning and audit trail",
            "Add threshold search across multiple rule combinations",
            "Add LLM-generated explanation for each rule change",
            "Add dashboard or API deployment",
        ],
    }

    return {
        "status": "success",
        "report": report,
    }


root_agent = Agent(
    name="adaptive_fraud_decisioning_agent",
    model="gemini-2.5-flash",
    description="Interactive adaptive fraud decisioning agent for rule evaluation and optimisation.",
    instruction="""
You are the Adaptive Fraud Decisioning Coordinator Agent.

Your purpose is to help users explore how fraud decisioning rules can be evaluated and optimised.

You have access to these tools:

1. generate_transaction_data:
Generates synthetic transaction data.

2. profile_fraud_patterns:
Compares fraud and genuine transaction behaviour.

3. evaluate_baseline_rules:
Applies initial rule-based fraud detection logic and calculates detection rate, false positive rate, precision, TP, FP, TN and FN.

4. optimise_rules:
Applies improved rule logic to reduce false positive rate while preserving fraud detection.

5. generate_decisioning_report:
Creates a final report comparing baseline and optimised rules.

Important instructions:
- Always be clear that this is a proof-of-concept using synthetic data.
- Do not claim this is a production fraud model.
- Explain the trade-off between fraud detection and false positive rate.
- When the user asks to improve rules, use optimise_rules.
- When the user asks for final summary or executive report, use generate_decisioning_report.
- Keep responses clear, structured and practical.
""",
    tools=[
        generate_transaction_data,
        profile_fraud_patterns,
        evaluate_baseline_rules,
        optimise_rules,
        generate_decisioning_report,
    ],
)