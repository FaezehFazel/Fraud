import os
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict

import numpy as np
import pandas as pd


# ============================================================
# 1. Rule definition
# ============================================================

@dataclass
class FraudRule:
    rule_id: str
    description: str
    amount_threshold: float
    min_txn_count_24h: int
    require_new_payee: bool = True
    require_new_device: bool = False
    require_previous_alert: bool = False

    def apply(self, df: pd.DataFrame) -> pd.Series:
        condition = df["amount"] >= self.amount_threshold
        condition &= df["txn_count_24h"] >= self.min_txn_count_24h

        if self.require_new_payee:
            condition &= df["new_payee"] == 1

        if self.require_new_device:
            condition &= df["new_device"] == 1

        if self.require_previous_alert:
            condition &= df["previous_fraud_alert"] == 1

        return condition


# ============================================================
# 2. Synthetic data generation
# ============================================================

def generate_transaction_data(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    df = pd.DataFrame({
        "transaction_id": [f"TXN{i:05d}" for i in range(1, n + 1)],
        "account_age_days": np.random.randint(1, 3000, n),
        "amount": np.round(np.random.gamma(shape=2.2, scale=900, size=n), 2),
        "txn_count_24h": np.random.poisson(lam=2.2, size=n) + 1,
        "new_payee": np.random.choice([0, 1], size=n, p=[0.72, 0.28]),
        "new_device": np.random.choice([0, 1], size=n, p=[0.80, 0.20]),
        "previous_fraud_alert": np.random.choice([0, 1], size=n, p=[0.88, 0.12]),
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

    return df


# ============================================================
# 3. Rule evaluation
# ============================================================

def evaluate_rule(df: pd.DataFrame, rule: FraudRule) -> Dict:
    triggered = rule.apply(df)

    tp = int(((triggered == 1) & (df["is_fraud"] == 1)).sum())
    fp = int(((triggered == 1) & (df["is_fraud"] == 0)).sum())
    tn = int(((triggered == 0) & (df["is_fraud"] == 0)).sum())
    fn = int(((triggered == 0) & (df["is_fraud"] == 1)).sum())

    detection_rate = tp / (tp + fn) if (tp + fn) else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0

    return {
        "rule_id": rule.rule_id,
        "description": rule.description,
        "amount_threshold": rule.amount_threshold,
        "min_txn_count_24h": rule.min_txn_count_24h,
        "require_new_payee": rule.require_new_payee,
        "require_new_device": rule.require_new_device,
        "require_previous_alert": rule.require_previous_alert,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "fraud_detection_rate": round(detection_rate, 3),
        "false_positive_rate": round(false_positive_rate, 3),
        "precision": round(precision, 3),
        "total_triggered": int(triggered.sum()),
    }


# ============================================================
# 4. Rule optimisation engine
# ============================================================

def optimise_rule(
    df: pd.DataFrame,
    base_rule: FraudRule,
    target_fpr: float = 0.15,
    min_detection_rate: float = 0.25,
) -> List[Dict]:

    iterations = []
    current_rule = base_rule

    for iteration in range(1, 6):
        result = evaluate_rule(df, current_rule)
        result["iteration"] = iteration
        result["rule_version"] = f"{base_rule.rule_id}_v{iteration}"

        explanation = []

        if result["false_positive_rate"] > target_fpr:
            explanation.append(
                f"FPR {result['false_positive_rate']} is above target {target_fpr}; tightening rule."
            )

            next_rule = FraudRule(
                rule_id=base_rule.rule_id,
                description=base_rule.description,
                amount_threshold=current_rule.amount_threshold + 500,
                min_txn_count_24h=current_rule.min_txn_count_24h,
                require_new_payee=current_rule.require_new_payee,
                require_new_device=True,
                require_previous_alert=current_rule.require_previous_alert,
            )

            if current_rule.require_new_device:
                next_rule.require_previous_alert = True

        elif result["fraud_detection_rate"] < min_detection_rate:
            explanation.append(
                f"Detection rate {result['fraud_detection_rate']} is below minimum {min_detection_rate}; loosening rule."
            )

            next_rule = FraudRule(
                rule_id=base_rule.rule_id,
                description=base_rule.description,
                amount_threshold=max(500, current_rule.amount_threshold - 500),
                min_txn_count_24h=max(1, current_rule.min_txn_count_24h - 1),
                require_new_payee=current_rule.require_new_payee,
                require_new_device=current_rule.require_new_device,
                require_previous_alert=current_rule.require_previous_alert,
            )

        else:
            explanation.append(
                "Rule meets target balance between fraud detection and false positive rate."
            )
            result["optimisation_action"] = "Keep rule"
            result["explanation"] = " ".join(explanation)
            iterations.append(result)
            break

        result["optimisation_action"] = "Modify rule"
        result["explanation"] = " ".join(explanation)
        iterations.append(result)

        current_rule = next_rule

    return iterations


# ============================================================
# 5. Reporting
# ============================================================

def create_report(results: pd.DataFrame) -> str:
    best_rows = (
        results.sort_values(
            by=["false_positive_rate", "fraud_detection_rate", "precision"],
            ascending=[True, False, False],
        )
        .groupby("rule_id")
        .head(1)
    )

    report = []
    report.append("# Adaptive Fraud Decisioning System Report\n")
    report.append("## Executive Summary\n")
    report.append(
        "This proof-of-concept simulates a fraud decisioning lifecycle where rules are evaluated, "
        "optimised, and compared across iterations using fraud detection rate, false positive rate, and precision.\n"
    )

    report.append("## Best Rule Versions\n")
    report.append(best_rows[
        [
            "rule_id",
            "rule_version",
            "fraud_detection_rate",
            "false_positive_rate",
            "precision",
            "total_triggered",
        ]
    ].to_markdown(index=False))

    report.append("\n## Key Observations\n")

    for _, row in best_rows.iterrows():
        report.append(
            f"- **{row['rule_id']}** selected version **{row['rule_version']}** "
            f"with FPR={row['false_positive_rate']}, detection={row['fraud_detection_rate']}, "
            f"precision={row['precision']}."
        )

    report.append("\n## Limitations\n")
    report.append(
        "- Uses synthetic data only.\n"
        "- Rule optimisation is simplified and deterministic.\n"
        "- Not a production fraud model.\n"
        "- Would require real labels, governance, monitoring, and validation before production use.\n"
    )

    report.append("\n## Next Steps\n")
    report.append(
        "- Add larger datasets and realistic fraud labels.\n"
        "- Add rule versioning and audit trail.\n"
        "- Add LLM-based explanation layer.\n"
        "- Add dashboard or chat interface for investigation teams.\n"
    )

    return "\n".join(report)


# ============================================================
# 6. Main workflow
# ============================================================

def main():
    os.makedirs("output", exist_ok=True)

    print("\nGenerating synthetic transaction data...")
    df = generate_transaction_data(n=1000)

    print(f"Generated {len(df)} transactions.")
    print(f"Fraud rate: {round(df['is_fraud'].mean(), 3)}")

    base_rules = [
        FraudRule(
            rule_id="R1",
            description="High amount + new payee",
            amount_threshold=2500,
            min_txn_count_24h=1,
            require_new_payee=True,
        ),
        FraudRule(
            rule_id="R2",
            description="High velocity + new payee",
            amount_threshold=1000,
            min_txn_count_24h=5,
            require_new_payee=True,
        ),
        FraudRule(
            rule_id="R3",
            description="Large amount + previous alert",
            amount_threshold=3000,
            min_txn_count_24h=1,
            require_new_payee=False,
            require_previous_alert=True,
        ),
    ]

    all_results = []

    print("\nRunning adaptive rule optimisation...\n")

    for rule in base_rules:
        print(f"Optimising {rule.rule_id}: {rule.description}")
        iterations = optimise_rule(df, rule)

        for item in iterations:
            all_results.append(item)
            print(
                f"  {item['rule_version']} | "
                f"Detection={item['fraud_detection_rate']} | "
                f"FPR={item['false_positive_rate']} | "
                f"Precision={item['precision']} | "
                f"Action={item['optimisation_action']}"
            )

        print()

    results_df = pd.DataFrame(all_results)

    results_path = "output/rule_optimisation_results.csv"
    report_path = "output/fraud_decisioning_report.md"

    results_df.to_csv(results_path, index=False)

    report = create_report(results_df)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n==============================")
    print("FINAL OPTIMISATION RESULTS")
    print("==============================")
    print(results_df[
        [
            "rule_id",
            "rule_version",
            "fraud_detection_rate",
            "false_positive_rate",
            "precision",
            "optimisation_action",
        ]
    ].to_string(index=False))

    print("\n==============================")
    print("REPORT CREATED")
    print("==============================")
    print(f"Results saved to: {results_path}")
    print(f"Report saved to: {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()