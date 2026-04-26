# Adaptive Fraud Decisioning Agent (ADK)

## Overview

This project is a proof-of-concept **agentic AI system** designed to simulate a real-world fraud decisioning workflow. Built using **Google Agent Development Kit (ADK)** and Python, the system demonstrates how coordinated AI agents can support the lifecycle of fraud detection, rule evaluation, and rule optimisation.

Unlike static rule-based systems, this prototype introduces an **iterative feedback loop**, where rules are evaluated and refined based on performance metrics such as fraud detection rate and false positive rate (FPR).

The system is fully **interactive via a chat interface**, allowing users to explore fraud analytics and decisioning processes step-by-step.

---

## Key Features

* **Interactive AI Agent (ADK-based)**
  Chat-driven interface for exploring fraud decisioning workflows

* **Synthetic Transaction Data Generation**
  Creates realistic transaction datasets with fraud labels

* **Fraud Pattern Profiling**
  Analyses differences between fraudulent and genuine transactions

* **Rule-Based Detection Engine**
  Applies baseline fraud rules to identify suspicious transactions

* **Performance Evaluation**
  Calculates:

  * Fraud Detection Rate
  * False Positive Rate (FPR)
  * Precision
  * Confusion Matrix (TP, FP, TN, FN)

* **Rule Optimisation Workflow**
  Iteratively refines rules to reduce false positives while maintaining detection performance

* **Automated Reporting**
  Generates an executive-style summary of results, trade-offs, and recommendations

---

## System Architecture

The system is structured as a coordinated set of agents:

* **Data Agent**
  Generates synthetic transaction data

* **Analytics Agent**
  Profiles fraud patterns and key risk indicators

* **Evaluation Agent**
  Applies and evaluates baseline fraud rules

* **Optimisation Agent**
  Refines rule logic based on performance metrics

* **Reporting Agent**
  Produces structured decisioning insights and recommendations

---

## Workflow

The system follows a structured decisioning lifecycle:

```
Generate Data → Analyse Patterns → Apply Rules → Evaluate Performance → Optimise Rules → Generate Report
```

This mirrors how real-world fraud systems evolve through continuous monitoring and tuning.

---

## Example Use Cases

* Exploring fraud detection trade-offs between **detection rate and customer impact**
* Understanding how **rule-based systems can be improved iteratively**
* Demonstrating **agentic AI workflows in decisioning systems**
* Prototyping ideas for **AI-assisted fraud analytics platforms**

---

## Limitations

* Uses **synthetic data only** (no real customer data)
* Simplified rule logic (not production-grade)
* No real-time streaming or deployment
* No governance or monitoring layer (required for production systems)

---

## Future Improvements

* Integration with real or larger datasets
* Rule versioning and audit trail
* LLM-based explanation for rule decisions
* Real-time data pipelines
* Dashboard or API deployment
* Model-based or hybrid (ML + rules) decisioning

---

## Tech Stack

* Python
* Google Agent Development Kit (ADK)
* Pandas / NumPy
* Gemini (Google Generative AI models)

---

## How to Run

1. Install dependencies:

```
pip install google-adk pandas numpy python-dotenv
```

2. Add your API key to `.env`:

```
GOOGLE_API_KEY=your_api_key_here
```

3. Run the ADK web interface:

```
adk web
```

4. Open the browser at:

```
http://127.0.0.1:8000
```

---

## Example Prompts

* Generate 500 synthetic transactions
* Profile fraud patterns
* Evaluate baseline rules
* Optimise rules for FPR below 15%
* Generate final decisioning report

---

## Summary

This project demonstrates how **agentic AI workflows** can be applied to fraud analytics and decisioning systems. By combining rule-based logic with iterative optimisation and interactive exploration, it highlights a pathway toward more **adaptive, explainable, and structured AI-driven decisioning systems**.

---

## Author

[Your Name]
