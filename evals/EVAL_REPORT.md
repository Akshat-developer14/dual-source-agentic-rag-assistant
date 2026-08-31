# Kara Agentic RAG — Benchmark Evaluation Report

**Date:** 2026-08-31 18:28:17
**Evaluator Judge Model:** `openai/gpt-oss-120b`

## Aggregate Scorecard

| Metric | Benchmark Score | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Routing Accuracy** | **100.0%** | $\ge 90\%$ | ✅ PASS |
| **Faithfulness (Anti-Hallucination)** | **92.9%** | $\ge 90\%$ | ✅ PASS |
| **Answer Relevance** | **84.3%** | $\ge 85\%$ | ⚠️ WARN |
| **Citation Compliance** | **100.0%** | $\ge 90\%$ | ✅ PASS |
| **Mean Latency** | **25673 ms** | $< 5000\text{ ms}$ | ⚠️ SLOW |
| **Mean Total Tokens** | **5299 tokens** | $< 3500\text{ tokens}$ | ✅ OPTIMAL |

## Detailed Test Case Breakdown

| ID | Category | Route (Exp / Act) | Faithfulness | Relevance | Latency | Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tc-001` | internal_architecture | `internal` / `internal` | 1.00 | 1.00 | 41979ms | 7109 |
| `tc-002` | internal_architecture | `internal` / `internal` | 1.00 | 1.00 | 42601ms | 8572 |
| `tc-003` | internal_architecture | `internal` / `internal` | 1.00 | 1.00 | 47143ms | 7541 |
| `tc-004` | web_comparison | `web` / `web` | 0.50 | 1.00 | 22326ms | 5478 |
| `tc-005` | web_live | `web` / `web` | 1.00 | 0.90 | 24052ms | 4799 |
| `tc-006` | chitchat | `chitchat` / `chitchat` | 1.00 | 1.00 | 883ms | 1839 |
| `tc-007` | unrelated | `unrelated` / `unrelated` | 1.00 | 0.00 | 728ms | 1756 |
