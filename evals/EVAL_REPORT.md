# Kara Agentic RAG — Benchmark Evaluation Report

**Date:** 2026-08-31 18:56:35
**Evaluator Judge Model:** `openai/gpt-oss-120b`

## Aggregate Scorecard

| Metric | Benchmark Score | Target Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Routing Accuracy** | **100.0%** | $\ge 90\%$ | ✅ PASS |
| **Faithfulness (Anti-Hallucination)** | **98.6%** | $\ge 90\%$ | ✅ PASS |
| **Answer Relevance** | **100.0%** | $\ge 85\%$ | ✅ PASS |
| **Citation Compliance** | **100.0%** | $\ge 90\%$ | ✅ PASS |
| **Mean Latency** | **22482 ms** | $< 5000\text{ ms}$ | ⚠️ SLOW |
| **Mean Total Tokens** | **5188 tokens** | $< 3500\text{ tokens}$ | ✅ OPTIMAL |

## Detailed Test Case Breakdown

| ID | Category | Route (Exp / Act) | Faithfulness | Relevance | Latency | Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `tc-001` | internal_architecture | `internal` / `internal` | 1.00 | 1.00 | 23061ms | 6717 |
| `tc-002` | internal_architecture | `internal` / `internal` | 1.00 | 1.00 | 40154ms | 8798 |
| `tc-003` | internal_architecture | `internal` / `internal` | 1.00 | 1.00 | 41318ms | 7283 |
| `tc-004` | web_comparison | `web` / `web` | 1.00 | 1.00 | 27076ms | 5192 |
| `tc-005` | web_live | `web` / `web` | 0.90 | 1.00 | 24239ms | 4739 |
| `tc-006` | chitchat | `chitchat` / `chitchat` | 1.00 | 1.00 | 767ms | 1839 |
| `tc-007` | unrelated | `unrelated` / `unrelated` | 1.00 | 1.00 | 757ms | 1750 |
