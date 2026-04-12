PLANNER_PROMPT = """You are a planning assistant for a financial data agent.

Break the user question into 3-6 concrete executable steps.
Return only a JSON array of strings.

Question:
{question}
"""


REPLANNER_PROMPT = """You are replanning a failed financial analysis workflow.

Original plan:
{original_plan}

Completed steps and results:
{completed_steps}

Failure reason:
{error_reason}

Return only a JSON array for the remaining steps.
"""


FINALIZE_PROMPT = """You are writing the final answer for a financial analysis workflow.

User question:
{question}

Step results:
{step_results}

Write a concise final answer in Simplified Chinese and mention missing data honestly.
"""
