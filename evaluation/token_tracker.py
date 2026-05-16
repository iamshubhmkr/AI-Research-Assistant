"""
Token Tracker — Per-agent token usage and cost reporting.

Interview talking point:
  "I track tokens per agent to identify cost hotspots.
   The synthesizer with ToT uses 60% of total tokens.
   Switching to standard CoT for simple queries cut average cost by 40%."
"""
from llm_client import get_token_usage, estimate_cost


class TokenTracker:
    def print_report(self, usage: dict = None):
        usage = usage or get_token_usage()
        if not usage:
            print("No token usage recorded.")
            return

        total_in = sum(d["input_tokens"] for d in usage.values())
        total_out = sum(d["output_tokens"] for d in usage.values())
        total_cost = estimate_cost(usage)

        print(f"\n{'═' * 60}")
        print(f"  TOKEN USAGE REPORT")
        print(f"{'═' * 60}")
        print(f"{'Agent':<20} {'Input':>8} {'Output':>8} {'Calls':>6} {'Latency':>10}")
        print(f"{'─' * 60}")

        for agent, data in sorted(usage.items(), key=lambda x: x[1]["input_tokens"], reverse=True):
            pct = (data["input_tokens"] + data["output_tokens"]) / max(total_in + total_out, 1) * 100
            print(f"{agent:<20} {data['input_tokens']:>8,} {data['output_tokens']:>8,} "
                  f"{data['calls']:>6} {data['total_ms']:>8,.0f}ms  ({pct:.0f}%)")

        print(f"{'─' * 60}")
        print(f"{'TOTAL':<20} {total_in:>8,} {total_out:>8,}")
        print(f"Estimated cost: ${total_cost:.4f}")
        print(f"{'═' * 60}\n")
