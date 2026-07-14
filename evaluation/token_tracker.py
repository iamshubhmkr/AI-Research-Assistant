"""Per-agent token + cost report (data captured in llm_client)."""
from llm_client import get_token_usage, estimate_cost


class TokenTracker:
    def print_report(self, usage=None):
        usage = usage or get_token_usage()
        if not usage:
            print("No token usage recorded.")
            return
        ti = sum(d["input_tokens"] for d in usage.values())
        to = sum(d["output_tokens"] for d in usage.values())
        print(f"\n{'Agent':<22}{'In':>9}{'Out':>9}{'Calls':>7}{'ms':>9}")
        for agent, d in sorted(usage.items(), key=lambda x: x[1]['input_tokens'], reverse=True):
            print(f"{agent:<22}{d['input_tokens']:>9,}{d['output_tokens']:>9,}"
                  f"{d['calls']:>7}{d['total_ms']:>9,.0f}")
        print(f"{'TOTAL':<22}{ti:>9,}{to:>9,}   est ${estimate_cost(usage):.4f}")
