import json
import sys

METRICS_PATH = "metrics_baseline.json"

# You can tune these later
MIN_F1_MACRO = 0.75
MIN_ACCURACY = 0.75


def main():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
    except FileNotFoundError:
        print(f"Metrics file not found: {METRICS_PATH}", file=sys.stderr)
        sys.exit(1)

    f1 = m.get("f1_macro", 0.0)
    acc = m.get("accuracy", 0.0)

    ok = True
    if f1 < MIN_F1_MACRO:
        print(f"FAIL: f1_macro {f1:.4f} < {MIN_F1_MACRO}", file=sys.stderr)
        ok = False
    else:
        print(f"OK: f1_macro {f1:.4f} >= {MIN_F1_MACRO}")

    if acc < MIN_ACCURACY:
        print(f"FAIL: accuracy {acc:.4f} < {MIN_ACCURACY}", file=sys.stderr)
        ok = False
    else:
        print(f"OK: accuracy {acc:.4f} >= {MIN_ACCURACY}")

    if not ok:
        sys.exit(1)
    print("Metric gate passed.")


if __name__ == "__main__":
    main()
