import argparse
import sys
import json
from magda_agent.metacognition.tracker import QualityTracker

def main() -> None:
    """
    CLI integration point for the Jules autonomous loop.
    Allows bash scripts (e.g., CI/CD pipelines or local automated tasks) to
    log continuous improvement metrics easily.
    """
    parser = argparse.ArgumentParser(description="Log and retrieve quality metrics for the Jules loop.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for logging
    log_parser = subparsers.add_parser("log", help="Log a new metric value.")
    log_parser.add_argument("metric_name", type=str, help="Name of the metric (e.g., test_pass_rate)")
    log_parser.add_argument("value", type=float, help="Value of the metric")
    log_parser.add_argument("--metadata", type=str, help="Optional JSON string with metadata", default="{}")
    log_parser.add_argument("--db", type=str, help="Path to SQLite DB", default="./metrics_db.sqlite3")

    # Subparser for getting averages
    avg_parser = subparsers.add_parser("average", help="Calculate average for a metric.")
    avg_parser.add_argument("metric_name", type=str, help="Name of the metric")
    avg_parser.add_argument("--limit", type=int, help="Number of recent entries to average", default=10)
    avg_parser.add_argument("--db", type=str, help="Path to SQLite DB", default="./metrics_db.sqlite3")

    args = parser.parse_args()
    tracker = QualityTracker(db_path=args.db)

    if args.command == "log":
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: Metadata must be a valid JSON string.", file=sys.stderr)
            sys.exit(1)

        tracker.log_metric(args.metric_name, args.value, metadata)
        print(f"Successfully logged {args.metric_name}: {args.value}")

    elif args.command == "average":
        avg = tracker.calculate_average(args.metric_name, args.limit)
        if avg is not None:
            print(f"Average for {args.metric_name} (last {args.limit} entries): {avg}")
        else:
            print(f"No entries found for {args.metric_name}")

if __name__ == "__main__":
    main()
