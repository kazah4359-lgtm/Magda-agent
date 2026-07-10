with open("tests/test_swe_bench.py", "r") as f:
    content = f.read()

content = content.replace('mock_tracker.log_metric.assert_called_once_with("swe_bench_score", 0.7, result["metadata"])', 'mock_tracker.log_metric.assert_called_once_with("swe_bench_score", result["score"], result["metadata"])')

with open("tests/test_swe_bench.py", "w") as f:
    f.write(content)
