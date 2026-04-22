# Real-log benchmarks

Not committed to the repository.

## Adding a log

1. Drop your `<name>.log` file into this directory.
2. Add a sibling `<name>.json` annotation:

    ```json
    {
      "expected_line_boundaries": [1234, 5678, 9012],
      "tolerance_lines": 100,
      "notes": "prod incident 2026-04-14, deploy → cascade → recovery"
    }
    ```

3. Run:

    ```bash
    python mentis_log_cli.py benchmark \
      --input-dir benchmarks/real \
      --output real_bench.json
    ```

If you cannot determine exact line boundaries, you can still benchmark
for runtime + detected boundary count by omitting the JSON.

## Notes on what makes a good real-log benchmark

- Annotate *phase-level* boundaries (normal → incident → recovery), not
  every individual error event.
- Leave a few hundred lines of context on either side of each
  transition so tolerance-based matching is meaningful.
- If the log has unique request IDs or trace IDs, the token-based
  Q-align path (default) already prunes hapax legomena via
  `min_token_freq=2`.
