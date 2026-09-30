# OrbitGuard AI Telemetry Data

Place your real CubeSat telemetry dataset in this directory.

The expected file name (by default) is `telemetry.csv`.

**IMPORTANT**: 
- Ensure the CSV file has a timestamp column and numerical feature columns.
- Update `config/config.yaml` to match the exact column names present in your dataset.
- Do NOT use simulated or fake data for training the models. Use only real, normal telemetry data.
