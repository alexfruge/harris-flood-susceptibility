Alex Fruge
CWID: 10888420
The code is split into various folders in src/, with sections dedicated to downloading the raw data, processing/clipping the data to the correct projection, training & evaluating the model, and finally visualization.

In order to run the code, you'll need to create a uv environment using the provided .venv. Using ```uv sync``` should accomplish this. Once the environment is set up, activate it and then run the following commands in order:

```uv run run_download.py ```
```uv run run_pipeline.py ```
```uv run run_model.py ```

Note: run_model will take the longest of these by far.

The files that cannot be automatically downloaded (NLCD and County shape) are included for convenience.