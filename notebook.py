"""
This is an initial local pipeline file.
"""
# %% [markdown]
# # Import Data & Libraries

# %%
import zipfile
import os
from typing import Text
from absl import logging
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration import metadata, pipeline
import pandas as pd


# ! kaggle datasets download colewelkins/cardiovascular-disease

# %%
with zipfile.ZipFile("cardiovascular-disease.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

print("Files extracted to folder data")

# # %%
df = pd.read_csv('data/cardio_data_processed.csv')
df.head(10)

# # %%
df.info()

# # %%
df = df.drop(columns=['id', 'age'])
df.info()

# # %%
df.to_csv('data/cardio_data_processed.csv', index=False)

# # %%
df.isna().sum()

# %%

# %% [markdown]
# # Pipelines

# %% [markdown]
# ## Set Variable Pipelines

# %%
PIPELINE_NAME = "faizahmp-pipeline"

# pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/cardiovaskular_transform.py"
TUNER_MODULE_FILE = "modules/cardiovaskular_tuner.py"
TRAINER_MODULE_FILE = "modules/cardiovaskular_trainer.py"
# requirement_file = os.path.join(root, "requirements.txt")

# pipeline outputs
OUTPUT_BASE = "outputs"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


# %% [markdown]
# ## Initialize local pipeline

# %%
def init_local_pipeline(
    components, root_path: Text
) -> pipeline.Pipeline:
    """
    Initialize a TFX pipeline with components.

    Args:
        components: list of TFX components
        pipeline_root: directory to save pipeline artifacts
    Returns:
        TFX pipeline
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        # 0 auto-detect based on on the number of CPUs available
        # during execution time.
        "----direct_num_workers=0",
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=root_path,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path),
        beam_pipeline_args=beam_args
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    p_components = init_components({
        'data_dir': DATA_ROOT,
        'transform_module': TRANSFORM_MODULE_FILE,
        'tuner_module': TUNER_MODULE_FILE,
        'training_module': TRAINER_MODULE_FILE,
        'training_steps': 500,
        'eval_steps': 100,
        'serving_model_dir': serving_model_dir
    })

    pipeline = init_local_pipeline(p_components, pipeline_root)
    BeamDagRunner().run(pipeline=pipeline)
