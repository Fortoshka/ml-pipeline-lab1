from datetime import datetime, timedelta
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from train_model import download_data, clear_data, train_model

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="cars_training_pipeline",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    tags=["ml", "training"],
) as dag:

    download_task = PythonOperator(
        task_id="download_data",
        python_callable=download_data,
    )

    clear_task = PythonOperator(
        task_id="clear_data",
        python_callable=clear_data,
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    download_task >> clear_task >> train_task
