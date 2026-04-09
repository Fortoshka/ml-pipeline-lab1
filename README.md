# Airflow ML Pipeline

Этот репозиторий содержит простой Apache Airflow DAG для загрузки, очистки и обучения модели машинного обучения.

## Структура

- `train_model.py` — функции `download_data`, `clear_data`, `train_model`
- `dags/train_pipe.py` — Airflow DAG `cars_training_pipeline`
- `airflow.cfg` — проектная конфигурация Airflow

> Удалены все временные и сгенерированные файлы: `venv`, `data/`, `logs/`, `airflow.db`, `simple_auth_manager_passwords.json.generated`.

## Запуск

1. Активируйте виртуальное окружение:

```bash
cd /home/maxim/ml-pipeline
. venv/bin/activate
```

2. Убедитесь, что переменные окружения настроены:

```bash
export AIRFLOW_HOME=/home/maxim/ml-pipeline
export AIRFLOW__CORE__DAGS_FOLDER=/home/maxim/ml-pipeline/dags
```

3. Инициализируйте базу данных Airflow (если ещё не была инициализирована):

```bash
airflow db reset --yes
```

4. Запустите web-сервер Airflow:

```bash
airflow webserver --port 8080
```

5. В другом терминале активируйте venv и запустите scheduler:

```bash
cd /home/maxim/ml-pipeline
. venv/bin/activate
export AIRFLOW_HOME=/home/maxim/ml-pipeline
export AIRFLOW__CORE__DAGS_FOLDER=/home/maxim/ml-pipeline/dags
airflow scheduler
```

6. Откройте UI в браузере:

```
http://localhost:8080
```

## Тестовый запуск DAG

Можно запустить DAG вручную через CLI:

```bash
airflow dags test cars_training_pipeline 2025-01-01
```

## Примечания

- Если вы используете другую папку для проекта, замените пути в `AIRFLOW_HOME` и `AIRFLOW__CORE__DAGS_FOLDER`.
- DAG расположен в `dags/train_pipe.py` и использует функции из `train_model.py`.
- Результаты обучения сохраняются в папке `data/`.
