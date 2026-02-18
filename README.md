# cfpb-complaint-triage

## Project Overview

# Триаж жалоб CFPB

Петров Виктор Иванович, Б05-222

## Постановка задачи

Цель проекта — построить модель, которая по текстовому описанию жалобы клиента (complaint narrative) автоматически определяет категорию финансового продукта (Product), к которому относится жалоба.

Это типичная задача триажа: при поступлении обращения его можно автоматически маршрутизировать в нужную продуктовую/юридическую команду, ускорять обработку и снижать нагрузку на операторов.
Дополнительно (если останется время): предсказание поля Issue или бинарного признака Timely response как отдельной/мультитасковой задачи.

## Формат входных и выходных данных

В режиме обучения вход — таблица (CSV/JSON) с как минимум такими полями:
• complaint_id (или аналогичный идентификатор);
• consumer_complaint_narrative (текст);
• product (целевой класс).

На инференсе интерфейс будет как у сервиса/пакета:
• вход: JSON {"text": "<complaint narrative>"}
• выход: JSON {"product": "<label>", "proba_topk": [{"label": "...", "p": ...}, ...]}
Дополнительно будет CLI-команда для пакетной разметки CSV (колонка text → pred_product, pred_proba).

## Метрики

Основная метрика: macro F1 по классам Product (чтобы учитывать дисбаланс). Также считаю accuracy и weighted F1.

Ожидаемые значения (для ограничения на top-N самых частых продуктов и наличии ~десятков тысяч текстов):
• бейзлайн (bag-of-words/embeddings + linear): macro F1 порядка 0.60–0.75;
• fine-tune DistilBERT: macro F1 порядка 0.75–0.85.

Финальные цели уточню после первичного EDA и оценки дисбаланса.

## Валидация и тест

Разделение данных:

1. фильтрую записи с непустым narrative (тексты публикуются только при согласии пользователя);
2. беру top-10 (или top-20) самых частых классов Product, чтобы задача была выполнимой и измеримой;
3. делаю stratified split на train/val/test (например 80/10/10) с фиксированным random seed.

Воспроизводимость:
• фиксирую seed (python/numpy/torch), deterministic-режим в torch по возможности;
• сохраняю версию датасета (скачивание по ссылке + checksum) и параметры фильтрации в конфиге.

## Датасеты

Источник данных: Consumer Complaint Database от CFPB (Consumer Financial Protection Bureau).
Скачивание: полный датасет доступен как CSV или JSON, также можно выгружать подмножества. (Ссылку укажу прямо в репозитории, в README и в скрипте загрузки.)

Объём: датасет большой (сотни МБ / ГБ в зависимости от выгрузки). В проекте буду использовать подмножество строк с narrative, чтобы обучение было быстрым.

Особенности/риски:
• сильный дисбаланс классов Product;
• тексты могут содержать редактированные/замаскированные фрагменты (XXXX);
• у части записей narrative отсутствует — нужно фильтровать.
Альтернатива/зеркало (если удобнее): датасет на Kaggle, реплицирующий Consumer Complaint Database.

## Моделирование

### Бейзлайн

Бейзлайн 1 (PyTorch): модель “average embeddings”:
• токенизация (простая, по пробелам) → словарь;
• embedding слой → усреднение по токенам → linear классификатор.
Это быстро, полностью на PyTorch, даст стартовую метрику.

Бейзлайн 2 (опционально): логистическая регрессия/linear SVM на TF-IDF (только для ориентиров, если преподаватели не против).

### Основная модель

Основная модель: fine-tuning трансформера для классификации текста (Hugging Face Transformers), например:
• distilbert-base-uncased (или multilingual, если буду брать много не-EN текстов).

Обучение:
• стандартный Trainer или кастомный PyTorch train loop;
• early stopping по val macro F1;
• логирование метрик и параметров экспериментов (например, MLflow/W&B) и сохранение лучшего чекпойнта.

## Внедрение

Формат внедрения:
• inference-сервис на FastAPI (endpoint /predict и /health);
• Dockerfile для воспроизводимого запуска;
• возможность batch-инференса через CLI.

Это покрывает типичный сценарий использования модели в проде: интеграция с тикет-системой/CRM через HTTP.

## Установка и подготовка окружения

Минимальная последовательность шагов:

```bash
python -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
poetry run pre-commit install
poetry run pre-commit run -a
```

Примечания:

- `poetry install` устанавливает все зависимости проекта (включая обучение и инференс).
- `pre-commit run -a` должен завершаться без ошибок перед запуском обучения.

## Обучение модели

Базовый запуск обучения (включая `ensure_data()` и препроцессинг):

```bash
poetry run python -m cfpb_complaint_triage.commands train
```

Примеры Hydra override-параметров:

```bash
poetry run python -m cfpb_complaint_triage.commands train data.max_rows=10000 train.epochs=3
poetry run python -m cfpb_complaint_triage.commands train logging.enable=false data.top_k_products=20
poetry run python -m cfpb_complaint_triage.commands train train.accelerator=auto train.devices=1
```

Что делает команда обучения:

- пытается восстановить данные через DVC (`dvc pull`);
- при неудаче пробует скачать датасет по `data.source_url`;
- при отсутствии доступа к данным создает синтетический датасет для smoke-run;
- обучает DistilBERT-классификатор через PyTorch Lightning;
- сохраняет лучший чекпойнт в `artifacts/checkpoints/best.ckpt`;
- сохраняет графики в `plots/` и, при включенном MLflow, логирует артефакты.

Примечание по устройству обучения:

- по умолчанию используется `train.accelerator=auto`, поэтому на Mac с MPS обучение может идти на GPU;
- чтобы принудительно запустить на CPU, укажи `train.accelerator=cpu`.

## Подготовка к продакшену

Экспорт ONNX-модели и сборка inference bundle:

```bash
poetry run python -m cfpb_complaint_triage.commands export_onnx
```

После экспорта формируется:

- `artifacts/model.onnx`
- `artifacts/infer_bundle/model.onnx`
- `artifacts/infer_bundle/label_maps.json`
- `artifacts/infer_bundle/tokenizer_name.txt`
- `artifacts/infer_bundle/resolved_config.yaml`

Экспорт TensorRT-движка (опционально, только для GPU-сред):

```bash
bash cfpb_complaint_triage/production/trt_export.sh artifacts/model.onnx artifacts/model.plan
```

## Inference API (FastAPI)

Локальный запуск API:

```bash
poetry run uvicorn cfpb_complaint_triage.production.api:create_app --factory --host 0.0.0.0 --port 8000
```

Проверка health endpoint:

```bash
curl -X GET http://127.0.0.1:8000/health
```

Запрос на предсказание:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged incorrect fees on my credit card."}'
```

Также можно вывести готовые команды:

```bash
poetry run python -m cfpb_complaint_triage.commands api_instructions
```

## Инференс

Одиночный текст:

```bash
poetry run python -m cfpb_complaint_triage.commands infer_text "I was charged incorrect fees on my credit card."
```

Пакетный инференс по CSV/Parquet:

```bash
poetry run python -m cfpb_complaint_triage.commands infer_batch data/processed/test.parquet artifacts/predictions.parquet consumer_complaint_narrative
```

В выходной таблице добавляются колонки:

- `pred_product`
- `pred_proba_topk_json`

## MLflow Serving

Для печати готовых команд сохранения pyfunc-модели и запуска сервинга:

```bash
poetry run python -m cfpb_complaint_triage.commands serving_instructions
```

Команда выводит:

- сохранение pyfunc-бандла;
- запуск `mlflow models serve`;
- пример запроса в `/invocations` с JSON payload.

## DVC и хранение артефактов

- Данные и крупные артефакты не должны попадать в git, они предназначены для DVC.
- Логика `ensure_data()`:
  1. попробовать `dvc pull`;
  2. при неудаче скачать данные из `configs/data.yaml` (`data.source_url`);
  3. если это тоже недоступно, создать небольшой синтетический датасет для проверки пайплайна.

## Docker

Сборка образа:

```bash
docker build -t cfpb-complaint-triage:latest .
```

Запуск контейнера:

```bash
docker run --rm -p 8000:8000 cfpb-complaint-triage:latest
```

После старта контейнера API доступно по адресу:

- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/predict`

## Диагностика проблем

- Если MLflow недоступен по `http://127.0.0.1:8080`:
  - запустите MLflow-сервер;
  - или используйте `logging.allow_local_fallback=true`.
- Если недоступна загрузка данных:
  - пайплайн автоматически переключится на synthetic fallback.
- Если не проходит экспорт ONNX:
  - проверьте наличие чекпойнта `artifacts/checkpoints/best.ckpt`.
