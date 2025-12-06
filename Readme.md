# README.md

# Student Performance — clustering & Bayesian network (minimal)

## Краткое описание

Проект выполняет предобработку данных об успеваемости студентов, строит несколько кластеров (KMeans, GMM, Agglomerative, DBSCAN), сравнивает их по silhouette-score, визуализирует результаты (PCA / UMAP) и обучает дискретную Байесовскую сеть на основе меток кластеров.

## Структура проекта

```
student_performance_clustering/
│   .gitignore
│   README.md
│   requirements.txt
│   run_pipeline.py
└── src/
    ├── data_prep.py        # загрузка и предобработка
    ├── clustering.py       # кластеризация и сохранение результатов
    ├── bayes_net.py        # обучение и инференс Байесовской сети
    └── visualize.py        # визуализации и анализ кластеров
```

## Установка

1. Клонировать репозиторий:

```bash
git clone <repo-url>
cd student_performance_clustering
```

2. Создать виртуальное окружение и активировать:

```bash
python -m venv .venv
# windows
.venv\Scripts\activate
# linux / mac
source .venv/bin/activate
```

3. Установить зависимости:

```bash
pip install -r requirements.txt
```

## Данные

Положите исходный Excel-файл `Students_Performance_data_set.xlsx` в папку `data/` или укажите путь в `run_pipeline.py`.

## Запуск

В корне проекта:

```bash
python run_pipeline.py
```

Это выполнит загрузку, предобработку, кластеризацию, визуализации и обучение байесовой сети (в зависимости от наличия пакетов и данных).

## Краткое описание модулей

* `src/data_prep.py` — загрузка, очистка, кодирование категорий, масштабирование, дискретизация (KBins).
* `src/clustering.py` — KMeans, GMM, Agglomerative, DBSCAN; сохраняет метки и модели в `clusters/`.
* `src/bayes_net.py` — строит структуру (HillClimbSearch) и обучает DiscreteBayesianNetwork (pgmpy), сохраняет в `bayes_nets/`.
* `src/visualize.py` — сравнение silhouette, PCA/UMAP визуализации, базовый анализ кластеров.

## Результаты

* `clusters/<method>/labels.csv` — метки кластеров.
* `clusters/<method>/model.pkl` — сериализованные модели кластеризации.
* `bayes_nets/<method>/` — структура и модель Байесовской сети.
