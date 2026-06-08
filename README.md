# Recommendation System Prototype for E-commerce

This repository contains a research prototype of a hybrid recommendation system for e-commerce. The project was developed as part of a bachelor's qualification work on the topic of development of proposals for building recommender systems in ecommerce.

The main goal of the prototype is to test a hybrid recommendation approach that combines content-based filtering, collaborative filtering and additional mechanisms for increasing catalog coverage and reducing the dominance of bestseller products.

## Project Overview

The system is designed as a local research prototype that emulates the main ideas of a production recommendation architecture. Instead of deploying a full industrial infrastructure with message brokers and distributed databases, the project uses local files and in-memory Python structures to validate the mathematical and algorithmic logic of the recommendation pipeline.

The prototype includes:

* data cleaning and preprocessing pipeline;
* content-based recommendation model using TF-IDF;
* collaborative filtering model using SVD matrix factorization;
* hybrid recommendation scoring;
* popularity-based bonus for less popular products;
* MMR diversification;
* exploration mechanism for increasing catalog coverage;
* evaluation of recommendation quality using standard metrics.

## Architecture

The system follows the idea of separating heavy offline computations from fast online recommendation generation.

The offline batch stage performs resource-intensive operations:

* raw event log loading;
* data cleaning and filtering;
* catalog extraction;
* TF-IDF vectorization;
* user profile construction;
* SVD-based collaborative factorization;
* saving prepared profiles and model factors.

The online stage loads the prepared data into memory and generates recommendations with low latency. It simulates a real-time recommendation scenario where the system reacts to a new user interaction and updates the recommendation output without retraining the full model.

## Main Components

The project is organized into several logical modules:

* `data_pipeline.py` — data loading, cleaning, filtering and train/test preparation;
* `content_based.py` — content-based model using TF-IDF vectors;
* `collaborative.py` — collaborative model based on sparse user-item matrix factorization;
* `stream_recommender.py` — hybrid recommendation engine and real-time interaction simulation;
* `evaluator.py` — evaluation of recommendation models and generation of experimental metrics;
* `main.py` — main orchestration script for running the prototype.

## Dataset

The project uses the Kaggle dataset:

**eCommerce behavior data from multi category store**
Author: M. Kechinov
Dataset file used in the experiment: `2019-Nov.csv`

The dataset is not included in this repository because of its large size. To run the full experiment, download the dataset from Kaggle and place the CSV file into the expected local data directory, for example:

```text
data/raw/2019-Nov.csv
```

## Data Processing

The preprocessing pipeline removes noisy and unreliable behavioral signals before model training. The implemented cleaning steps include:

* filtering accidental clicks with very short interaction duration;
* removing abnormal or bot-like user activity;
* trimming users and products with too few interactions;
* splitting the cleaned data into training and testing parts;
* extracting catalog metadata;
* calculating `popularity_weight` for products based on the number of unique interacting users.

The popularity weight is used later as part of the mechanism for supporting less popular products from the long tail of the catalog.

## Recommendation Models

The prototype compares several recommendation approaches:

1. **Baseline Top-N model**
   A non-personalized recommendation model based on globally popular items.

2. **Content-Based model**
   Uses TF-IDF vectorization of product metadata and builds user profiles based on previously interacted items.

3. **Collaborative model**
   Uses sparse user-item interaction matrix factorization with SVD.

4. **Hybrid model**
   Combines content-based and collaborative scores and applies additional post-processing mechanisms:

   * popularity bonus;
   * MMR diversification;
   * exploration replacement.

## Evaluation Metrics

The system is evaluated using the following metrics:

* `Precision@10`;
* `Recall@10`;
* `NDCG@10`;
* `Coverage`;
* `Latency`.

The final experiment compares the baseline, content-based, collaborative and hybrid models.

## Experimental Result Summary

In the final experiment, the hybrid model achieved the best overall result among the tested approaches. It improved recommendation quality metrics and significantly increased catalog coverage while keeping recommendation latency at a low millisecond level.

The experiment demonstrates that combining traditional recommendation models with diversification and long-tail support mechanisms can improve the balance between relevance and catalog coverage.

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/Oleh1711/recommendation-system-prototype.git
cd recommendation-system-prototype
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place `2019-Nov.csv` into the required data folder.

5. Run the main script:

```bash
python main.py
```

Depending on the current configuration of `main.py`, the script may either run the full preprocessing and model-building pipeline or load already prepared cached files for faster repeated experiments.

## Notes

This repository represents a research prototype, not a production-ready recommendation service. The goal of the project is to validate the algorithmic approach and evaluate the behavior of the hybrid recommendation model in a local experimental environment.

The following production components are intentionally simplified:

* Apache Kafka or other message brokers are not deployed;
* distributed databases are not used;
* Feature Store behavior is emulated using serialized files and in-memory Python structures;
* real-time processing is simulated locally.

These simplifications make the experiment reproducible on a local machine while preserving the core recommendation logic.

## Author

Oleh Tkachenko
Bachelor's qualification work
Specialty: 122 Computer Science
