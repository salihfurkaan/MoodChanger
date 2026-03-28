# MoodChanger: Athlete Wellness Analytics

## Overview
A comprehensive analytics pipeline for athlete wellness monitoring using wearable sensor data. Generates synthetic data, processes it through an end-to-end pipeline, and provides interactive dashboards with PDF reporting.

## Features
- **Synthetic Data Generation**: Realistic wearable sensor data simulation
- **Analytics Pipeline**: Feature extraction, derived states, scoring models
- **Interactive Dashboard**: Streamlit-based visualization with multiple tabs
- **PDF Reports**: Automated professional report generation and download

## Quick Start
1. Install dependencies: `pip install streamlit pandas numpy plotly scikit-learn matplotlib`
2. Run pipeline: `python pipeline.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. Generate report: `python generate_report.py` or use dashboard download

## Project Structure
- `data_architecture.py`: Data generation and system documentation
- `synthetic_data.py`: Daily record generation
- `preprocessing.py`: Data cleaning and aggregation
- `pipeline.py`: Feature extraction and derived states
- `streamlit_app.py`: Interactive dashboard
- `generate_report.py`: PDF report generator

## Data Flow
Raw Wearable Data → Preprocessing → Feature Extraction → Derived States → Dashboard/Reports

## Documentation
See `data_architecture.py` for detailed system documentation including data structures, pipeline, models, and limitations.