# Python Data & APIs: Weather Analysis & Reporting Pipeline

An end-to-end Python data analysis application that consumes live meteorological data from a REST API, cleans and transforms the dataset using **Pandas** and **NumPy**, generates visual insights with **Matplotlib** and **Seaborn**, and exports an automated Markdown report.

## Key Objectives Covered
1. **NumPy & Pandas DataFrames:** Vectorized feature engineering (`feels_like_c`, comfort classification), Pearson correlation, and multi-metric city aggregations.
2. **Data Cleaning & Visualization:** Timestamp parsing, duplicate removal, median imputation, and multi-panel plots (`lineplot`, `boxplot`, `scatterplot`).
3. **REST API Consumption:** Live HTTP `GET` requests to the Open-Meteo API with timeout/error handling and JSON payload normalization.
4. **Automated Report Generation:** Exports `cleaned_weather_data.csv`, high-DPI `.png` charts, and `analysis_report.md`.

## Setup & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/ritick161-sys/python-data-api-analyzer.git](https://github.com/ritick161-sys/python-data-api-analyzer.git)
   cd python-data-api-analyzer
