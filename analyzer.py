from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CITIES = {
    "New Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
}


def fetch_weather_data(cities: dict[str, tuple[float, float]]) -> pd.DataFrame:
  """Objective 3: Consume REST API via HTTP GET and parse JSON payloads."""
  records = []
  base_url = "https://api.open-meteo.com/v1/forecast"

  for city, (lat, lon) in cities.items():
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "forecast_days": 3,
        "timezone": "Asia/Kolkata",
    }
    try:
      response = requests.get(base_url, params=params, timeout=10)
      response.raise_for_status()
      payload = response.json()
      hourly = payload.get("hourly", {})

      city_df = pd.DataFrame({
          "city": city,
          "timestamp": hourly.get("time", []),
          "temperature_c": hourly.get("temperature_2m", []),
          "humidity_pct": hourly.get("relative_humidity_2m", []),
          "wind_speed_kmh": hourly.get("wind_speed_10m", []),
      })
      records.append(city_df)
    except requests.RequestException as exc:
      print(f"Warning: Failed to fetch data for {city}: {exc}")

  if not records:
    raise RuntimeError("No data retrieved from API. Check internet connection.")

  return pd.concat(records, ignore_index=True)


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
  """Objective 1 & 2: Clean missing values, convert types, and engineer NumPy features."""
  cleaned = df.copy()

  # Parse timestamps and drop duplicates
  cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
  cleaned.dropna(subset=["timestamp"], inplace=True)
  cleaned.drop_duplicates(subset=["city", "timestamp"], inplace=True)

  # Fill any missing numeric values with city-wise median
  numeric_cols = ["temperature_c", "humidity_pct", "wind_speed_kmh"]
  for col in numeric_cols:
    cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned[col] = cleaned.groupby("city")[col].transform(
        lambda s: s.fillna(s.median())
    )

  # NumPy vectorized calculations: Heat Index approximation & Comfort Category
  temp = cleaned["temperature_c"].to_numpy()
  hum = cleaned["humidity_pct"].to_numpy()
  cleaned["feels_like_c"] = np.round(temp + 0.05 * hum - 2.0, 2)

  cleaned["comfort_level"] = np.where(
      cleaned["temperature_c"] >= 32,
      "Hot",
      np.where(cleaned["temperature_c"] >= 22, "Pleasant", "Cool"),
  )
  cleaned["date"] = cleaned["timestamp"].dt.date
  return cleaned


def analyze_data(df: pd.DataFrame) -> pd.DataFrame:
  """Objective 1: Compute statistical aggregations using Pandas and NumPy."""
  summary = (
      df.groupby("city")
      .agg(
          records=("temperature_c", "count"),
          mean_temp_c=("temperature_c", lambda x: round(np.mean(x), 2)),
          std_temp_c=("temperature_c", lambda x: round(np.std(x), 2)),
          max_temp_c=("temperature_c", "max"),
          min_temp_c=("temperature_c", "min"),
          avg_humidity_pct=("humidity_pct", lambda x: round(np.mean(x), 1)),
          max_wind_kmh=("wind_speed_kmh", "max"),
      )
      .sort_values(by="mean_temp_c", ascending=False)
      .reset_index()
  )
  return summary


def generate_visualizations(df: pd.DataFrame) -> list[Path]:
  """Objective 2: Create Matplotlib & Seaborn charts and save to disk."""
  sns.set_theme(style="whitegrid")
  saved_charts = []

  # Chart 1: Temperature Trend Line Plot
  plt.figure(figsize=(10, 5))
  sns.lineplot(
      data=df, x="timestamp", y="temperature_c", hue="city", linewidth=2
  )
  plt.title("3-Day Hourly Temperature Forecast Across Indian Cities")
  plt.xlabel("Date & Time (IST)")
  plt.ylabel("Temperature (°C)")
  plt.xticks(rotation=30)
  plt.tight_layout()
  chart1 = OUTPUT_DIR / "temperature_trends.png"
  plt.savefig(chart1, dpi=150)
  plt.close()
  saved_charts.append(chart1)

  # Chart 2: Temperature Distribution Boxplot + Humidity Scatter
  fig, axes = plt.subplots(1, 2, figsize=(12, 5))
  sns.boxplot(data=df, x="city", y="temperature_c", ax=axes[0], palette="Set2")
  axes[0].set_title("Temperature Distribution by City")
  axes[0].set_xlabel("City")
  axes[0].set_ylabel("Temperature (°C)")

  sns.scatterplot(
      data=df,
      x="temperature_c",
      y="humidity_pct",
      hue="city",
      alpha=0.7,
      ax=axes[1],
  )
  axes[1].set_title("Temperature vs. Relative Humidity")
  axes[1].set_xlabel("Temperature (°C)")
  axes[1].set_ylabel("Humidity (%)")

  plt.tight_layout()
  chart2 = OUTPUT_DIR / "city_distributions.png"
  plt.savefig(chart2, dpi=150)
  plt.close()
  saved_charts.append(chart2)

  return saved_charts


def generate_report(
    df: pd.DataFrame, summary: pd.DataFrame, charts: list[Path]
) -> Path:
  """Objective 4: Export cleaned CSV and generate an automated Markdown analysis report."""
  csv_path = OUTPUT_DIR / "cleaned_weather_data.csv"
  df.to_csv(csv_path, index=False)

  warmest_city = summary.iloc[0]["city"]
  warmest_temp = summary.iloc[0]["mean_temp_c"]
  coolest_city = summary.iloc[-1]["city"]
  coolest_temp = summary.iloc[-1]["mean_temp_c"]
  overall_corr = round(
      np.corrcoef(df["temperature_c"], df["humidity_pct"])[0, 1], 3
  )

  report_lines = [
      "# Automated Weather Data Analysis Report",
      f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}",
      f"**Total Records Analyzed:** {len(df)} hourly observations across {df['city'].nunique()} cities",
      "",
      "## 1. Key Insights",
      f"- **Warmest City (Mean):** {warmest_city} ({warmest_temp}°C)",
      f"- **Coolest City (Mean):** {coolest_city} ({coolest_temp}°C)",
      f"- **Temperature-Humidity Pearson Correlation:** `{overall_corr}`",
      "",
      "## 2. City-Wise Statistical Summary",
      summary.to_markdown(index=False),
      "",
      "## 3. Generated Visualizations",
      f"- `{charts[0].name}` — Hourly temperature trajectory across cities",
      f"- `{charts[1].name}` — Temperature spread (boxplot) & humidity correlation",
      f"- `{csv_path.name}` — Cleaned and feature-engineered dataset",
  ]

  report_path = OUTPUT_DIR / "analysis_report.md"
  report_path.write_text("\n".join(report_lines), encoding="utf-8")
  return report_path


def main() -> None:
  print("1. Fetching live weather data from Open-Meteo REST API...")
  raw_df = fetch_weather_data(CITIES)

  print("2. Cleaning data and engineering NumPy features...")
  clean_df = clean_and_transform(raw_df)

  print("3. Running statistical analysis...")
  summary_df = analyze_data(clean_df)
  print("\n--- Summary Statistics ---")
  print(summary_df.to_string(index=False))

  print("\n4. Generating Matplotlib & Seaborn charts...")
  charts = generate_visualizations(clean_df)

  print("5. Generating final report and cleaned CSV...")
  report_file = generate_report(clean_df, summary_df, charts)
  print(f"\nDone! Report saved to: {report_file}")


if __name__ == "__main__":
  main()
