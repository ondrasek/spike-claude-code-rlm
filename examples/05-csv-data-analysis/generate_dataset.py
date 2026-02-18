#!/usr/bin/env python3
"""Generate a synthetic CO2 emissions dataset for RLM analysis.

Produces ~500 rows of deterministic CSV data to stdout.
Uses only stdlib — no external dependencies.

Columns: country, year, emissions_mt, population, gdp_per_capita, energy_source_mix
"""

import csv
import random
import sys

SEED = 12345
random.seed(SEED)

COUNTRIES = {
    "United States": {
        "base_emissions": 5000,
        "base_pop": 310_000_000,
        "base_gdp": 48000,
        "trend": -0.008,
        "energy_mix": {"coal": 20, "gas": 40, "oil": 25, "nuclear": 8, "renewables": 7},
    },
    "China": {
        "base_emissions": 8000,
        "base_pop": 1_340_000_000,
        "base_gdp": 5000,
        "trend": 0.025,
        "energy_mix": {"coal": 60, "gas": 8, "oil": 18, "nuclear": 5, "renewables": 9},
    },
    "India": {
        "base_emissions": 2000,
        "base_pop": 1_210_000_000,
        "base_gdp": 1500,
        "trend": 0.035,
        "energy_mix": {"coal": 55, "gas": 6, "oil": 25, "nuclear": 3, "renewables": 11},
    },
    "Germany": {
        "base_emissions": 800,
        "base_pop": 81_000_000,
        "base_gdp": 42000,
        "trend": -0.02,
        "energy_mix": {"coal": 25, "gas": 25, "oil": 20, "nuclear": 10, "renewables": 20},
    },
    "Brazil": {
        "base_emissions": 450,
        "base_pop": 195_000_000,
        "base_gdp": 11000,
        "trend": 0.01,
        "energy_mix": {"coal": 5, "gas": 12, "oil": 35, "nuclear": 1, "renewables": 47},
    },
    "Japan": {
        "base_emissions": 1200,
        "base_pop": 128_000_000,
        "base_gdp": 44000,
        "trend": -0.012,
        "energy_mix": {"coal": 27, "gas": 30, "oil": 25, "nuclear": 5, "renewables": 13},
    },
    "Nigeria": {
        "base_emissions": 90,
        "base_pop": 160_000_000,
        "base_gdp": 2500,
        "trend": 0.04,
        "energy_mix": {"coal": 1, "gas": 15, "oil": 55, "nuclear": 0, "renewables": 29},
    },
    "United Kingdom": {
        "base_emissions": 500,
        "base_pop": 63_000_000,
        "base_gdp": 39000,
        "trend": -0.03,
        "energy_mix": {"coal": 10, "gas": 35, "oil": 25, "nuclear": 15, "renewables": 15},
    },
    "Australia": {
        "base_emissions": 400,
        "base_pop": 22_000_000,
        "base_gdp": 52000,
        "trend": -0.005,
        "energy_mix": {"coal": 35, "gas": 25, "oil": 25, "nuclear": 0, "renewables": 15},
    },
    "South Korea": {
        "base_emissions": 600,
        "base_pop": 50_000_000,
        "base_gdp": 23000,
        "trend": 0.015,
        "energy_mix": {"coal": 30, "gas": 25, "oil": 25, "nuclear": 12, "renewables": 8},
    },
}

BASE_YEAR = 2010
YEARS = list(range(BASE_YEAR, BASE_YEAR + 15))  # 2010-2024


def main() -> None:
    """Generate the dataset and write CSV to stdout."""
    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "country",
            "year",
            "emissions_mt",
            "population",
            "gdp_per_capita",
            "energy_source_mix",
        ]
    )

    for country, params in COUNTRIES.items():
        for i, year in enumerate(YEARS):
            # Apply trend with noise
            trend_factor = (1 + params["trend"]) ** i
            noise = random.uniform(0.95, 1.05)
            emissions = round(params["base_emissions"] * trend_factor * noise, 1)

            # Population grows ~0.5-1.5% per year depending on country
            pop_growth = random.uniform(1.005, 1.015) ** i
            population = int(params["base_pop"] * pop_growth)

            # GDP grows ~1-5% per year with noise
            gdp_growth = random.uniform(1.01, 1.05) ** i
            gdp_noise = random.uniform(0.97, 1.03)
            gdp_per_capita = round(params["base_gdp"] * gdp_growth * gdp_noise, 2)

            # Energy mix evolves: renewables grow, coal shrinks
            mix = dict(params["energy_mix"])
            renewable_gain = min(i * random.uniform(0.5, 1.5), 20)
            coal_loss = min(i * random.uniform(0.3, 1.0), mix["coal"] * 0.6)
            mix["renewables"] = round(mix["renewables"] + renewable_gain, 1)
            mix["coal"] = round(max(mix["coal"] - coal_loss, 0), 1)
            # Normalize to 100%
            total = sum(mix.values())
            mix = {k: round(v / total * 100, 1) for k, v in mix.items()}

            energy_str = "; ".join(f"{k}:{v}%" for k, v in mix.items())

            writer.writerow(
                [
                    country,
                    year,
                    emissions,
                    population,
                    gdp_per_capita,
                    energy_str,
                ]
            )


if __name__ == "__main__":
    main()
