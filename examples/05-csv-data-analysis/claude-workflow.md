# CSV Data Analysis — Claude Code Workflow

Multi-step analysis of CO2 emissions data using the RLM plugin.

## Prerequisites

1. Generate the dataset:
   ```bash
   python generate_dataset.py > emissions.csv
   ```
2. RLM plugin installed in Claude Code (see [plugin docs](../../plugin/README.md))

## Step 1: Schema Exploration

Ask Claude Code:

```
/rlm:rlm examples/05-csv-data-analysis/emissions.csv Explore the structure of this CSV dataset. How many rows and columns? What are the column types and value ranges? Show summary statistics.
```

## Step 2: Trend Analysis

```
/rlm:rlm examples/05-csv-data-analysis/emissions.csv Which countries have reduced their CO2 emissions over the time period, and which have increased? Quantify the percentage change for each country from the first to last year.
```

## Step 3: Per-Capita Regional Comparison

```
/rlm:rlm examples/05-csv-data-analysis/emissions.csv Calculate per-capita emissions for each country across all years. Which countries have the highest and lowest per-capita emissions? How do developed vs developing nations compare?
```

## Step 4: GDP-Emissions Correlation

```
/rlm:rlm examples/05-csv-data-analysis/emissions.csv Analyze the relationship between GDP per capita and total emissions. Is there a correlation? Do richer countries emit more per capita? Are there outliers?
```

## What to Expect

- **Step 1** uses the REPL to parse CSV and compute basic statistics with `collections`
- **Step 2** demonstrates time-series analysis using the pre-imported `itertools` and `collections` modules
- **Step 3** combines multiple columns (emissions / population) for derived metrics
- **Step 4** explores cross-variable relationships — the REPL's `math` module enables correlation calculations
