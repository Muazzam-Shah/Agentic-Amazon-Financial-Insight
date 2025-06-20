# Agentic-Amazon-Financial-Insight
*An intelligent data analysis engine for Amazon FBA trading data leveraging Google Gemini and LangChain.*



## Project Overview

This repository implements a Python-based AI assistant for analyzing Amazon FBA trading data. It combines:

* **`AmazonTradingAnalyzer`**: A data-analysis engine that loads up to 150 records from Excel or CSV, computes summary statistics, detects sales trends, ranks top products, and compares marketplace performance.
* **`BasicAIAgent`**: A conversational agent built with LangChain and Google Generative AI (Gemini). It exposes analytical tools and maintains dialogue history, enabling interactive, natural-language queries over the dataset.

---

## Features

* Load and summarize Amazon trading data (sales, units, refunds, profit).
* Analyze monthly or weekly sales trends.
* Identify top-performing products by sales, profit, or units.
* Compare performance across marketplaces.
* Conversational query interface powered by LangChain and Gemini AI.
* Automatic saving of analysis outputs to Markdown files.

---

## Prerequisites

* Python 3.9 or higher
* A valid Google Gemini API key
* Required Python packages (see Installation)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Muazzam-Shah/Agentic-Amazon-Financial-Insight.git
   cd Agentic-Amazon-Financial-Insight
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On Windows use `venv\Scripts\activate`
   ```


---

## Configuration

1. **Set the Gemini API key**

   * Option A: Export as an environment variable

     ```bash
     export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
     ```
   * Option B: Create a file named `Gemeni_API_KEY` in the project root containing the key.

2. **Prepare your data file**

   * Supported formats: `.xlsx`, `.csv`
   * The agent will read only the first 150 rows.

---

## Usage

Run the AI agent from the command line:

```bash
python basic_ai_agent.py <data_file.xlsx>
```

Upon startup, the following commands are available in the interactive prompt:

* **`summary`**: Obtain an overview of total sales, units, refunds, and net profit.
* **`trends`**: Examine recent monthly sales trends.
* **`top products`**: Identify top 5 products by sales.
* **`marketplace comparison`**: Compare performance across marketplaces.
* **`history`**: Display conversation history.
* **`clear`**: Clear the conversational memory.
* **`exit`** / **`quit`** / **`q`**: Exit the program.

All analysis outputs are automatically saved under the `analysis_results/` directory as Markdown files.

---

## Tool Descriptions

| Tool Name              | Description                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| `get_data_summary`     | Returns dataset summary: total records, organic sales, units, refunds, and net profit.             |
| `analyze_sales_trends` | Analyzes sales trends over time (monthly or weekly) and returns recent trend figures.              |
| `get_top_products`     | Lists top N products by sales, profit, or units; default metric “sales” and limit 5.               |
| `compare_marketplaces` | Compares total sales, units, and profit across different Amazon marketplaces.                      |
| `save_analysis`        | Saves a given string of analysis content to a timestamped Markdown file under `analysis_results/`. |

---
