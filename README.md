# Gusteau, The CookBro 👨‍🍳

<div align="center">
    <img src="assets/img/Gusteau.png" alt="Gusteau Logo" width="500"/>
</div>

**Gusteau** is an NLP project dedicated to generating creative and accurate cooking recipes. Leveraging state-of-the-art Large Language Models (LLMs), the project explores various fine-tuning techniques to adapt general-purpose models into specialized culinary assistants.

Currently, the project features a **Qwen 2.5 0.5B** model fine-tuned using **QLoRA**, accessible via a **Streamlit** web interface.

---

## 🚀 Usage

### 1. Prerequisites
- **Python 3.10+**
- **Gemini API Key** (for evaluation):
    - Create a [Google API Key](https://ai.google.dev/gemini-api/docs?hl=fr)
    - Create a `.env` file at the root:
      ```
      GEMINI_API_KEY=your_api_key
      ```

### 2. Run the Pipeline (CLI)
To run the full data processing, training, and benchmark pipeline:

### 3. Run the Application
Start the interactive Streamlit dashboard to generate recipes:

```bash
streamlit run app.py
```
You can choose between the **Base Model** (General purpose) and **Qwen (QLoRA)** (Specialized) via the sidebar.



```bash
python main.py
```

---

## 📊 Dataset & Exploratory Data Analysis (EDA)

We rely on the **Food.com Recipes Dataset**, a rich collection of cooking recipes and user interactions.

| Attribute            | Description |
|---------------------|-------------|
| **Source**           | [Kaggle - Food.com Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) |
| **Paper**            | [https://arxiv.org/pdf/1909.00105](https://arxiv.com/pdf/1909.00105) |
| **Size**             | 180K Recipes |
| **Type**             | CSV (Converted to JSONL for training) |

**Download Options:**
1.  **Automatic:** `main.py` will attempt to download via Kaggle API (requires `~/.kaggle/kaggle.json`).
2.  **Manual:** Download `RAW_recipes.csv` from Kaggle and place it in `data/`.

### Insights
Before training, we analyzed the dataset to understand the distribution of ingredients, cuisines, and recipe complexity.

| Metric | Visualization |
| :--- | :--- |
| **Ingredients per Cuisine** | ![Ingredients by Cuisine](img/eda/n_ingredients_by_cuisine.png) |
| **Recipe Steps** | ![Steps Distribution](img/eda/n_steps_distribution.png) |

*(More visualizations available in `notebooks/EDA/`)*

---

## 🧪 Preprocessing

This section details the preprocessing steps applied to the dataset.

### 3.1. Removing Columns and Rows
The first step in our preprocessing was to remove non-relevant columns for our exploratory data analysis (EDA). For example, we excluded the `contributor_id` column (which identifies the author of the recipe) and the `submitted` column which indicates the submission date, as they did not serve our analytical objectives.

Next, we removed recipes with a total preparation time greater or equal than 5 hours, considering them outliers. This step eliminated approximately 10,000 recipes from the original dataset of around 231,000 entries.

### 3.2. Lemmatizer & Stop-Word Removing
We then lemmatized the text in the `name` column using the `WordNetLemmatizer` of the `nltk` library and removed stopwords. Additionally, we used a custom list of over 400 irrelevant or noisy words, such as names like "ashley" or "aston", to further clean the data.

**Examples of Name Cleaned into Concise Title**

| Before Cleaning               | After Cleaning        |
| :---------------------------- | :-------------------- |
| OH MY GOD ITS SO AMAZINGGGGG  | potatoes with chicken |
| potatoes with chicken yummy yummy | potatoes with chicken |

### 3.3. Cleaning and Standardizing Instructions
We also performed preprocessing on the `steps_strings` column, which contains the step-by-step instructions for each recipe. This involved the following key tasks:

1.  **Correcting Typos in Units**
    We manually corrected common misspellings, such as ‘”minteus”‘ instead of ‘”minutes”‘, to ensure consistency in unit recognition.
2.  **Standardizing Units**
    To maintain uniformity, we converted various units to standard formats:
    *   milliliters (mL) → liters (L)
    *   inches → centimeters (cm)
    *   fahrenheit → celsius (°C)
3.  **Converting Imprecise Quantities**
    Imprecise or range-based quantities were normalized to approximate average values. Examples include:
    *   `1/2` → `0.5`
    *   `”2-3”` or `”2 to 3”` → `2.5`
4.  **Stemming Words**
    To further reduce variability and enhance text matching, we applied stemming to all words in the instructions. This helped standardize different forms of the same root word (e.g., ‘”chopped”‘, ‘”chopping”‘, ‘”chop”‘ → ‘”chop”‘).

### 3.4. Expanding Columns
After the general preprocessing, we decided (following the recommendation from RecipeNLG [1]) to expand the `nutrition` column into separate features for better insight into the food’s composition.

The `nutrition` column was originally a list of values. We split it into the following individual columns: calories, total fat, sugar, sodium, protein, saturated fat, carbohydrates. This allowed for more granular analysis and visualization of nutritional contents.

---

## 🏗️ Project Architecture

This project is structured around three main pillars:
1.  **Data Processing:** Cleaning and formatting the Food.com dataset.
2.  **Fine-Tuning:** Adapting LLMs using efficient techniques like QLoRA.
3.  **Evaluation:** Rigorous testing using an "LLM-as-a-Judge" approach.

### 📂 Directory Structure
- `data/`: Contains raw and preprocessed datasets.
- `models/`: Stores model adapters and configurations.
- `notebooks/`: Jupyter notebooks for EDA and experiments.
- `src/`: Source code for data prep, training, and evaluation.
- `img/`: Visualizations and assets.

---

## 🧠 Fine-Tuning Methods

We aim to explore multiple fine-tuning strategies. Currently implemented:

### QLoRA (Quantized Low-Rank Adaptation)
We used **QLoRA** to fine-tune `Qwen/Qwen2.5-0.5B-Instruct`.
- **Why?** It drastically reduces memory usage by quantizing the base model to 4-bit while keeping the LoRA adapters in higher precision.
- **Outcome:** Allows training on consumer hardware while retaining high performance.
- **Status:** **Implemented & Available**


---

## ⚖️ Evaluation Methodology

To objectively measure the quality of generated recipes, we use an automated **LLM-as-a-Judge** system.

### 🤖 LLM-as-a-Judge (Gemini)
We employ **Google Gemini** to evaluate the generated recipes based on:
1.  **Coherence:** Do the steps follow a logical order?
2.  **Completeness:** Are all ingredients used?
3.  **Safety:** Are the cooking instructions safe and realistic?

**Sample Benchmark Results:**
| Model | Dish Name | Score (1-5) | Reasoning |
| :--- | :--- | :--- | :--- |
| Qwen-0.5B (Base) | Pizza Dough | 1/5 | Critical flaw: omitted yeast. |
| Qwen-0.5B (Base) | Ice Cream | 2/5 | Texture issues, contradictory steps. |

*Note: The benchmark is designed to be rigorous. Higher scores indicate production-ready recipes.*
