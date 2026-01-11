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
| **Ingredients per Cuisine** | ![Ingredients by Cuisine](assets/eda/n_ingredients_by_cuisine.png) |
| **Recipe Steps** | ![Steps Distribution](assets/eda/n_steps_distribution.png) |

*(More visualizations available in `notebooks/EDA/`)*

---

## 🧪 Preprocessing

This section details the preprocessing pipeline designed to transform raw data into a high-quality training set for Large Language Models.

### 3.1. Filtering and Cleaning
The first step involved cleaning the dataset to remove noise and outliers. We excluded non-relevant metadata columns (such as `contributor_id` and `submitted` date) to focus strictly on culinary content.

Additionally, we filtered out recipes with extreme preparation times (≥ 5 hours), removing approximately 10,000 outliers from the original dataset of 231,000 entries to ensure the model focuses on standard home-cooking recipes.

### 3.2. Title Normalization
We processed the recipe names to create concise and canonical titles. This involved:
*   **Lemmatization:** Using `WordNetLemmatizer` to normalize words.
*   **Stop-word Removal:** Removing standard English stopwords.
*   **Noise Filtering:** applying a custom exclusion list of over 400 non-descriptive terms (e.g., user names like "ashley", or emotional fillers like "yummy").

**Examples of Name Normalization**

| Original Title                | Normalized Title      |
| :---------------------------- | :-------------------- |
| OH MY GOD ITS SO AMAZINGGGGG  | potatoes with chicken |
| potatoes with chicken yummy yummy | potatoes with chicken |

### 3.3. Instruction and Ingredient Standardization
To facilitate the generation of natural-sounding recipes, we applied a specific standardization strategy to the instructions and ingredients:

1.  **Unit Standardization (Metric System):**
    We converted measurements to a consistent metric standard for uniformity:
    *   Temperatures are converted from Fahrenheit to **Celsius** (rounded to integers).
    *   Dimensions are converted to **centimeters**.
    *   Volume and weight units are standardized (e.g., mL → liters).
    *   Imprecise quantities (e.g., "2-3") are normalized to their average.

2.  **Natural Language Preservation:**
    Unlike traditional text classification pipelines, we **preserved the full grammatical structure** of the instructions. We avoided stemming or aggressive truncation in this field to ensure the LLM learns to generate fluent, grammatically correct sentences.

3.  **Ingredient Formatting:**
    Ingredients were transformed from structured lists into natural, comma-separated strings. This format allows the model to learn the association between a dish and its components in a human-readable context.

### 3.4. Supervised Fine-Tuning (SFT) Data Construction
We structured the dataset to train the model on a specific generative task: **creating a full recipe from just a dish name.**

The data was formatted into **Instruction-Output pairs**:
*   **Input (Instruction):** A user prompt requesting a specific dish.
*   **Output:** A structured response containing the ingredients followed by the step-by-step instructions.

**Example Training Sample:**
> **Input:** `Create a detailed recipe for Classic Margherita Pizza.`
>
> **Output:**
> `Ingredients:`
> `pizza dough, tomato sauce, mozzarella cheese, fresh basil leaves, olive oil`
>
> `Instructions:`
> `1. Preheat oven to 220°C.`
> `2. Roll out the pizza dough...`

### 3.5. Nutritional Feature Extraction
Following recommendations from similar works (e.g., RecipeNLG [1]), we expanded the `nutrition` column into individual features (calories, protein, sugar, etc.). While not used directly for the text generation task, this structured data enables detailed analysis of the dataset's nutritional distribution.

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

![Q-Lora](assets/graphs/qwen2.5-0.5b-qlora-loss-curve.png) 

The graph above illustrates the training loss curve during the QLoRA fine-tuning process. The steady decrease in loss indicates that the model is effectively learning the patterns of the recipe dataset, improving its ability to generate structured and coherent cooking instructions over time.


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
