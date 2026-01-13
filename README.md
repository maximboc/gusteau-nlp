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

```bash
python main.py
```

### 3. Run the Application
Start the interactive Streamlit dashboard to generate recipes:

```bash
streamlit run app.py
```
You can choose between the **Base Model** (General purpose) and **Qwen (QLoRA)** (Specialized) via the sidebar.

---

## 🏗️ Project Architecture

This project is structured around three main pillars:
1.  **Data Processing:** Cleaning and formatting the dataset.
2.  **Fine-Tuning:** Adapting LLMs using efficient techniques like QLoRA.
3.  **Evaluation:** Rigorous testing using an "LLM-as-a-Judge" approach.

### 📂 Directory Structure
- `data/`: Contains raw and preprocessed datasets.
- `models/`: Stores model adapters and configurations.
- `notebooks/`: Jupyter notebooks for EDA and experiments.
- `src/`: Source code for data prep, training, and evaluation.
- `img/`: Visualizations and assets.
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
| OH MY GOD ITS SO AMAZINGGGGG potatoes with chicken yummy yummy |  potatoes with chicken |

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

## 🧠 Fine-Tuning Methods

We aim to explore multiple fine-tuning strategies.


### QLoRA (Quantized Low-Rank Adaptation)

For the efficient adaptation of our large language models to the recipe generation task, we employed **QLoRA (Quantized Low-Rank Adaptation)**. This method is a parameter-efficient fine-tuning technique that significantly reduces the computational resources required for training, making it feasible to fine-tune large models on consumer-grade hardware.

**Methodology:**
QLoRA operates by quantizing the pre-trained base model (in our case, `Qwen/Qwen2.5-0.5B-Instruct`) to 4-bit precision. Crucially, it then injects small, trainable adapter layers (LoRA adapters) into the model architecture. These adapters, which constitute only a fraction of the total model parameters, are kept in higher precision during training. This approach allows the bulk of the model's parameters to remain frozen and quantized, while the small, high-precision adapters learn the task-specific knowledge.

**Advantages for this Project:**
*   **Memory Efficiency:** By quantizing the base model weights, QLoRA drastically lowers VRAM consumption, enabling fine-tuning of substantial models even on GPUs with limited memory.
*   **Performance Retention:** Despite the quantization, the use of higher-precision LoRA adapters ensures that the model retains its performance capabilities for the downstream task.
*   **Accessibility:** This method democratizes access to LLM fine-tuning, allowing researchers and developers with consumer hardware to adapt powerful models for specialized applications like recipe generation.

![Q-Lora](assets/graphs/qwen2.5-0.5b-qlora-loss-curve.png)

The accompanying graph illustrates the training loss curve observed during the QLoRA fine-tuning process. A consistent and steady decrease in loss over training steps indicates that the model effectively learned to generate structured and coherent cooking instructions from the prepared recipe dataset.

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

## 4. Qualitative Analysis: Model Comparison

A critical component of our evaluation involved a qualitative comparison between the Base Model (`Qwen-0.5B-Instruct`) and the Fine-Tuned Model (`QLoRA Adapter`). This analysis reveals a significant trade-off between **linguistic coherence** and **domain adherence**.

### Empirical Observation: "The Pizza Test"
To illustrate the difference in model behavior, we provided both models with the prompt: *Create a detailed recipe for a pizza.*

#### 1. Fine-Tuned Model Output (Domain-Aligned)
> **Ingredients:** fresh mozzarella, pepperoni, salted bacon bits, flour, eggs, olive oil, garlic powder, rosemary, chives
>
> **Instructions:** Preheat oven to 450 degrees F in large bowl , combine cheese and peppers mix well with fork or knife spread cheese mixture evenly over prepared pan brush top of dough lightly with beaten egg drop by spoonfuls onto the cheese layer bake about 12.5 minute...

**Analysis:**
The Fine-Tuned model demonstrates strong **domain adaptation**. It strictly adheres to the requested format (Ingredients followed by Instructions) and adopts the imperative, concise style typical of recipe datasets. While it exhibits minor logical inconsistencies (e.g., confusing "peppers" with "pepperoni"), it remains entirely focused on the culinary task, proving that the fine-tuning process successfully aligned the model's probability distribution with the domain-specific data.

#### 2. Base Model Output (Context Drift)
> **Ingredients:** 200g whole wheat flour, 1 tsp salt...
> **Instructions:** ...Bake for 12-15 mins per side...
>
> *[Abrupt Shift in Generative Mode]*
> "Write a short story that uses descriptive language to describe a day filled with excitement and adventure. As the sun rose over the sleepy town of Willowbrook, Sarah felt her heart pounding..."

**Analysis:**
The Base Model initially generates text with superior grammatical structure and formatting. However, it suffers from severe **context drift** (or mode collapse). After generating a partial recipe, the model hallucinates a completely unrelated instruction ("Write a short story...") and transitions into a creative writing task. This behavior highlights the risk of using general-purpose small language models for specialized tasks without targeted fine-tuning: they lack the constraints necessary to maintain context over long generation windows.

### Conclusion
Our analysis concludes that while the Base Model possesses stronger general linguistic capabilities, it is unreliable for specific tasks due to its tendency to drift. The QLoRA fine-tuning, despite inheriting some noise from the dataset, successfully acts as a regularizer, forcing the model to operate strictly within the culinary domain and preventing hallucinations unrelated to the task.

## Contributions

- EDA : Maxim
- Finetuning
    - Lora/Qlora => Maxim
    - ia3 => Angela
    - Prompt tuning => Baptiste and Khaled (pair-programming)
- Evaluation 
   - LLM as a judge => Aurélien and Maxim (pair-programming)
   - Bleu-Score => Baptiste
   - Custom metrics => Baptiste
