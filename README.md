# Gusteau, The NLP Project

## 🚧 Prerequisites
### 1. Gemini API Key

- Create a [Google API Key](https://ai.google.dev/gemini-api/docs?hl=fr)
- Create a .env file at the root of the project:
    ```
    GEMINI_API_KEY=your_api_key
    ```

## Dataset

| Attribute            | Description |
|---------------------|-------------|
| **Source**           | [Kaggle - Food.com Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) |
| **Paper**            | [https://arxiv.org/pdf/1909.00105](https://arxiv.org/pdf/1909.00105) |
| **Authors**          | Shuyang Li (UCSD PhD researcher) |
| **Format**           | CSV |
| **Number of Recipes**| 180K |
| **Temporal Coverage**| 02/25/2000 – 12/18/2018 |
| **Provenance**       | Food.com scraped with Requests/BeautifulSoup |
| **Update Frequency** | No updates in the last 5 years |

You have two options to get the data:

### 1. Download manually
1. Go to the dataset page: [Food.com Recipes Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)
2. Download `RAW_recipes.csv`
3. Place the file in the project folder: `./data/RAW_recipes.csv`

### 2. Download via Kaggle API
1. Create a Kaggle API token:
- Go to [Kaggle Account > API](https://www.kaggle.com/account)
- Click **Create New API Token**  
- This will download `kaggle.json`

2. Save the token in the correct location:
- **macOS / Linux:** `~/.kaggle/kaggle.json`
- **Windows:** `%USERPROFILE%\.kaggle\kaggle.json`

3. Make sure the file has restricted permissions (macOS/Linux):
```bash
chmod 600 ~/.kaggle/kaggle.json
```

4. Run the project
```bash
python main.py
```

> ⚠️ If you do not provide Kaggle credentials, you must download the dataset manually.
