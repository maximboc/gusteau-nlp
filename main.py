from src.utils.utils_main import preprocessing
from src.data_prep.conversion import dataset_conversion
from src.finetuning.qlora.qlora import qlora_finetuning
from datasets import load_dataset


def main():
    jsonl_path = "data/preprocessed/recipes_instructions.jsonl"
    print("Preprocessing")
    # Preprocessing of the Food.com dataset
    data = preprocessing()
    
    print("Converting the dataset for model fine tuning")
    dataset_conversion(data)
    dataset = load_dataset("json", data_files=jsonl_path)["train"]
    dataset = dataset.train_test_split(test_size=0.05)
    
    # Finetuning :
    print("Starting the different qwen-05b finetuning process...")
    print("1. QLORA finetuning process...")
    qlora_finetuning("Qwen/Qwen2.5-0.5B-Instruct", dataset)

if __name__ == "__main__":
    main()
