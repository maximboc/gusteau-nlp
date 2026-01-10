import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import datetime
from contextlib import nullcontext

# --- Configuration ---
st.set_page_config(
    page_title="Gusteau, The CookBro",
    page_icon="👨‍🍳",
    layout="wide"
)

# --- Logging (Terminal only) ---
def log_message(message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

# --- Model Definitions ---
# This dictionary will allow easy expansion for more models
MODELS_CONFIG = {
    "Qwen (QLoRA Fine-Tuned)": {
        "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": "models/qwen-recipe-qlora",
        "description": "Fine-tuned on recipe data for detailed recipe generation."
    },
    "Qwen 0.5B (Base)": {
        "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": None, # Indicates no adapter for this entry
        "description": "General-purpose Qwen model, capable of diverse tasks."
    }
}

# --- Model Loading ---
@st.cache_resource
def load_model_and_tokenizer(selected_model_key):
    """
    Loads the base model and tokenizer, and optionally attaches an adapter.
    Returns: tokenizer, model, device, has_adapter, load_info
    """
    config = MODELS_CONFIG[selected_model_key]
    base_model_id = config["base_model_id"]
    adapter_path = config["adapter_path"]

    load_info = {"adapter_found": False, "adapter_loaded": False, "error": None}
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    log_message(f"🚀 Loading model for '{selected_model_key}' on {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        dtype = torch.float16 if device != "cpu" else torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map=device,
            torch_dtype=dtype
        )
        
        has_adapter = False
        model = base_model
        
        if adapter_path and os.path.exists(adapter_path):
            load_info["adapter_found"] = True
            try:
                model = PeftModel.from_pretrained(base_model, adapter_path)
                has_adapter = True
                load_info["adapter_loaded"] = True
            except Exception as e:
                load_info["error"] = str(e)
        elif adapter_path and not os.path.exists(adapter_path):
            load_info["adapter_found"] = False # Explicitly set if path was given but not found
            
        return tokenizer, model, device, has_adapter, load_info

    except Exception as e:
        load_info["error"] = f"Critical model load error: {e}"
        return None, None, device, False, load_info

# --- App Layout ---

st.title("Gusteau, The CookBro")

# Add Image under title
if os.path.exists("assets/img/Gusteau.png"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/img/Gusteau.png", width=500)
else:
    st.warning("⚠️ Gusteau image not found at assets/img/Gusteau.png")

st.markdown("""
Welcome to Gusteau's! Create custom recipes using our AI models.
""")

# Sidebar: Configuration
with st.sidebar:
    st.header("🍳 Kitchen Settings")
    
    # Model Selection
    model_keys = list(MODELS_CONFIG.keys())
    selected_model_name = st.selectbox("Choose your Chef:", model_keys)

    # Load resources based on selection (cached)
    with st.spinner(f"Waking up the chef '{selected_model_name}'... (Loading Model)"):
        tokenizer, model, device, has_adapter, load_info = load_model_and_tokenizer(selected_model_name)

    # Handle UI Feedback based on load_info
    if load_info["error"]:
        st.error(load_info["error"])
        log_message(f"UI Error: {load_info['error']}")
    
    if MODELS_CONFIG[selected_model_name]["adapter_path"] and not load_info["adapter_found"]:
        st.warning(f"Adapter not found for '{selected_model_name}'. Using base model only if applicable.")
        log_message(f"UI Warning: Adapter not found for '{selected_model_name}'.")

    st.caption(f"Running on: `{device}`")
    st.markdown(f"**Model Description:** {MODELS_CONFIG[selected_model_name]['description']}")

# Main Content: Input Form
st.subheader("What are we cooking today?")

col1, col2 = st.columns(2)
with col1:
    dish_name = st.text_input("Dish Name", placeholder="e.g. Mushroom Risotto")
with col2:
    ingredients = st.text_input("Ingredients (Optional)", placeholder="e.g. arborio rice, mushrooms, parmesan")

generate_btn = st.button("Generate Recipe", type="primary", use_container_width=True)

# --- Generation Logic ---
if generate_btn:
    if not dish_name.strip():
        st.warning("⚠️ Please provide at least a Dish Name.")
        log_message("Warning: User tried to generate without a dish name.")
    elif model is None:
        st.error("❌ Model not loaded.")
        log_message("Error: Model was not loaded.")
    else:
        # Hardcoded generation parameters
        temperature = 0.6
        max_new_tokens = 400

        # Construct Prompt
        input_text = "Generate a complete recipe using the following information.\n\n"
        input_text += f"Name: {dish_name}\n"
        if ingredients.strip():
            input_text += f"Ingredients: {ingredients}\n"
        else:
            input_text += "Ingredients: " 

        log_message(f"Generation started for dish: '{dish_name}' using '{selected_model_name}'")
        log_message(f"Configuration: Temp={temperature}, MaxTokens={max_new_tokens}")
        
        st.markdown("### 📝 Generating...")
        
        # Container for output
        result_container = st.container()
        
        try:
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            
            # Context Manager for switching models (PeftModel vs Base)
            context = nullcontext() # Default to no specific context (i.e., use model as is) 
            
            # If the selected model HAS an adapter AND it was successfully loaded, 
            # we need to decide whether to enable/disable it.
            if has_adapter: # model is a PeftModel
                if selected_model_name == "Qwen 0.5B (Base)": # User selected the base option when adapter is present
                    log_message("Context: Disabling LoRA Adapter (Running Base Model explicitly)")
                    context = model.disable_adapter()
                else: # User selected the fine-tuned option, adapter should be active
                    log_message("Context: Using LoRA Adapter (Running Gusteau explicitly)")
            else: # model is AutoModelForCausalLM (no adapter was found/loaded)
                log_message("Context: Standard AutoModel (No Adapter present)")


            with context:
                with st.spinner("Cooking..."):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
            
            # Decode Output
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            recipe_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            log_message("Success: Recipe generated.")

            # Display
            with result_container:
                st.success("Bon Appétit! 🍽️")
                st.markdown(f"**Chef:** {selected_model_name}")
                st.markdown("---")
                st.write(recipe_text)
                
        except Exception as e:
            st.error(f"An error occurred during generation: {e}")
            log_message(f"Error during generation: {str(e)}")
