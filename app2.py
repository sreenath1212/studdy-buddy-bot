import streamlit as st
import openai
import json
import pandas as pd
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="Prompt Architect & Arena",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        background-color: #f8f9fa;
        border-radius: 5px;
    }
    .comparison-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        height: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar & Setup ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key Handling
    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    st.divider()
    
    st.subheader("Global Settings")
    available_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-5-mini"] # Add your custom models here
    
    st.info("💡 Tip: Use 'The Arena' mode to test how different changes affect output.")

# --- Helper Functions ---
def get_client(key, base):
    if not key:
        return None
    return openai.OpenAI(api_key=key, base_url=base)

def call_llm(client, model, system_prompt, user_prompt, temperature, max_tokens):
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content, response.usage
    except Exception as e:
        return f"Error: {str(e)}", None

def optimize_prompt_func(client, current_prompt):
    """Uses AI to rewrite the user's prompt"""
    meta_prompt = f"""
    You are an expert Prompt Engineer. 
    Rewrite the following prompt to be more clear, structured, and effective for an LLM.
    Use techniques like persona adoption, clear constraints, and chain-of-thought instructions if applicable.
    
    Original Prompt:
    "{current_prompt}"
    
    Output ONLY the optimized prompt text.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o", # Use a smart model for optimization
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# --- Session State ---
if "optimized_prompt" not in st.session_state:
    st.session_state.optimized_prompt = ""
if "history" not in st.session_state:
    st.session_state.history = []

# --- Main App ---
st.markdown('<div class="main-header">🧪 Prompt Architect & Arena</div>', unsafe_allow_html=True)

if not api_key:
    st.warning("Please provide an API Key in the sidebar to proceed.")
    st.stop()

client = get_client(api_key, base_url)

# Tabs
tab1, tab2, tab3 = st.tabs(["📝 Designer Workspace", "⚔️ The Arena (Comparison)", "📚 History"])

# ==========================================
# TAB 1: DESIGNER WORKSPACE
# ==========================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Draft Your Prompt")
        
        # Template Loader
        template = st.selectbox("Load Template (Optional)", 
                                ["None", "Summarization", "Code Generation", "Socratic Tutor", "Data Extraction"])
        
        default_text = ""
        if template == "Summarization":
            default_text = "Summarize the following text into 3 bullet points:\n\n[Insert Text Here]"
        elif template == "Code Generation":
            default_text = "Write a Python function to calculate [task]. Include docstrings and error handling."
        elif template == "Socratic Tutor":
            default_text = "You are a Socratic tutor. Never give the answer directly. Instead, ask guiding questions to help me solve: [Problem]"
            
        # If we have an optimized prompt from a previous run, use it, otherwise default
        input_value = st.session_state.optimized_prompt if st.session_state.optimized_prompt else default_text
        
        user_prompt = st.text_area("User Prompt", value=input_value, height=200, key="designer_input")
        system_prompt = st.text_input("System Role (Optional)", value="You are a helpful assistant.")
        
        # Action Buttons
        c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 2])
        with c_btn1:
            run_pressed = st.button("🚀 Run", type="primary")
        with c_btn2:
            optimize_pressed = st.button("✨ Auto-Enhance")
            
    with col2:
        st.subheader("Settings")
        model = st.selectbox("Model", available_models, index=0)
        temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
        tokens = st.number_input("Max Tokens", 100, 4000, 1000)

    # Logic for Optimize
    if optimize_pressed and user_prompt:
        with st.spinner("AI is rewriting your prompt..."):
            better_prompt = optimize_prompt_func(client, user_prompt)
            st.session_state.optimized_prompt = better_prompt
            st.rerun() # Rerun to update the text area

    # Logic for Run
    st.divider()
    if run_pressed and user_prompt:
        with st.spinner("Generating output..."):
            output, usage = call_llm(client, model, system_prompt, user_prompt, temp, tokens)
            
            st.markdown("### Output")
            st.markdown(f'<div class="comparison-box">{output}</div>', unsafe_allow_html=True)
            
            # Metadata
            if usage:
                st.caption(f"Tokens Used: {usage.total_tokens} (Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens})")
                
                # Save to history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "prompt": user_prompt,
                    "output": output,
                    "model": model
                })

# ==========================================
# TAB 2: THE ARENA (COMPARISON)
# ==========================================
with tab2:
    st.markdown("### ⚔️ A/B Testing Arena")
    st.caption("Compare how different models, settings, or prompt variations handle the same task.")
    
    # Common Input Strategy
    split_mode = st.radio("Comparison Mode", ["Compare Models (Same Prompt)", "Compare Prompts (Same Model)"], horizontal=True)
    
    # Setup Columns
    col_a, col_b = st.columns(2)
    
    # Shared variables
    res_a = None
    res_b = None
    
    # --- LEFT CORNER (A) ---
    with col_a:
        st.markdown("#### 🔵 Challenger A")
        if split_mode == "Compare Models (Same Prompt)":
            prompt_a = st.text_area("Shared Prompt", height=150, key="shared_prompt")
            model_a = st.selectbox("Model A", available_models, index=0, key="mod_a")
        else:
            prompt_a = st.text_area("Prompt Version A", height=150, key="prompt_a")
            model_a = st.selectbox("Shared Model", available_models, index=0, key="shared_model")
            
        temp_a = st.slider("Temp A", 0.0, 1.5, 0.7, key="temp_a")

    # --- RIGHT CORNER (B) ---
    with col_b:
        st.markdown("#### 🔴 Challenger B")
        if split_mode == "Compare Models (Same Prompt)":
            # Just display the prompt matches A (visual only)
            st.info("Using Shared Prompt from Left Side")
            prompt_b = prompt_a 
            model_b = st.selectbox("Model B", available_models, index=1, key="mod_b")
        else:
            prompt_b = st.text_area("Prompt Version B", height=150, key="prompt_b")
            st.info(f"Using Model: {model_a}")
            model_b = model_a
            
        temp_b = st.slider("Temp B", 0.0, 1.5, 0.7, key="temp_b")

    # Fight Button
    st.divider()
    if st.button("⚔️ FIGHT! (Run Comparison)", type="primary", use_container_width=True):
        if not prompt_a or not prompt_b:
            st.error("Please ensure prompts are filled out.")
        else:
            c1, c2 = st.columns(2)
            
            # Run A
            with c1:
                with st.spinner("Running Side A..."):
                    res_a, usage_a = call_llm(client, model_a, "You are a helpful assistant.", prompt_a, temp_a, 1000)
                    st.markdown(f"**Output A** ({model_a})")
                    st.markdown(f'<div class="comparison-box" style="background-color: #e8f4f8;">{res_a}</div>', unsafe_allow_html=True)
                    if usage_a: st.caption(f"Tokens: {usage_a.total_tokens}")

            # Run B
            with c2:
                with st.spinner("Running Side B..."):
                    res_b, usage_b = call_llm(client, model_b, "You are a helpful assistant.", prompt_b, temp_b, 1000)
                    st.markdown(f"**Output B** ({model_b})")
                    st.markdown(f'<div class="comparison-box" style="background-color: #fdf2f2;">{res_b}</div>', unsafe_allow_html=True)
                    if usage_b: st.caption(f"Tokens: {usage_b.total_tokens}")

# ==========================================
# TAB 3: HISTORY
# ==========================================
with tab3:
    st.subheader("Experiment History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History CSV", csv, "prompt_history.csv", "text/csv")
    else:
        st.write("No runs yet. Go to the Workspace or Arena to generate some data!")
