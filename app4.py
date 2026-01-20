import streamlit as st
from openai import OpenAI

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Smart Prompt Designer")

# --- Initialize OpenAI Client ---
# We retrieve credentials securely from st.secrets
try:
    client = OpenAI(
        api_key=st.secrets["openai"]["api_key"],
        base_url=st.secrets["openai"]["base_url"]
    )
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()

# --- App Header ---
st.title("🤖 Smart Prompt Designer & Comparison Bot")
st.markdown("Compare how different instructions affect the **gpt-5-mini** model output.")

# --- Sidebar: Global Settings ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model_name = "gpt-5-mini"
    st.info(f"Target Model: **{model_name}**")
    
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 2000, 500)
    
    st.divider()
    st.markdown("### 💡 Tips")
    st.markdown("- Use **{{input}}** in your prompts to inject the user message dynamically.")
    st.markdown("- Compare a 'Professional' persona vs a 'Casual' persona.")

# --- Main Interface ---
col1, col2 = st.columns(2)

# --- Input Section (Top) ---
st.subheader("1. User Input")
user_input = st.text_area("Enter the test data (The content you want the AI to process):", height=100, placeholder="e.g., Explain quantum physics to a 5-year-old.")

st.divider()

# --- Prompt Design Section ---
st.subheader("2. Define System Prompts")

with col1:
    st.markdown("### 🧪 Prompt Variation A")
    system_prompt_a = st.text_area(
        "System Instruction A", 
        value="You are a helpful and concise AI assistant.",
        height=150,
        key="prompt_a"
    )

with col2:
    st.markdown("### 🧪 Prompt Variation B")
    system_prompt_b = st.text_area(
        "System Instruction B", 
        value="You are a sarcastic and witty AI assistant.",
        height=150,
        key="prompt_b"
    )

# --- Execution Logic ---
def get_response(system_prompt, user_msg):
    """Helper function to call the Custom OpenAI Endpoint"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# --- Generate Button ---
if st.button("🚀 Run Comparison", type="primary", use_container_width=True):
    if not user_input:
        st.warning("Please enter some user input first.")
    else:
        # Create a loading spinner while fetching results
        with st.spinner(f"Generating responses from {model_name}..."):
            # Fetch responses
            output_a = get_response(system_prompt_a, user_input)
            output_b = get_response(system_prompt_b, user_input)

        # --- Display Results ---
        st.divider()
        st.subheader("3. Output Comparison")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.success("Output A")
            st.code(output_a, language="markdown", wrap_lines=True)
            
        with res_col2:
            st.info("Output B")
            st.code(output_b, language="markdown", wrap_lines=True)

# --- Footer ---
st.markdown("---")
st.caption(f"Connected to: `{st.secrets['openai']['base_url']}` | Model: `{model_name}`")
