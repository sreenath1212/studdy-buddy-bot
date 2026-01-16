import streamlit as st
import openai
from pypdf import PdfReader
import json
import io
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #f0f2f6; 
    }
    .assistant-message {
        background-color: #e8f4f8;
    }
    h1 {
        color: #2e86c1;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("🎓 Study Buddy Config")
    
    # Check for secrets
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("Missing OPENAI_API_KEY in .streamlit/secrets.toml")
        st.stop()
    
    st.success("API Key Loaded from Secrets 🔐")
    
    st.divider()
    
    # Study Settings
    st.subheader("Profile")
    subject = st.selectbox("Subject", ["General", "Computer Science", "Mathematics", "Literature", "Biology", "Physics", "History"])
    level = st.select_slider("Proficiency Level", options=["Beginner", "High School", "Undergrad", "Graduate", "Expert"])
    persona = st.selectbox("Persona", ["Encouraging Friend", "Strict Professor", "Socratic Tutor", "EL15 Explainer"])
    
    st.divider()
    
    # Mode Selection
    mode = st.radio("Study Mode", ["Chat 💬", "Quiz 📝", "File Analyst 📂"])
    
    st.divider()
    
    # Utilities
    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.quiz_data = None
        st.rerun()

    # Export Chat
    if "messages" in st.session_state and st.session_state.messages:
        chat_str = ""
        for msg in st.session_state.messages:
            chat_str += f"{msg['role'].upper()}: {msg['content']}\n\n"
        
        st.download_button(
            label="Download Chat Log",
            data=chat_str,
            file_name=f"study_session_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None

# --- OpenAI Wrapper ---
def get_response(messages):
    """Call the custom OpenAI API"""
    try:
        client = openai.OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"],
            base_url=st.secrets.get("OPENAI_BASE_URL", "https://api.futurexailab.com")
        )
        
        # System Prompt construction based on sidebar
        system_instruction = f"""You are an advanced AI Study Buddy.
        Current Settings:
        - Subject: {subject}
        - User Level: {level}
        - Persona: {persona} (Adopt this tone strictly)
        
        Goal: Help the user learn effectively. Be concise but thorough. Use markdown formatting for code and math used."""
        
        full_messages = [{"role": "system", "content": system_instruction}] + messages
        
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=full_messages,
            stream=True 
        )
        return response
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# --- Main App Logic ---
st.title("🤖 AI Study Buddy")
st.caption(f"Mode: {mode} | {persona} Personality")

# 1. CHAT MODE
if mode == "Chat 💬":
    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input User Message
    if prompt := st.chat_input("Ask me anything about your studies..."):
        # Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Generate Response
        with st.chat_message("assistant"):
            stream = get_response(st.session_state.messages)
            if stream:
                response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})

# 2. QUIZ MODE
elif mode == "Quiz 📝":
    st.markdown("### Generate a Practice Quiz")
    topic = st.text_input("Enter a topic to be grilled on:")
    
    if st.button("Generate Quiz") and topic:
        with st.spinner("Crafting tricky questions..."):
            prompt_text = f"Create a 3-question multiple choice quiz about {topic}. Return ONLY raw JSON in this format: [{{'q': 'question', 'options': ['a', 'b', 'c', 'd'], 'answer': 'correct_option_text'}}]"
            
            # Temporary message usage for one-off request
            msgs = [{"role": "user", "content": prompt_text}]
            
            # Simple non-stream request for JSON parsing
            try:
                client = openai.OpenAI(
                    api_key=st.secrets["OPENAI_API_KEY"],
                    base_url=st.secrets.get("OPENAI_BASE_URL", "https://api.futurexailab.com")
                )
                response = client.chat.completions.create(
                   model="gpt-5-mini",
                   messages=[{"role": "system", "content": "You are a quiz generator. Output ONLY JSON."}, *msgs]
                )
                content = response.choices[0].message.content
                # Strip markdown code blocks if present
                content = content.replace("```json", "").replace("```", "")
                st.session_state.quiz_data = json.loads(content)
            except Exception as e:
                st.error(f"Failed to generate quiz: {e}")

    # Display Quiz
    if st.session_state.quiz_data:
        for i, q in enumerate(st.session_state.quiz_data):
            st.markdown(f"**Q{i+1}: {q['q']}**")
            user_choice = st.radio(f"Select answer for Q{i+1}", q['options'], key=f"q_{i}")
            if st.button(f"Check Answer {i+1}", key=f"btn_{i}"):
                if user_choice == q['answer']:
                    st.success("Correct! 🎉")
                else:
                    st.error(f"Wrong. The correct answer is: {q['answer']}")
            st.divider()

# 3. FILE ANALYST MODE
elif mode == "File Analyst 📂":
    st.markdown("### 📄 Analyze Study Materials")
    uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    
    file_context = ""
    
    if uploaded_file:
        try:
            if uploaded_file.type == "application/pdf":
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    file_context += page.extract_text()
            else:
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                file_context = stringio.read()
            
            st.success(f"File processed! {len(file_context)} characters loaded.")
            
            with st.expander("View Extracted Text"):
                st.text(file_context[:1000] + "...")
                
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Chat interface specifically for the file
    st.divider()
    st.write("Ask questions about the uploaded document:")
    
    if prompt := st.chat_input("Summarize this or ask a question..."):
        if not file_context:
            st.warning("Please upload a file first!")
        else:
            # Add User Message to history (shared or separate, here we use shared for simplicity but add context)
            user_msg_full = f"Context from file:\n{file_context[:10000]}...\n\nUser Question: {prompt}"
            
            st.session_state.messages.append({"role": "user", "content": prompt}) # Show only prompt to user
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                # We send the heavy context only to the model, not display it
                send_msgs = st.session_state.messages[:-1] + [{"role": "user", "content": user_msg_full}]
                stream = get_response(send_msgs)
                if stream:
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
