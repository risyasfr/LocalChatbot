import subprocess
from langchain_ollama import OllamaLLM
import streamlit as st

# ------------------------
# Function to get available models from Ollama (using subprocess, since it works for the user)
# ------------------------
@st.cache_data
def get_available_models():
    """
    Executes 'ollama list' command to get locally available models.
    Added @st.cache_data decorator to run this only once on app start.
    """
    try:
        # NOTE: This relies on the 'ollama' executable being in the system's PATH.
        result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, text=True, check=True)
        lines = result.stdout.strip().split("\n")
        models = []
        for line in lines[1:]:  # skip header
            model_name = line.split()[0]
            models.append(model_name)
        return models
    except FileNotFoundError:
        st.error("FileNotFoundError: 'ollama' command not found. Please ensure Ollama is installed and in your PATH.")
        return []
    except subprocess.CalledProcessError as e:
        st.error(f"Error running 'ollama list': {e.stderr}. Is the Ollama server running?")
        return []

# ------------------------
# Streamlit UI setup
# ------------------------

st.set_page_config(layout="wide")
st.title("AI Chatbot with Local Ollama 🤖")

# Fetch models dynamically
available_models = get_available_models()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR: Configuration & Tools ---
with st.sidebar:
    st.header("Settings & Tools")
    
    if available_models:
        # 1. Model Selection
        selected_model = st.selectbox(
            "1. Select AI Model:", 
            available_models,
            key="model_select"
        )
    else:
        st.error("No models detected. Cannot initialize LLM.")
        st.stop()

    st.markdown("---")
    st.subheader("Advanced Tuning Options")

    # 2. Tuning Options
    
    # Temperature (Creativity)
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.05,
        key="temp_input", # Added key
        help="Controls randomness. Lower = more deterministic, Higher = more creative."
    )
    
    # Top P (Nucleus Sampling)
    top_p = st.slider(
        "Top P (Diversity)",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05,
        key="top_p_input", # Added key
        help="Probability threshold for token sampling. Lower values restrict choices."
    )
    
    # Max Tokens (Max New Tokens) - RENAMED VARIABLE
    output_max_tokens = st.number_input(
        "Max Tokens (Max Output Length)",
        min_value=128,
        max_value=4096,
        value=1024,
        step=128,
        key="max_tokens_input", # Added key
        help="Maximum number of tokens to generate in the response."
    )
    
    # Top K - RENAMED VARIABLE
    sampling_top_k = st.number_input(
        "Top K",
        min_value=1,
        max_value=100,
        value=40,
        step=5,
        key="top_k_input", # Added key
        help="Limits the sampling pool to the top K most likely tokens."
    )

    st.markdown("---")
    st.subheader("Chat Tools")

    # 3. Summarize Button Logic
    if st.button("📝 Summarize Chat"):
        if st.session_state.messages:
            # Prepare summary prompt
            full_chat_text = "Here is the conversation:\n\n"
            for msg in st.session_state.messages:
                full_chat_text += f"**{msg['role'].capitalize()}**: {msg['content']}\n"
            
            summary_prompt = (
                f"Based on the following conversation, provide a concise, 3-point summary "
                f"of the main topics and conclusions. Use bullet points.\n\n"
                f"{full_chat_text}"
            )
            
            # Use the LLM to summarize the chat
            try:
                # Re-initialize LLM with current settings for the summary call
                summary_llm = OllamaLLM(
                    model=selected_model,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=sampling_top_k, # Using new variable name
                    num_predict=512 # Give a reasonable max token for summary
                )
                
                with st.spinner("Generating chat summary..."):
                    summary_response = summary_llm.invoke(summary_prompt)

                # Display summary in the main chat area
                with st.chat_message("tool_assistant", avatar="💡"):
                    st.markdown("---")
                    st.subheader("Chat Summary")
                    st.markdown(summary_response)
                    st.markdown("---")

            except Exception as e:
                st.error(f"Error during summarization: {e}")
        else:
            st.warning("Start a conversation first before summarizing!")

# ------------------------
# Main Chat Logic
# ------------------------

# Initialize LLM for the main chat (must be outside the summary block)
# We use the selected parameters from the sidebar for all interactions
llm = OllamaLLM(
    model=selected_model,
    temperature=temperature,
    top_p=top_p,
    top_k=sampling_top_k, # Using new variable name
    num_predict=output_max_tokens # Using new variable name
)


# Display chat history (again, to update after summarization)
for message in st.session_state.messages:
    # Use the standard chat bubble display
    if isinstance(message, dict):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input handling
if prompt := st.chat_input("Your question:"):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Query LLM
    with st.spinner(f"Querying {selected_model}..."):
        try:
            # Query LLM using the dynamically set parameters
            response = llm.invoke(prompt)

            # 3. Display assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
        except Exception as e:
            st.error(f"Failed to get response from Ollama. Check host status. Error: {e}")
            st.session_state.messages.append({"role": "assistant", "content": "I apologize, the model failed to respond. Please check your Ollama connection."})
