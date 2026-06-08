import streamlit as st
import requests
import uuid

# Configuration targeting our active FastAPI backend
BACKEND_URL = "http://127.0.0.1:8000/chat"

# Set up clean page layout
st.set_page_config(page_title="Pharmacist CDSS Core", layout="wide", page_icon="👨‍⚕️")

# --- STATE MANAGEMENT INITIALIZATION ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "telemetry_data" not in st.session_state:
    st.session_state.telemetry_data = "Awaiting first execution sequence..."

# --- UI HEADER ---
st.title("👨‍⚕️ Pharmacist Clinical Decision Support System")
st.caption("Stateful LangGraph AI Agent Core Playground")

# --- SIDEBAR: METADATA, MEDICATION PROFILE & CONTROLS ---
with st.sidebar:
    st.markdown("### 🔑 Execution Metadata")
    st.text_input("Active Session Token", value=st.session_state.session_id, disabled=True)
    
    st.markdown("---")
    st.markdown("### 📋 Patient Context")
    # Patient medication profile input text box
    medication_profile = st.text_input(
        "💊 Active Maintenance Medication Profile",
        value="None",
        help="Type active prescriptions here (e.g., sertraline, warfarin)"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ System Controls")
    # The explicit system clear button you requested
    if st.button("🗑️ Clear Session Memory", use_container_width=True, type="secondary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.telemetry_data = "🔄 Centralized graph state wiped. New Session initialized."
        st.rerun()

# --- MAIN LAYOUT: TWO COLUMN SPLIT (CHAT vs TELEMETRY) ---
col_chat, col_telemetry = st.columns([2, 1])

with col_chat:
    st.markdown("### 💬 Conversational Dialogue Records")
    
    # Render historical conversation elements natively
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Native Chat input box at the bottom of the dialogue window
    if user_query := st.chat_input("Describe current patient presentation or triage scenario..."):
        
        # Immediately display user message in the UI
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Prepare package contract payload for FastAPI
        payload = {
            "session_id": st.session_state.session_id,
            "message": user_query,
            "medication": medication_profile if medication_profile else "None"
        }
        
        # Fire request through the network bridge
        with st.chat_message("assistant"):
            with st.spinner("🧠 Agent routing state nodes..."):
                try:
                    response = requests.post(BACKEND_URL, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        guidance = data.get("clinical_guidance", "Empty data structure.")
                        emotions = data.get("detected_emotions", [])
                        context_blocks = data.get("retrieved_database_context", [])
                        
                        # Display bot response
                        st.markdown(guidance)
                        st.session_state.chat_history.append({"role": "assistant", "content": guidance})
                        
                        # Format telemetry tracking data
                        telemetry_string = f"🧠 NLP EMOTION ENGINE STATE:\n{emotions}\n\n"
                        telemetry_string += "🔧 ACTIVE RETRIEVED GRAPH CONTEXT STORES:\n"
                        for i, chunk in enumerate(context_blocks):
                            telemetry_string += f"📍 Source [{i+1}]: {chunk}\n"
                        st.session_state.telemetry_data = telemetry_string
                        
                    else:
                        error_msg = f"❌ Backend Error Status: {response.status_code}"
                        st.error(error_msg)
                        st.session_state.telemetry_data = error_msg
                except Exception as e:
                    connection_error = f"❌ Fatal Loop Connection Error: {str(e)}"
                    st.error(connection_error)
                    st.session_state.telemetry_data = connection_error
        
        # Force refresh to sync state layout changes
        st.rerun()

with col_telemetry:
    st.markdown("### 📊 Live Telemetry Panel")
    # Clean text container displaying our asynchronous LangGraph tracking logs
    st.text_area(
        "Asynchronous Runtime Graph Variables",
        value=st.session_state.telemetry_data,
        height=400,
        disabled=True
    )
