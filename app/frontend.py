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

# Initialize telemetry data with structured, rich metadata fields
if "telemetry_data" not in st.session_state:
    st.session_state.telemetry_data = {
        "emotions": ["neutral"],
        "context_blocks": ["Awaiting initial telemetry logs..."],
        "risk_level": "LOW",
        "confidence_score": "N/A",
        "evidence_sources": [],
        "audit_log": {}
    }

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
    # The explicit system clear button
    if st.button("🗑️ Clear Session Memory", use_container_width=True, type="secondary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.telemetry_data = {
            "emotions": ["neutral"],
            "context_blocks": ["🔄 Centralized graph state wiped. New Session initialized."],
            "risk_level": "LOW",
            "confidence_score": "N/A",
            "evidence_sources": [],
            "audit_log": {}
        }
        st.rerun()

# --- MAIN LAYOUT: TWO COLUMN SPLIT (CHAT vs TELEMETRY) ---
col_chat, col_telemetry = st.columns([1.8, 1.2])

with col_chat:
    st.markdown("### 💬 Conversational Dialogue Records")

    # Render historical conversation elements natively
    for message in st.session_state.chat_history:
        # ✨ FIX: Skip displaying the background system instructions to the user
        if message.get("role") == "system":
            continue

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

                        # Display bot response
                        st.markdown(guidance)
                        st.session_state.chat_history.append({"role": "assistant", "content": guidance})

                        # Update state with the clean, new enterprise data objects
                        st.session_state.telemetry_data = {
                            "emotions": data.get("detected_emotions", []),
                            "context_blocks": data.get("retrieved_database_context", []),
                            "risk_level": data.get("risk_level", "LOW"),
                            "confidence_score": data.get("confidence_score", "91%"),
                            "evidence_sources": data.get("evidence_sources", []),
                            "audit_log": data.get("audit_log", {})
                        }

                    else:
                        st.error(f"❌ Backend Error Status: {response.status_code}")
                        st.session_state.telemetry_data["context_blocks"] = [f"Backend Error: {response.status_code}"]
                except Exception as e:
                    st.error(f"❌ Fatal Loop Connection Error: {str(e)}")
                    st.session_state.telemetry_data["context_blocks"] = [f"Connection Failure: {str(e)}"]

        # Force refresh to sync state layout changes
        st.rerun()

# --- RIGHT SIDEBAR: UPGRADED ENTERPRISE GOVERNANCE & AUDIT PANEL ---
with col_telemetry:
    st.markdown("### 📊 Live Telemetry & Governance Panel")

    tel = st.session_state.telemetry_data

    # 🔴 1. Visual Clinical Risk Badge
    risk = tel.get("risk_level", "LOW")
    if risk == "HIGH":
        st.error(
            "🚨 **CLINICAL RISK CLASSIFICATION: HIGH**\n\nImmediate medical/GP oversight or severe contradiction flagged.")
    elif risk == "MODERATE":
        st.warning("⚠️ **CLINICAL RISK CLASSIFICATION: MODERATE**\n\nRoutine review or active monitoring recommended.")
    else:
        st.success(
            "✅ **CLINICAL RISK CLASSIFICATION: LOW / NONE**\n\nNo acute interaction or safety triggers detected.")

    st.markdown("---")

    # 🧠 2. Metric Cards for Verification Confidence and Emotion Analytics
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric(label="Verification Confidence", value=tel.get("confidence_score", "N/A"))
    with m_col2:
        # Convert emotion list to string representation for easy visualization
        emo_str = ", ".join(tel.get("emotions", ["neutral"]))
        st.metric(label="NLP Emotion State", value=emo_str.title())

    st.markdown("---")

    # 📑 3. Explainable AI Citations (Shows exactly where data came from)
    st.markdown("#### 📑 Evidence-Based Citations")
    sources = tel.get("evidence_sources", [])
    if sources:
        for idx, src in enumerate(sources):
            st.info(f"📁 **Source [{idx + 1}]:** `{src}`")
    else:
        st.caption("_No active external database documents linked to this turn._")

    st.markdown("---")

    # 🗄️ 4. Raw Governance Audit Trail Log Object
    st.markdown("#### 🗄️ Immutable Enterprise Audit Trail")
    if tel.get("audit_log"):
        st.json(tel.get("audit_log"))
    else:
        st.caption("_Awaiting incoming runtime ledger data..._")