import streamlit as st
import httpx
import re
import json

st.set_page_config(page_title="Pharmacist CDSS Portal", layout="wide", page_icon="🏥")

st.title("🏥 Clinical Decision Support System (CDSS) Dashboard")
st.caption("Stateful Multi-Turn RAG Pipeline for Medication Safety and Clinical Risk Assessment")

# Initialize session state tracking components
if "session_id" not in st.session_state:
    st.session_state.session_id = "clinical-session-101"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Configuration Layout
with st.sidebar:
    st.header("⚙️ Session Configuration")
    st.session_state.session_id = st.text_input("Active Session Identifier", value=st.session_state.session_id)
    
    medication_input = st.text_input("Current Patient Medication (Optional)", value="None", help="Pass a specific drug name to explicitly pull matched clinical guidelines.")
    
    st.markdown("---")
    st.markdown("### 🔍 Live Metadata Tracking")
    
    # Reset history button
    if st.button("🔄 Clear Session History", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Session state tracking cleared.")

# Main Application Layout: Two-column layout split (Chat vs Live Analysis Metrics)
col_chat, col_analysis = st.columns([3, 2])

with col_chat:
    st.subheader("💬 Chat Interface")
    
    # Display historical dialog turns cleanly
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.write(turn["text"])
            
    # Input box for user messaging
    if user_message := st.chat_input("Enter clinical observations or patient symptoms..."):
        # Instantly append user's text turn to local screen history
        st.session_state.chat_history.append({"role": "user", "text": user_message})
        with st.chat_message("user"):
            st.write(user_message)
            
        # Post the transmission request to our local running backend API
        with st.spinner("Processing clinical guidelines and running emotional detection model..."):
            try:
                response = httpx.post(
                    "http://127.0.0.1:8000/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": user_message,
                        "medication": medication_input
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    raw_guidance = data.get("clinical_guidance", "")
                    
                    # Strip out the technical [CLINICAL_SUMMARY] block before displaying to keep chat clean
                    clean_guidance = re.sub(r"\[CLINICAL_SUMMARY\].*?\[/CLINICAL_SUMMARY\]", "", raw_guidance, flags=re.DOTALL).strip()
                    
                    # Store response tokens and data packages inside session properties for the metrics sidebar
                    st.session_state.latest_analysis = data
                    st.session_state.chat_history.append({"role": "assistant", "text": clean_guidance})
                    
                    with st.chat_message("assistant"):
                        st.write(clean_guidance)
                        
                    # Rerun to cleanly update the analysis summary charts simultaneously
                    st.rerun()
                else:
                    st.error(f"Backend Server Error: Status Code {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to backend engine: {str(e)}")

# Right side column: Real-Time Parsed Safety Insights Metrics
with col_analysis:
    st.subheader("📊 Real-Time Automated Analysis")
    
    if "latest_analysis" in st.session_state:
        analysis_data = st.session_state.latest_analysis
        raw_guidance = analysis_data.get("clinical_guidance", "")
        
        # Extract and parse structured variables out of the [CLINICAL_SUMMARY] block
        summary_match = re.search(r"\[CLINICAL_SUMMARY\](.*?)\[/CLINICAL_SUMMARY\]", raw_guidance, flags=re.DOTALL)
        
        if summary_match:
            summary_content = summary_match.group(1).strip()
            
            # Map values out manually using string searching expressions safely
            severity = "unknown"
            sev_match = re.search(r"severity:\s*(\w+)", summary_content)
            if sev_match:
                severity = sev_match.group(1).lower()
                
            # Render a color-coded alert notification block depending on parsed security level metrics
            if severity in ["severe", "moderate"]:
                st.error(f"🚨 **CRITICAL RISK LEVEL DETECTED: {severity.upper()}**")
            elif severity in ["mild", "minimal"]:
                st.warning(f"⚠️ **MODERATE RISK LEVEL DETECTED: {severity.upper()}**")
            else:
                st.info(f"ℹ️ **RISK LEVEL DETECTED: {severity.upper()}**")
                
            # Display detailed extracted key-value properties inside expandable containers
            with st.expander("📝 Extracted Summary Details", expanded=True):
                st.text(summary_content)
        
        # Display the output from our local multi-label classification layer
        with st.expander("🧠 Fine-Tuned DistilBERT Emotions", expanded=False):
            emotions = analysis_data.get("detected_emotions", [])
            if emotions:
                st.write(", ".join(emotions))
            else:
                st.write("*No distinct emotional variants flagged below the inference boundary threshold.*")
                
        # Display raw database string paragraphs matches pulled out by vector spacing distance criteria
        with st.expander("📚 Matched RAG Clinical Sources", expanded=False):
            contexts = analysis_data.get("retrieved_database_context", [])
            for idx, context in enumerate(contexts, 1):
                st.markdown(f"**Source Document [{idx}]:**")
                st.code(context, language="markdown")
    else:
        st.info("Awaiting user chat prompt input transmission to initialize streaming pipeline evaluations.")
