import streamlit as st
import requests
import uuid
import re

# Points to your healthy Docker Container endpoint
BACKEND_URL = "http://127.0.0.1:8000/chat"

# Secure API Key configuration matching your backend APIKeyHeader validation fallback
API_KEY = "prod-secret-fallback-key"

st.set_page_config(
    page_title="Pharmacist CDSS",
    layout="wide",
    page_icon="👨‍⚕️"
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "telemetry_data" not in st.session_state:
    st.session_state.telemetry_data = {
        "emotions": ["neutral"],
        "context_blocks": ["Awaiting first query..."],
        "risk_level": "LOW",
        "confidence_score": "N/A",
        "evidence_sources": [],
        "audit_log": {},
        "clinical_summary": {}
    }


def clean_response(text: str) -> str:
    """Remove CLINICAL_SUMMARY block from chat display."""
    cleaned = re.sub(
        r'\[CLINICAL_SUMMARY\].*?\[/CLINICAL_SUMMARY\]',
        '',
        text,
        flags=re.DOTALL
    ).strip()
    return cleaned


def extract_summary(text: str) -> dict:
    """Extract CLINICAL_SUMMARY data for telemetry panel."""
    summary = {}
    block = re.search(
        r'\[CLINICAL_SUMMARY\](.*?)\[/CLINICAL_SUMMARY\]',
        text, re.DOTALL
    )
    if not block:
        return summary

    content = block.group(1)

    for field in ["severity", "recommended_action", "follow_up_question"]:
        match = re.search(rf'{field}:\s*"?([^"\n]+)"?', content)
        if match:
            summary[field] = match.group(1).strip()

    symptoms_match = re.search(r'key_symptoms:\s*(\[.*?\])', content)
    if symptoms_match:
        summary["key_symptoms"] = symptoms_match.group(1)

    return summary


# --- HEADER ---
st.title("👨‍⚕️ Pharmacist Clinical Decision Support System")
st.caption("AI-powered clinical safety layer for pharmacy consultations")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🔑 Session")
    st.text_input("Session ID", value=st.session_state.session_id, disabled=True)

    st.markdown("---")
    st.markdown("### 💊 Patient Medication Profile")
    medication_profile = st.text_input(
        "Active medications",
        value="None",
        help="e.g. sertraline 50mg, warfarin 5mg"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Session", use_container_width=True, type="secondary"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.telemetry_data = {
            "emotions": ["neutral"],
            "context_blocks": ["Session cleared."],
            "risk_level": "LOW",
            "confidence_score": "N/A",
            "evidence_sources": [],
            "audit_log": {},
            "clinical_summary": {}
        }
        st.rerun()

# --- MAIN LAYOUT ---
col_chat, col_telemetry = st.columns([1.8, 1.2])

with col_chat:
    st.markdown("### 💬 Consultation")

    for message in st.session_state.chat_history:
        if message.get("role") == "system":
            continue
        with st.chat_message(message["role"]):
            display_text = clean_response(message["content"])
            st.markdown(display_text)

    if user_query := st.chat_input("Describe patient presentation or ask a pharmacy question..."):

        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # CRITICAL FIX: Match the clean case-sensitive header expected by your middleware
        headers = {
            "X-API-KEY": API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "session_id": st.session_state.session_id,
            "message": user_query,
            "medication": medication_profile if medication_profile else "None"
        }

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data streams..."):
                try:
                    response = requests.post(BACKEND_URL, json=payload, headers=headers, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        raw_guidance = data.get("clinical_guidance", "No response.")

                        display_guidance = clean_response(raw_guidance)
                        st.markdown(display_guidance)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": raw_guidance
                        })

                        summary = extract_summary(raw_guidance)

                        st.session_state.telemetry_data = {
                            "emotions": data.get("detected_emotions", ["neutral"]),
                            "context_blocks": data.get("retrieved_database_context", []),
                            "risk_level": data.get("risk_level", "LOW"),
                            "confidence_score": data.get("confidence_score", "N/A"),
                            "evidence_sources": data.get("evidence_sources", []),
                            "audit_log": data.get("audit_log", {}),
                            "clinical_summary": summary
                        }

                        # Force refresh the UI to render telemetry state immediately
                        st.rerun()

                    else:
                        st.error(f"Backend API Error: Received Status Code {response.status_code}")
                        st.info("Check your Docker terminal logs to see why the API header validation rejected this request.")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach the Docker container backend! Is it still running on port 8000?")
                except Exception as e:
                    st.error(f"Processing error: {str(e)}")

# --- TELEMETRY PANEL ---
with col_telemetry:
    st.markdown("### 📊 Live Telemetry")

    tel = st.session_state.telemetry_data
    risk = tel.get("risk_level", "LOW")

    if risk == "HIGH":
        st.error("🚨 **CLINICAL RISK: HIGH**\n\nImmediate medical countermeasure validation required.")
    elif risk == "MODERATE":
        st.warning("⚠️ **CLINICAL RISK: MODERATE**\n\nActive monitoring and validation recommended.")
    else:
        st.success("✅ **CLINICAL RISK: LOW**\n\nNo acute safety triggers detected.")

    st.markdown("---")

    m1, m2 = st.columns(2)

    with m1:
        st.caption("Confidence")
        st.markdown(
            f"<span style='font-size:16px;font-weight:600'>{tel.get('confidence_score', 'N/A')}</span>",
            unsafe_allow_html=True
        )

    with m2:
        emo = ", ".join(tel.get("emotions", ["neutral"]))
        st.caption("Emotion State")
        st.markdown(
            f"<span style='font-size:16px;font-weight:600'>{emo.title()}</span>",
            unsafe_allow_html=True
        )

    summary = tel.get("clinical_summary", {})
    if summary:
        st.markdown("---")
        st.markdown("#### 🩺 Clinical Summary")

        sev = summary.get("severity", "")
        if sev.lower() == "severe":
            st.error(f"**Severity:** {sev.upper()}")
        elif sev.lower() == "moderate":
            st.warning(f"**Severity:** {sev.upper()}")
        elif sev.lower() in ["mild", "minimal"]:
            st.info(f"**Severity:** {sev.upper()}")

        if summary.get("key_symptoms"):
            st.markdown(f"**Key Symptoms:** {summary['key_symptoms']}")
        if summary.get("recommended_action"):
            st.markdown(f"**Action:** {summary['recommended_action']}")
        if summary.get("follow_up_question"):
            st.markdown(f"**Follow-up:** _{summary['follow_up_question']}_")

    st.markdown("---")
    st.markdown("#### 📑 Evidence Sources")
    sources = tel.get("evidence_sources", [])
    if sources:
        for i, src in enumerate(sources):
            st.info(f"📁 **Source {i + 1}:** `{src}`")
    else:
        st.caption("No sources retrieved.")

    st.markdown("---")
    st.markdown("#### 🗄️ Audit Trail")
    if tel.get("audit_log"):
        st.json(tel.get("audit_log"))
    else:
        st.caption("Awaiting data...")