"""Streamlit frontend for the Clinical Decision Support System."""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Clinical Decision Support", layout="centered")

st.title("🏥 Clinical Decision Support System")
st.markdown("Upload a chest X-ray and enter patient clinical notes for automated analysis and triage.")

# Sidebar for configuration
st.sidebar.header("Configuration")
api_key_input = st.sidebar.text_input("API Key (if required)", type="password")
headers = {"x-api-key": api_key_input} if api_key_input else {}

# Main form layout
uploaded_file = st.file_uploader("Upload Chest X-Ray (JPEG or PNG)", type=["jpg", "jpeg", "png"])
clinical_note = st.text_area("Patient Clinical Notes", placeholder="Enter notes here (e.g., patient complains of persistent cough and fever...)...", max_chars=2000)

if st.button("Run Full Analysis", type="primary"):
    if uploaded_file is None:
        st.warning("Please upload a chest X-ray image first.")
    elif not clinical_note.strip():
        st.warning("Please enter a clinical note.")
    else:
        with st.spinner("Analyzing image and processing notes..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                data = {"note": clinical_note}
                
                response = requests.post(f"{API_URL}/analyze", files=files, data=data, headers=headers)
                
                if response.status_code == 200:
                    res = response.json()
                    st.success("Analysis complete!")
                    
                    # Display Results in columns
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("🖼️ Image Prediction")
                        pred = res["prediction"]
                        st.metric(label="Primary Label", value=pred["label"])
                        st.write("Probabilities:")
                        st.json(pred["probabilities"])
                        st.text(f"Latency: {pred['latency_ms']} ms")
                    
                    with col2:
                        st.subheader("📋 Triage Decision")
                        decision_color = "🔴" if res["decision"] == "REFER" else "🟢"
                        st.markdown(f"### Decision: {decision_color} **{res['decision']}**")
                        st.write("**Reasons:**")
                        for reason in res["reasons"]:
                            st.write(f"- {reason}")
                    
                    st.divider()
                    st.subheader("🔒 Masked Note & Extracted Fields")
                    st.text_area("Sanitized Note (PII Masked)", res["note_masked"], disabled=True)
                    
                    if res["urgent_terms"]:
                        st.warning(f"⚠️ Urgent terms detected: {', '.join(res['urgent_terms'])}")
                        
                    st.info(res["disclaimer"])
                    
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI server. Make sure uvicorn is running!")