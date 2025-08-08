# src/frontend/app.py

import streamlit as st
import requests

# Point everything to the gateway
GATEWAY_URL = "http://localhost:8002"  # Adjust if you run on Docker network

st.set_page_config(page_title="ML Inference Demo", layout="centered")

def login_form():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        with st.spinner("Authenticating..."):
            try:
                resp = requests.post(f"{GATEWAY_URL}/login", json={
                    "username": username,
                    "password": password
                }, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["jwt_token"] = data.get("access_token")
                    st.session_state["user_role"] = data.get("role")
                    st.success("Login successful")
                    st.rerun()
                else:
                    st.error(f"Login failed: {resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"Login failed: {e}")

def predict_form():
    st.title("Price Prediction")
    sku = st.text_input("SKU")
    if st.button("Predict"):
        token = st.session_state.get("jwt_token")
        headers = {"Authorization": f"Bearer {token}"}
        with st.spinner("Predicting..."):
            try:
                resp = requests.post(f"{GATEWAY_URL}/predict", json={
                    "sku": sku
                }, headers=headers, timeout=5)
                if resp.status_code == 200:
                    result = resp.json()
                    st.success(
                        f"SKU: {result['sku']}\n\n"
                        f"Predicted Price: {result['predicted_price']} €\n\n"
                        f"Timestamp: {result['timestamp']}"
                    )
                else:
                    st.error(f"Prediction failed: {resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Admin-only reload-model
    if st.session_state.get("user_role") == "admin":
        if st.button("Reload Model"):
            token = st.session_state.get("jwt_token")
            headers = {"Authorization": f"Bearer {token}"}
            try:
                resp = requests.post(f"{GATEWAY_URL}/reload-model", headers=headers, timeout=5)
                if resp.status_code == 200:
                    st.success("Model reloaded successfully.")
                else:
                    st.error(f"Reload failed: {resp.json().get('detail', resp.text)}")
            except Exception as e:
                st.error(f"Reload failed: {e}")

    if st.button("Logout"):
        st.session_state.pop("jwt_token", None)
        st.session_state.pop("user_role", None)
        st.rerun()

def main():
    if "jwt_token" not in st.session_state:
        login_form()
    else:
        predict_form()

if __name__ == "__main__":
    main()
