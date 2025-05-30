import streamlit as st
from PIL import Image
import numpy as np
import os
import base64
import io # For handling image bytes

# --- Gemini API Integration ---
import google.generativeai as genai

# Configure Gemini API Key securely
# For Streamlit Community Cloud: Use st.secrets
# For local testing: Use environment variable or hardcode temporarily for hackathon (NOT PRODUCTION!)
try:
    # Attempt to get API key from environment variable (for local testing)
    # or from Streamlit secrets (for Streamlit Community Cloud deployment)
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    # Display a warning if the API key isn't found, but allow the app to run
    # (solution generation will fail without the key)
    st.warning("Gemini API Key not found. Please set GEMINI_API_KEY environment variable (local) or in Streamlit secrets (cloud). Solution generation may fail.")


def get_solution_from_gemini(disease_name, plant_type="plant"):
    """
    Calls the Gemini API to get a concise solution and prevention guide for a given plant disease.

    Args:
        disease_name (str): The name of the plant disease.
        plant_type (str): The type of plant (e.g., "Apple", "Tomato"). Defaults to "plant".

    Returns:
        str: A Markdown-formatted string with the solution, or an error message.
    """
    if not GEMINI_API_KEY:
        return "Error: Gemini API key not configured. Cannot fetch solution."
    try:
        # Initialize the GenerativeModel with the specified model name
        model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest') # Using 'flash' for faster responses

        # Construct the prompt for the Gemini model
        # Re-typing triple quotes to resolve potential 'unterminated string literal' error
        prompt = f"""
Provide a concise and actionable guide for treating and preventing the plant disease: '{disease_name}' in {plant_type}s.
Include the following sections if possible:
1.  **Description**: Briefly describe the disease.
2.  **Symptoms**: Key visual symptoms.
3.  **Treatment**: Organic and chemical treatment options.
4.  **Prevention**: Steps to prevent future occurrences.
Keep the language clear and practical for a general audience. If the disease name seems very generic or unclear, provide general plant care advice.
"""
        # Generate content using the prompt
        response = model.generate_content(prompt)
        return response.text # Return the generated text
    except Exception as e:
        # Catch and display any errors during the API call
        st.error(f"Error calling Gemini API: {e}")
        return f"Sorry, I couldn't fetch a solution at this time. Error: {e}"

# --- Removed Vertex AI Initialization ---
# The Vertex AI client initialization and endpoint connection are removed
# as the prediction part is now being manually bypassed for demonstration.
# PROJECT_ID = "genaihackathon1"
# REGION = "us-central1"
# ENDPOINT_ID = "YOUR_VERTEX_AI_ENDPOINT_ID"
# endpoint = None # No longer needed as we're not attempting connection

# --- Image Preprocessing (still needed for image display) ---
def load_and_preprocess_image_for_display(image_pil):
    """
    Prepares a PIL Image for display in Streamlit.
    (No base64 encoding needed as we are not sending to Vertex AI for prediction)

    Args:
        image_pil (PIL.Image.Image): The input image in PIL format.

    Returns:
        PIL.Image.Image: The processed image for display.
    """
    return image_pil.convert('RGB') # Ensure image is in RGB format for consistent display


# --- Streamlit UI Layout and Interaction ---
# Set basic page configuration for a wide layout and title
st.set_page_config(layout="wide", page_title="Plant Disease Classifier")
st.title("🌿 Plant Disease Classifier & Solution Advisor")
st.markdown("Upload an image of a plant leaf to classify its disease and get treatment advice.")

# Create two columns for layout: one for image upload/classification, one for solution
col1, col2 = st.columns(2)

with col1:
    # File uploader widget for image input
    uploaded_file = st.file_uploader("Choose a plant image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file) # Open directly
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Button to trigger the classification and solution generation
        if st.button('🔍 Classify Disease & Get Solution', type="primary"):
            with st.spinner('Analyzing image and fetching advice...'):
                # --- Bypassing Vertex AI Prediction ---
                # Manually set the predicted disease to "Wheat Rust"
                predicted_class_name = "Wheat___Rust" # Hardcoded for demonstration
                confidence = 0.95 # Dummy confidence for demonstration

                st.warning("Live Vertex AI classification is currently bypassed for demonstration. Displaying hardcoded 'Wheat Rust' classification.")

                # Display classification results
                st.success(f"**Predicted Disease:** {predicted_class_name.replace('___', ' - ').replace('_', ' ')}")
                st.info(f"**Confidence:** {confidence*100:.2f}%")

                # --- Get solution from Gemini based on the hardcoded disease ---
                if predicted_class_name != "Unknown Disease":
                    # Extract plant type and disease name for a more specific Gemini prompt
                    # This logic will now always process "Wheat___Rust"
                    plant_name_parts = predicted_class_name.split("___")
                    plant_type = plant_name_parts[0].replace('_', ' ') if len(plant_name_parts) > 0 else "plant"
                    disease_actual_name = plant_name_parts[1].replace('_', ' ') if len(plant_name_parts) > 1 else predicted_class_name.replace('_', ' ')

                    with col2:
                        st.subheader("💡 Recommended Solution & Prevention")
                        # Call the Gemini function to get the solution
                        solution_text = get_solution_from_gemini(disease_actual_name, plant_type)
                        st.markdown(solution_text) # Display the solution in Markdown format
                else:
                     with col2:
                        st.warning("Could not determine a specific disease to fetch a solution.")

    else:
        # Message displayed when no image is uploaded yet
        with col2:
            st.info("Upload an image to get started.")

# --- Sidebar Information ---
st.sidebar.header("About This Project")
st.sidebar.info(
    "This app uses a Deep Learning model deployed on **Vertex AI** to classify plant diseases "
    "and leverages the **Gemini API** to provide treatment and prevention advice. "
    "Built for the GenAI Hackathon."
)

st.sidebar.header("Why This Project?")
st.sidebar.markdown(
    """
    A growing number of individuals are embracing personal farming for health, cost savings, and personal satisfaction. 
    Simultaneously, a significant portion of these individuals, including many software developers, are venturing into 
    commercial farming, either as a primary or secondary source of income.

    This application aims to assist them with crucial plant disease classification, providing optimized guidance 
    to protect their crops and livelihoods.
    """
)

st.sidebar.header("Configuration (for developers)")
st.sidebar.markdown(
    """
- **GCP Project ID:** `genaihackathon1` (Not directly used in this bypassed version)
- **Vertex AI Region:** `us-central1` (Not directly used in this bypassed version)
- **Vertex AI Endpoint ID:** `YOUR_VERTEX_AI_ENDPOINT_ID` (Not used; classification is hardcoded)
- **Gemini Model:** `gemini-1.5-flash-latest`
"""
)
st.sidebar.markdown(
    """
    **Note:** The Vertex AI Endpoint for live classification has been bypassed in this version for demonstration purposes 
    due to cost considerations. The predicted disease is **hardcoded to "Wheat Rust"** to showcase the Gemini API's 
    ability to provide remedies. To enable live plant disease classification, you would need to deploy your own 
    Vertex AI AutoML model and integrate it.
    """
)

st.sidebar.header("Future Plans")
st.sidebar.markdown(
    """
    Future enhancements include enabling **Google Translate for local language support**, 
    making this app accessible and beneficial for elderly farmers who may not be proficient in English.
    """
)
