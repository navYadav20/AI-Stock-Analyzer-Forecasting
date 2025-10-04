import os
import textwrap
import streamlit as st
import google.generativeai as genai

# Configuration
API_CONFIGURED = False
GEMINI_MODEL = None

# Try to read key from Streamlit secrets first, then environment
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")

if GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash")
        API_CONFIGURED = True
    except Exception as e:
        st.error(f"Could not configure Gemini API: {e}")
        API_CONFIGURED = False
else:
    API_CONFIGURED = False


                # Helper utilities

  #Return a truncated text for preview
def shorten_text(text: str, max_chars: int = 700) -> str: # set the max char size to 700 in tab 1

    if not text or len(text) <= max_chars:
        return text
    return textwrap.shorten(text, width=max_chars, placeholder=" ...")


def safe_text_from_gemini(response) -> str:
    # extracts text from a  API response
    if not response:
        return ""
    try:
        return response.text.strip() # split & return
    except AttributeError:
        # Fallback for unexpected response structures
        try:
            return " ".join(part.text for part in response.candidates[0].content.parts).strip()
        except (IndexError, AttributeError):
            return "Could not parse Gemini response."


# generates answer using the configured model
def get_gemini_analysis(prompt: str, concise: bool = True) -> str:

    if not API_CONFIGURED or not GEMINI_MODEL:
        return "API not configured. Set GEMINI_API_KEY in Streamlit secrets."

    if concise:
        prompt = (
            "Be concise. Limit to ~200-300 words. Use a 2-line executive summary followed by short bullet points.\n\n"
            + prompt
        )
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        return safe_text_from_gemini(response)
    except Exception as e:
        return f"Error calling API: {e}"
