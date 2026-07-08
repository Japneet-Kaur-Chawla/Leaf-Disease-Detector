# app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image
import streamlit.components.v1 as components
from feature_extraction import extract_features_from_image

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Plant Disease Detector 🌿",
    page_icon="🌿",
    layout="wide"
)

# -----------------------------
# Session state
# -----------------------------
_session_defaults = {
    "prediction": None,
    "show_panel": None,
    "history": [],
    "theme": "dark",
    "print_trigger": False,
}
for _key, _val in _session_defaults.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# -----------------------------
# Color Palettes
# -----------------------------
_PALETTES = {
    "dark": {
        "bg": "#050b1f",
        "sidebar_bg": "#0a1230",
        "text": "#ffffff",
        "panel_bg": "rgba(0, 0, 0, 0.35)",
        "history_bg": "rgba(255, 255, 255, 0.03)",
        "history_border": "#444444",
        "popover_bg": "#0a1230",
    },
    "light": {
        "bg": "#f2f4fb",
        "sidebar_bg": "#e6eaf7",
        "text": "#0b1330",
        "panel_bg": "rgba(255, 255, 255, 0.65)",
        "history_bg": "rgba(0, 0, 0, 0.04)",
        "history_border": "#c7cbdb",
        "popover_bg": "#e6eaf7",
    },
}
_palette = _PALETTES[st.session_state.theme]

# -----------------------------
# Global / Structural CSS
# -----------------------------
st.markdown("""
<style>
/* Background */
.stApp { background-color: #050b1f; }
[data-testid="stHeader"], [data-testid="stToolbar"], 
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background-color: #050b1f; }

/* Hide default streamlit deploy/menu buttons */
[data-testid="stDeployButton"], [data-testid="stMainMenu"], 
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"] { display: none !important; }

/* Subtext */
p, label, span { color: #ffffff !important; }

/* ---------------- Sidebar ---------------- */
[data-testid="stSidebar"] {
    background-color: #0a1230;
    border-right: 2px solid #00e5ff;
    box-shadow: 4px 0 15px rgba(0, 229, 255, 0.3);
}

/* Updated Sidebar Controls Heading */
[data-testid="stSidebar"] h2 {
    color: #FFFF00 !important;
    text-shadow: 0 0 12px #FFFF00;
    text-align: center;
    font-size: 2.2rem !important; 
    margin-bottom: 10px;
}

/* Updated File Uploader - Removed Outer Box */
[data-testid="stFileUploader"] {
    border: none !important;
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin-bottom: 6px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Hide the "Drag and drop file here" and "Limit 200MB" text */
[data-testid="stFileUploaderDropzoneInstructions"] { 
    display: none !important; 
}

/* File Uploader Button - Matched to other sidebar buttons */
[data-testid="stFileUploader"] button {
    background-color: transparent !important;
    color: white !important;
    border: 2px solid #00e5ff !important;
    padding: 0.6em 1em !important;
    font-size: 15px !important;
    border-radius: 10px !important;
    font-weight: normal !important;
    box-shadow: 0 0 8px #00e5ff !important;
    transition: 0.3s !important;
    width: 100% !important; /* Spans the full width */
}

[data-testid="stFileUploader"] button:hover {
    background-color: #00e5ff !important;
    color: #050b1f !important;
    box-shadow: 0 0 18px #00e5ff !important;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background-color: transparent;
    color: white;
    border: 2px solid #00e5ff;
    padding: 0.6em 1em;
    font-size: 15px;
    border-radius: 10px;
    box-shadow: 0 0 8px #00e5ff;
    transition: 0.3s;
    width: 100%;
    margin-bottom: 6px;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #00e5ff;
    color: #050b1f;
    box-shadow: 0 0 18px #00e5ff;
}
.clear-btn > button { border-color: #ff3860 !important; box-shadow: 0 0 8px #ff3860 !important; }
.clear-btn > button:hover { background-color: #ff3860 !important; color: #050b1f !important; box-shadow: 0 0 18px #ff3860 !important; }

/* ---------------- 3-Dot Trigger Button ---------------- */
div[data-testid="stPopover"] > div > button {
    background-color: #000000 !important;
    border: 2px solid #ff0000 !important;
    width: 45px !important;
    height: 45px !important;
    min-width: 45px !important;
    padding: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 8px !important;
    box-shadow: 0 0 15px #ff0000 !important;
    transition: all 0.3s ease !important;
    float: right !important;
}

div[data-testid="stPopover"] > div > button * {
    color: #ffffff !important;
    font-size: 26px !important;
    font-weight: bold !important;
    margin: 0 !important;
    padding: 0 !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stPopover"] > div > button:hover {
    background-color: #ff0000 !important;
    box-shadow: 0 0 25px #ff0000 !important;
}

div[data-testid="stPopover"] > div > button:hover * {
    color: #000000 !important;
}

/* ---------------- Popover Dropdown Container (Green Outer Box) ---------------- */
div[data-testid="stPopoverBody"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

div[data-baseweb="popover"] > div {
    border: 2px solid #39ff14 !important; 
    border-radius: 16px !important;
    box-shadow: 0 0 15px rgba(57, 255, 20, 0.5) !important; 
    padding: 15px !important; 
}

/* ---------------- Popover Inner Buttons (Neon Green Floating Boxes) ---------------- */
.st-key-print_btn button,
.st-key-theme_btn button {
    background-color: #0a1230 !important; 
    color: white !important;
    border: 2px solid #39ff14 !important; 
    padding: 0.8em 1em !important;
    font-size: 15px !important;
    border-radius: 12px !important;
    box-shadow: 0 0 12px rgba(57, 255, 20, 0.4) !important;
    transition: 0.3s !important;
    width: 100% !important;
    margin-bottom: 15px !important;
}

.st-key-theme_btn button {
    margin-bottom: 0px !important; 
}

.st-key-print_btn button:hover,
.st-key-theme_btn button:hover {
    background-color: #39ff14 !important;
    color: #050b1f !important;
    box-shadow: 0 0 20px #39ff14 !important;
}

.st-key-print_btn button:hover *,
.st-key-theme_btn button:hover * {
    color: #050b1f !important;
}

/* ---------------- Main panels ---------------- */
h1 {
    color: #FFFF00 !important;
    text-align: center;
    text-shadow: 0 0 12px #FFFF00;
    margin-top: -15px; 
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 2px solid #00e5ff !important;
    border-radius: 14px !important;
    background-color: rgba(0, 0, 0, 0.35);
    box-shadow: 0 0 15px #00e5ff;
    min-height: 750px;
}
.panel-box h3 { color: #00e5ff; text-shadow: 0 0 8px #00e5ff; margin-top: 0; text-align: center; }

/* Result Box */
.result-box {
    margin-top: 10px;
    padding: 15px;
    border-radius: 12px;
    border: 2px solid #39ff14;
    background-color: rgba(0, 0, 0, 0.4);
    color: white;
    text-align: center;
    font-size: 20px;
    box-shadow: 0 0 15px #39ff14;
}

/* Info panels */
.info-box {
    margin-top: 14px;
    padding: 16px 18px;
    border-radius: 12px;
    border: 2px solid #a259ff;
    background-color: rgba(0, 0, 0, 0.45);
    color: #ffffff;
    box-shadow: 0 0 12px #a259ff;
    line-height: 1.6;
}
.info-box h4 { color: #a259ff; text-shadow: 0 0 8px #a259ff; margin-top: 0; }
.cure-box { border-color: #39ff14; box-shadow: 0 0 12px #39ff14; }
.cure-box h4 { color: #39ff14; text-shadow: 0 0 8px #39ff14; }

.history-entry {
    padding: 8px 12px;
    margin-bottom: 6px;
    border-radius: 8px;
    border: 1px solid #444;
    background-color: rgba(255,255,255,0.03);
    font-size: 14px;
}
img { border-radius: 12px; border: 2px solid #00e5ff; box-shadow: 0 0 15px #00e5ff; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Theme-aware CSS overrides
# -----------------------------
st.markdown(f"""
<style>
.stApp, [data-testid="stHeader"], [data-testid="stToolbar"], 
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color: {_palette['bg']}; }}

p, label, span {{ color: {_palette['text']} !important; }}
[data-testid="stSidebar"] {{ background-color: {_palette['sidebar_bg']}; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background-color: {_palette['panel_bg']}; }}
.result-box, .info-box {{ color: {_palette['text']}; }}
.history-entry {{ background-color: {_palette['history_bg']}; border-color: {_palette['history_border']}; color: {_palette['text']}; }}

/* Ensure outer box background matches the theme correctly */
div[data-baseweb="popover"] > div {{ 
    background-color: {_palette['popover_bg']} !important; 
}}

/* Sync floating button backgrounds with theme */
.st-key-print_btn button, .st-key-theme_btn button {{
    background-color: {_palette['sidebar_bg']} !important;
    color: {_palette['text']} !important;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Paths & Disease Data Knowledge Base
# -----------------------------
RF_MODEL_PATH = "results/PlantVillage_selected_rf.pkl"
SVM_MODEL_PATH = "results/PlantVillage_selected_svm.pkl"
GA_MASK_PATH = "results/PlantVillage_ga_mask.npz"
LABEL_MAP_PATH = "results/PlantVillage_features_train.npz"

DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "about": "A common bacterial disease caused by Xanthomonas. Spread through infected seeds, plant debris, rain splash, overhead irrigation, tools, and workers. Warm (25–30°C) and humid conditions favor rapid spread.",
        "symptoms": "Appears as small, water-soaked spots that later become dark brown or black. Severe infection causes leaf drop, fruit spots, and reduced yield.",
        "cure": "Use certified disease-free seeds. Remove infected leaves and crop debris. Avoid overhead watering. Rotate crops for 2–3 years. Apply copper-based bactericides (preventive). Grow resistant pepper varieties whenever available."
    },
    "Pepper__bell___healthy": {
        "about": "Healthy pepper plant with vigorous growth and normal flowering and fruit production. No infection present.",
        "symptoms": "Leaves are uniformly green with no spots or discoloration.",
        "cure": "Provide adequate sunlight (6–8 hours/day). Maintain proper watering and apply balanced fertilizers. Regularly inspect for pests and diseases."
    },
    "Potato___Early_blight": {
        "about": "Caused by the fungus Alternaria solani. Spread by wind, rain splash, infected soil, and crop residue. Favored by warm temperatures (24–29°C) with moisture.",
        "symptoms": "Produces brown circular lesions with concentric 'target-like' rings. Usually starts on older leaves.",
        "cure": "Remove infected leaves. Practice crop rotation. Apply fungicides such as chlorothalonil or mancozeb. Maintain proper plant spacing."
    },
    "Potato___healthy": {
        "about": "Healthy potato plant exhibiting normal growth and tuber development. No infection present.",
        "symptoms": "Leaves remain dark green and free from lesions.",
        "cure": "Ensure proper irrigation, balanced fertilization, and weed management. Monitor regularly for diseases."
    },
    "Potato___Late_blight": {
        "about": "Caused by Phytophthora infestans (responsible for the historic Irish Potato Famine). Spreads rapidly through wind-blown spores and thrives in cool (10–20°C), wet weather.",
        "symptoms": "Produces large water-soaked lesions with white fungal growth underneath leaves.",
        "cure": "Remove infected plants immediately. Avoid overhead irrigation. Apply fungicides before severe infection. Use resistant potato varieties and destroy infected crop residues."
    },
    "Tomato___Target_Spot": {
        "about": "Caused by the fungus Corynespora cassiicola. Spread through wind, rain splash, and infected debris. Warm and humid conditions favor disease.",
        "symptoms": "Brown lesions with concentric rings appear on leaves and fruits. Severe infection leads to premature defoliation.",
        "cure": "Remove infected leaves. Improve air circulation. Avoid wet foliage. Apply fungicides if necessary."
    },
    "Tomato___Tomato_mosaic_virus": {
        "about": "Viral disease spread through infected seeds, contaminated tools, hands, and plant-to-plant contact. Virus survives in plant debris.",
        "symptoms": "Causes mottled light and dark green patches. Leaves become distorted and fruits may be malformed.",
        "cure": "No cure once infected. Remove infected plants. Disinfect gardening tools. Wash hands after handling infected plants. Use resistant varieties."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "about": "Viral disease transmitted mainly by whiteflies. Does not spread through soil.",
        "symptoms": "Causes yellow curled leaves and severe stunting. Greatly reduces fruit production.",
        "cure": "No cure after infection. Control whiteflies. Remove infected plants. Use insect-proof nets. Plant resistant varieties."
    },
    "Tomato___Bacterial_spot": {
        "about": "Caused by Xanthomonas bacteria. Spread through infected seeds, rain splash, irrigation water, tools, and workers. Warm, humid conditions increase spread.",
        "symptoms": "Produces dark brown to black spots on leaves and fruits. Severe cases lead to leaf drop.",
        "cure": "Use certified seeds. Remove infected leaves. Rotate crops. Apply copper-based bactericides. Avoid working with wet plants."
    },
    "Tomato___Early_blight": {
        "about": "Caused by Alternaria solani. Wind, rain, infected soil, and crop residue spread spores. Warm temperatures and humidity favor infection.",
        "symptoms": "Circular brown spots with concentric rings. Starts on lower leaves.",
        "cure": "Remove infected leaves. Mulch around plants. Apply fungicides. Rotate crops."
    },
    "Tomato___healthy": {
        "about": "Healthy tomato plant with strong stems and normal fruit growth. No infection present.",
        "symptoms": "Bright green leaves without discoloration or spots.",
        "cure": "Maintain regular watering, balanced fertilizer, proper sunlight, and good air circulation."
    },
    "Tomato___Late_blight": {
        "about": "Caused by Phytophthora infestans. Wind-blown spores spread the disease, encouraged by cool, wet weather.",
        "symptoms": "Produces large irregular dark lesions with white mold under leaves. Can destroy crops within days.",
        "cure": "Remove infected plants. Apply preventive fungicides. Avoid overhead irrigation. Improve ventilation."
    },
    "Tomato___Leaf_Mold": {
        "about": "Caused by the fungus Passalora fulva. Spread through airborne spores and favored by high humidity (above 85%).",
        "symptoms": "Yellow spots appear on upper leaf surfaces. Olive-green mold develops underneath leaves.",
        "cure": "Improve greenhouse ventilation. Reduce humidity. Remove infected leaves. Apply fungicides. Grow resistant varieties."
    },
    "Tomato___Septoria_leaf_spot": {
        "about": "Caused by Septoria lycopersici. Spread by rain splash, irrigation water, infected debris, and wind in warm, humid weather.",
        "symptoms": "Numerous small circular gray spots with dark borders. Begins on lower leaves.",
        "cure": "Remove infected leaves. Rotate crops. Apply fungicides. Mulch around plants. Avoid overhead watering."
    },
    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "about": "Caused by tiny sap-sucking spider mites (Tetranychus urticae). Mites spread by wind, clothing, tools, and movement between plants. Hot, dry conditions favor rapid growth.",
        "symptoms": "Leaves develop yellow speckles, bronzing, curling, and fine webbing. Heavy infestations reduce photosynthesis and fruit yield.",
        "cure": "Spray leaves with water to reduce mite populations. Introduce predatory mites. Apply insecticidal soap or horticultural oil. Use miticides if severe."
    }
}

DEFAULT_INFO = {
    "about": "Detailed information for this class hasn't been added to the knowledge base yet.",
    "symptoms": "Not available yet — please add this class to DISEASE_INFO in app.py.",
    "cure": "Not available yet — please add treatment guidance for this class in app.py.",
}

def get_disease_info(class_name: str) -> dict:
    clean_target = class_name.lower().replace("_", "").replace(" ", "")
    for key, info in DISEASE_INFO.items():
        clean_key = key.lower().replace("_", "").replace(" ", "")
        if clean_key == clean_target:
            return info
    return DEFAULT_INFO

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_assets():
    rf_model = joblib.load(RF_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    mask = np.load(GA_MASK_PATH)["mask"].astype(bool)
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True)["label_map"].item()
    inv_label_map = {v: k for k, v in label_map.items()}
    return rf_model, svm_model, mask, inv_label_map

rf_model, svm_model, ga_mask, label_map = load_assets()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## ⚡️ Controls")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    predict_clicked = st.button("🔮 Predict")
    info_clicked = st.button("📖 Disease Info")
    cure_clicked = st.button("💊 Cure & Treatment")
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    clear_clicked = st.button("🗑️ Clear")
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Handle button logic
# -----------------------------
image = Image.open(uploaded_file) if uploaded_file else None

if clear_clicked:
    st.session_state.prediction = None
    st.session_state.show_panel = None
    uploaded_file = None
    image = None
    st.rerun()

if predict_clicked:
    if image is None:
        st.sidebar.warning("Please upload an image first.")
    else:
        with st.spinner("Analyzing leaf pattern... 🌿"):
            features = extract_features_from_image(image)
            features = np.array(features)
            features = features[ga_mask]
            features = features.reshape(1, -1)

            pred = rf_model.predict(features)[0]
            class_name = label_map.get(pred, "Unknown")

        st.session_state.prediction = class_name
        st.session_state.show_panel = None
        st.session_state.history.insert(0, class_name)

if info_clicked:
    if st.session_state.prediction is None:
        st.sidebar.warning("Run a prediction first.")
    else:
        st.session_state.show_panel = "info"

if cure_clicked:
    if st.session_state.prediction is None:
        st.sidebar.warning("Run a prediction first.")
    else:
        st.session_state.show_panel = "cure"


# -----------------------------
# Browser-side actions (Print)
# -----------------------------
_PRINT_JS = "window.parent.print();"

def _fire_browser_action(js_code: str) -> None:
    snippet = "<script>\ntry {\n" + js_code + "\n} catch (e) {\n    console.error('Browser action failed:', e);\n}\n</script>"
    components.html(snippet, height=0, width=0)

# -----------------------------
# Main area — Layout adjusted for Top-Right Button
# -----------------------------
col1, col2 = st.columns([10, 1])

with col2:
    with st.popover("⋮"):
        print_clicked = st.button("🖨️ Print", key="print_btn", use_container_width=True)
        theme_label = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
        theme_clicked = st.button(theme_label, key="theme_btn", use_container_width=True)

st.title("🌿 Plant Disease Detection System")

# Button Triggers
if print_clicked: 
    st.session_state.print_trigger = True

if theme_clicked:
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

# Execute Javascript based on state
if st.session_state.print_trigger:
    _fire_browser_action(_PRINT_JS)
    st.session_state.print_trigger = False

left_col, right_col = st.columns(2)

# ---- Left box: image upload / display ----
with left_col:
    panel = st.container(border=True)
    with panel:
        st.markdown("### 🖼️ Leaf Image")
        if image is not None:
            st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
        else:
            st.info("No image uploaded yet. Use **Upload** in the left sidebar panel.")

# ---- Right box: prediction output / info / cure / history ----
with right_col:
    panel = st.container(border=True)
    with panel:
        st.markdown("### 📊 Results")
        if st.session_state.prediction is None:
            st.info("No prediction yet. Upload an image and click **Predict**.")
        else:
            class_name = st.session_state.prediction
            st.markdown(f'<div class="result-box">🌿 <b>Prediction:</b> {class_name}</div>', unsafe_allow_html=True)
            info = get_disease_info(class_name)

            if st.session_state.show_panel == "info":
                st.markdown(f'<div class="info-box"><h4>📖 About {class_name}</h4><p><b>Description:</b> {info["about"]}</p><p><b>Symptoms:</b> {info["symptoms"]}</p></div>', unsafe_allow_html=True)
            elif st.session_state.show_panel == "cure":
                st.markdown(f'<div class="info-box cure-box"><h4>💊 Cure & Treatment</h4><p>{info["cure"]}</p></div>', unsafe_allow_html=True)

        if st.session_state.history:
            st.markdown("#### 🕘 History")
            for i, past_class in enumerate(st.session_state.history[:10], start=1):
                st.markdown(f'<div class="history-entry">{i}. {past_class}</div>', unsafe_allow_html=True)