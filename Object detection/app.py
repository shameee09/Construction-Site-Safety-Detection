import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os


st.set_page_config(
    page_title="Construction Safety Detection",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MODEL PATH

MODEL_PATH = "best.pt"

# SIDEBAR

with st.sidebar:

    st.title("🦺 Construction Safety Detection")

    st.caption("Construction Safety Intelligence")

    st.divider()

    st.subheader("About")

    st.write(
        "Construction Safety Detection uses  "
        "to detect construction safety equipment "
        "from uploaded images."
    )

    st.divider()

    st.subheader("Detects")

    st.write("⛑️ Safety Helmet")
    st.write("🦺 Safety Vest")

    st.divider()

    st.divider()

    st.caption(
        "AI Safety Monitoring System"
    )

# MAIN HEADER

st.title("🦺 Construction Safety Detection")

st.subheader(
    "Intelligent based Construction Site Safety Detection"
)

st.write(
    "Detect safety helmets and high-visibility vests "
)

st.divider()

# MODEL CHECK

if not os.path.exists(MODEL_PATH):

    st.error("❌ YOLO model not found.")

    st.write("Expected model location:")

    st.code(MODEL_PATH)

    st.stop()

# UPLOAD SECTION

st.header("📤 Upload Construction Image")

uploaded_file = st.file_uploader(
    "Upload a construction-site image",
    type=["jpg", "jpeg", "png"]
)

# NO IMAGE

if uploaded_file is None:

    st.info(
        "👆 Upload a construction-site image to begin Intelligent inspection."
    )

    st.subheader("🔎 Inspection Workflow")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### 1️⃣")
        st.write("Upload Image")

    with step2:
        st.markdown("### 2️⃣")
        st.write("AI Analysis")

    with step3:
        st.markdown("### 3️⃣")
        st.write("Detect Equipment")

    with step4:
        st.markdown("### 4️⃣")
        st.write("View Results")

# IMAGE UPLOADED

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    st.header("🔍 Safety Inspection")

    col1, col2 = st.columns(2)

    # ORIGINAL IMAGE

    with col1:

        st.subheader("📷 Original Image")

        st.image(
            image,
            use_container_width=True
        )

    # DETECTION BUTTON

    with col2:

        st.subheader("🎯 AI Detection")

        detect_button = st.button(
            "🔍 Run Safety Inspection",
            type="primary",
            use_container_width=True
        )

        if not detect_button:

            st.info(
                "Click the button above to analyze the image."
            )

    # RUN DETECTION

    if detect_button:

        try:

            with st.spinner(
                "Loading YOLO11 model..."
            ):

                model = YOLO(MODEL_PATH)

            with st.spinner(
                "Analyzing construction image..."
            ):

                image_array = np.array(image)

                results = model.predict(
                    source=image_array,
                    conf=0.40,
                    imgsz=640,
                    iou=0.50,
                    device="cpu",
                    verbose=False
                )

                result = results[0]

            annotated_image = result.plot()

            with col2:

                st.subheader("🎯 Detection Result")

                st.image(
                    annotated_image,
                    channels="RBG",
                    use_container_width=True
                )


            helmet_count = 0
            vest_count = 0

            if result.boxes is not None:

                for class_id in result.boxes.cls:

                    class_id = int(class_id)

                    if class_id == 0:

                        helmet_count += 1

                    elif class_id == 1:

                        vest_count += 1

            total_objects = (
                helmet_count + vest_count
            )

            st.divider()

            st.header("📊 Detection Summary")

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "⛑️ Safety Helmets",
                    helmet_count
                )

            with c2:

                st.metric(
                    "🦺 Safety Vests",
                    vest_count
                )

            with c3:

                st.metric(
                    "📦 Total Objects",
                    total_objects
                )


            st.divider()

            st.header("🚨 Safety Assessment")

            if helmet_count > 0 and vest_count > 0:

                st.success(
                    "✅ SAFETY EQUIPMENT DETECTED\n\n"
                    "Helmet and safety vest detected."
                )

            elif helmet_count > 0:

                st.warning(
                    "⚠️ PARTIAL SAFETY COMPLIANCE\n\n"
                    "Helmet detected, but safety vest was not detected."
                )

            elif vest_count > 0:

                st.warning(
                    "⚠️ PARTIAL SAFETY COMPLIANCE\n\n"
                    "Safety vest detected, but helmet was not detected."
                )

            else:

                st.error(
                    "🚨 SAFETY EQUIPMENT NOT DETECTED\n\n"
                    "No helmet or safety vest detected."
                )

        except Exception as e:

            st.error(
                "❌ Detection failed."
            )

            st.exception(e)


st.divider()

st.caption(
    "SafeBuild Intelligence • "
    "Construction Safety Detection"
)
