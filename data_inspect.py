import streamlit as st
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Set page config
st.set_page_config(page_title="OCR Data Inspector", layout="wide")

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'inspection_results' not in st.session_state:
    st.session_state.inspection_results = {}
if 'edited_texts' not in st.session_state:
    st.session_state.edited_texts = {}
if 'show_exit_dialog' not in st.session_state:
    st.session_state.show_exit_dialog = False

# Load data
@st.cache_data
def load_data():
    csv_path = "data/processed/ocr_output.csv"
    df = pd.read_csv(csv_path)
    return df

# Load the data
df = load_data()
page_ids = df['page_id'].tolist()
total_pages = len(page_ids)

# Header with End Inspection button
col1, col2 = st.columns([6, 1])
with col1:
    st.title("OCR Data Inspector")
with col2:
    if st.button("🏁 End Inspection", type="primary"):
        st.session_state.show_exit_dialog = True

# Show exit dialog
if st.session_state.show_exit_dialog:
    st.warning("### Do you want to save the inspection results?")
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ Save Results"):
            # Save inspection results
            if st.session_state.inspection_results or st.session_state.edited_texts:
                results_data = []
                for page_id in set(list(st.session_state.inspection_results.keys()) + list(st.session_state.edited_texts.keys())):
                    status = st.session_state.inspection_results.get(page_id, "not_marked")
                    edited_text = st.session_state.edited_texts.get(page_id, None)
                    results_data.append({
                        "page_id": page_id,
                        "status": status,
                        "edited_text": edited_text if edited_text else "",
                        "timestamp": datetime.now().isoformat()
                    })
                results_df = pd.DataFrame(results_data)
                output_path = "data/processed/inspection_results.csv"
                results_df.to_csv(output_path, index=False)
                st.success(f"✅ Results saved to {output_path}")
                st.info(f"Total inspected: {len(st.session_state.inspection_results)} pages | Edited: {len(st.session_state.edited_texts)} pages")
            else:
                st.info("No inspection results to save.")
            st.session_state.show_exit_dialog = False
            st.stop()
    with col2:
        if st.button("❌ Don't Save"):
            st.info("Exiting without saving.")
            st.session_state.show_exit_dialog = False
            st.stop()
    st.divider()

# Get current page data
current_page_id = page_ids[st.session_state.current_index]
current_text = df[df['page_id'] == current_page_id]['text'].values[0]
# Use edited text if available
if current_page_id in st.session_state.edited_texts:
    current_text = st.session_state.edited_texts[current_page_id]
image_path = f"data/raw/pdf_images/page_{current_page_id}.png"

# Display page_id centered at the top
st.markdown(f"<h2 style='text-align: center;'>Page ID: {current_page_id}</h2>", unsafe_allow_html=True)
st.divider()

# Navigation buttons
col1, col2, col3, col4, col5 = st.columns([1, 1, 3, 1, 1])

with col1:
    if st.button("⬅️ Previous", disabled=(st.session_state.current_index == 0)):
        st.session_state.current_index -= 1
        st.rerun()

with col2:
    st.write(f"**{st.session_state.current_index + 1} / {total_pages}**")

with col4:
    if st.button("Next ➡️", disabled=(st.session_state.current_index == total_pages - 1)):
        st.session_state.current_index += 1
        st.rerun()

st.divider()

# Main content: Image and Text in parallel
col_img, col_text = st.columns([1, 1])

with col_img:
    st.subheader("📷 Image")
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.error(f"Image not found: {image_path}")

with col_text:
    st.subheader("📝 OCR Text (Editable)")
    edited_text = st.text_area(
        "Extracted Text", 
        current_text, 
        height=1220, 
        key=f"text_area_{current_page_id}",
        help="You can edit this text. Changes will be saved when you mark the page or end inspection."
    )
    # Save edited text to session state
    if edited_text != df[df['page_id'] == current_page_id]['text'].values[0]:
        st.session_state.edited_texts[current_page_id] = edited_text
        st.caption("✏️ *Text has been edited*")
    elif current_page_id in st.session_state.edited_texts and edited_text == df[df['page_id'] == current_page_id]['text'].values[0]:
        # Remove from edited_texts if reverted to original
        del st.session_state.edited_texts[current_page_id]

st.divider()

# Confirmation buttons
st.subheader("Mark OCR Quality")
col1, col2 = st.columns([1, 1])

# Show current status if already marked
if current_page_id in st.session_state.inspection_results:
    current_status = st.session_state.inspection_results[current_page_id]
    status_emoji = "✅" if current_status == "good" else "❌"
    st.info(f"Current status: {status_emoji} **{current_status.upper()}**")

with col1:
    if st.button("Good OCR", width='stretch', type="primary"):
        st.session_state.inspection_results[current_page_id] = "good"
        st.success("Marked as GOOD!")
        st.rerun()

with col2:
    if st.button("Bad OCR", width='stretch'):
        st.session_state.inspection_results[current_page_id] = "bad"
        st.error("Marked as BAD!")
        st.rerun()

# Show inspection progress
st.divider()
inspected_count = len(st.session_state.inspection_results)
good_count = sum(1 for status in st.session_state.inspection_results.values() if status == "good")
bad_count = sum(1 for status in st.session_state.inspection_results.values() if status == "bad")

st.caption(f"📊 Progress: {inspected_count}/{total_pages} pages inspected | ✅ Good: {good_count} | ❌ Bad: {bad_count}")
