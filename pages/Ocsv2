from __future__ import annotations
import streamlit as st
from tinydb import TinyDB, Query
import os
from datetime import datetime
import pandas as pd

# 🔴 NEW: html5-qrcode based scanner
from streamlit_qrcode_scanner import qrcode_scanner

# ==========================
# INITIAL SETTINGS
# ==========================

st.set_page_config(page_title="Technical Manual – Procedures", layout="wide")

st.title("📘 Technical Manual – Machines / Equipment")

DB_PATH = "manual_db.json"
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

db = TinyDB(DB_PATH)
procedures_table = db.table("procedures")
steps_table = db.table("steps")
parts_table = db.table("parts")
serials_table = db.table("serials")
Q = Query()

# ==========================
# HELPER FUNCTIONS
# ==========================

def save_image(uploaded_file, prefix: str) -> str | None:
    """Saves an uploaded image and returns the relative path."""
    if not uploaded_file:
        return None
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
    path = os.path.join(IMAGES_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path

def get_procedure_choices():
    procs = procedures_table.all()
    if not procs:
        return {}, []
    mapping = {f"{p['name']} (ID {p.doc_id})": p.doc_id for p in procs}
    labels = list(mapping.keys())
    return mapping, labels

def get_part_choices():
    parts = parts_table.all()
    if not parts:
        return {}, []
    mapping = {
        f"{p['name']} – {p.get('machine_model', 'Model not provided')} (ID {p.doc_id})": p.doc_id
        for p in parts
    }
    labels = list(mapping.keys())
    return mapping, labels


# ==========================
# PAGES
# ==========================

def page_view_manual():
    st.header("📚 View Procedure Manual")

    procs = procedures_table.all()
    if not procs:
        st.info("No procedures registered yet. Go to **'➕ Register Procedure'** to add the first one.")
        return

    categories = sorted(set(p.get("category", "Uncategorized") for p in procs))
    
    col1, col2 = st.columns([1, 2])
    with col1:
        cat_filter = st.selectbox("Filter by category:", ["All"] + categories)
    with col2:
        text_filter = st.text_input("Search by name / description:")

    for p in procs:
        cat_ok = (cat_filter == "All") or (p.get("category") == cat_filter)

        text_ok = True
        if text_filter:
            text = (p.get("name", "") + " " + p.get("description", "")).lower()
            text_ok = text_filter.lower() in text

        if not (cat_ok and text_ok):
            continue

        with st.expander(
            f"📘 {p['name']} – {p.get('category', 'Uncategorized')} (ID {p.doc_id})",
            expanded=False
        ):
            st.markdown(f"**Description:** {p.get('description', 'No description')}")
            st.caption(f"Created on: {p.get('created_at', 'Unknown')}")

            steps = steps_table.search(Q.procedure_id == p.doc_id)
            if not steps:
                st.warning("No steps registered for this procedure yet.")
            else:
                steps_sorted = sorted(steps, key=lambda s: s.get("step_number", 0))

                for s_step in steps_sorted:
                    st.markdown(f"### Step {s_step.get('step_number', '?')}")
                    st.write(s_step.get("text", ""))

                    img_path = s_step.get("image_path")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)

                    st.markdown("---")


def page_add_procedure():
    st.header("➕ Register New Procedure")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Procedure name*", placeholder="Ex: PM – Preventive Maintenance")
        category = st.text_input("Category*", placeholder="Ex: Maintenance, Installation, Software, Hardware")
    with col2:
        machine_model = st.text_input("Equipment model (optional)", placeholder="Ex: SST RDL 39893")

    description = st.text_area(
        "General procedure description",
        placeholder="Describe here the purpose, context, and type of machine for this procedure."
    )

    if st.button("Save procedure", type="primary"):
        if not name or not category:
            st.error("Name and category are required.")
        else:
            pid = procedures_table.insert({
                "name": name,
                "category": category,
                "machine_model": machine_model,
                "description": description,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success(f"Procedure saved successfully! (ID {pid})")

    st.markdown("---")
    st.subheader("📄 Previously registered procedures")

    procs = procedures_table.all()
    if procs:
        df = pd.DataFrame(
            [
                {
                    "ID": p.doc_id,
                    "Name": p.get("name"),
                    "Category": p.get("category"),
                    "Model": p.get("machine_model", ""),
                    "Created on": p.get("created_at", "")
                }
                for p in procs
            ]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No procedures registered yet.")



def page_add_steps():
    st.header("🧩 Register Procedure Steps")

    mapping, labels = get_procedure_choices()
    if not labels:
        st.warning("No procedures found. First register one in **'➕ Register Procedure'**.")
        return

    selected_label = st.selectbox("Choose a procedure:", labels)
    selected_pid = mapping[selected_label]

    st.markdown(f"Selected: **{selected_label}**")

    step_text = st.text_area(
        "Step description",
        placeholder="Ex: Open the machine’s front door and take a general picture of the interior."
    )

    step_image = st.file_uploader("Illustrative image (optional)", type=["jpg", "jpeg", "png"])

    if st.button("Add step", type="primary"):
        if not step_text:
            st.error("Step description is required.")
        else:
            existing = steps_table.search(Q.procedure_id == selected_pid)
            next_number = max((s.get("step_number", 0) for s in existing), default=0) + 1

            img_path = None
            if step_image:
                img_path = save_image(step_image, f"proc{selected_pid}_step{next_number}")

            steps_table.insert({
                "procedure_id": selected_pid,
                "step_number": next_number,
                "text": step_text,
                "image_path": img_path,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

            st.success(f"Step {next_number} added to the procedure!")

    st.markdown("---")
    st.subheader("📑 Steps for this procedure")

    steps = steps_table.search(Q.procedure_id == selected_pid)
    if not steps:
        st.info("No steps registered for this procedure yet.")
    else:
        steps_sorted = sorted(steps, key=lambda s: s.get("step_number", 0))

        for s_step in steps_sorted:
            with st.expander(f"Step {s_step.get('step_number', '?')} – {s_step.get('text', '')[:40]}..."):
                st.write(s_step.get("text", ""))

                img_path = s_step.get("image_path")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)

                st.caption(f"Registered on: {s_step.get('created_at', '')}")


def page_parts_and_serials():
    st.header("🔧 Parts & Serial Numbers")

    tab1, tab2 = st.tabs(["📍 Machine Parts", "🔢 Serial Numbers"])

    # -------- TAB 1: PARTS --------
    with tab1:
        st.subheader("📍 Register New Part / Component")

        col1, col2 = st.columns(2)
        with col1:
            part_name = st.text_input("Part name*", placeholder="Ex: Input roller, Optical sensor, Logic board")
            machine_model = st.text_input("Machine model", placeholder="Ex: SST RDL 39893")

        with col2:
            location_description = st.text_area(
                "Location inside the machine*",
                placeholder="Describe where this part is located (e.g., 'Front area, right side, behind panel X').",
                height=100
            )

        part_notes = st.text_area("Additional notes (optional)")
        part_image = st.file_uploader("Photo of the part / location (optional)", type=["jpg", "jpeg", "png"])

        if st.button("Save part", type="primary", key="save_part"):
            if not part_name or not location_description:
                st.error("Part name and location are required.")
            else:
                img_path = None
                if part_image:
                    img_path = save_image(part_image, f"part_{part_name.replace(' ', '_')}")

                pid = parts_table.insert({
                    "name": part_name,
                    "machine_model": machine_model,
                    "location_description": location_description,
                    "notes": part_notes,
                    "image_path": img_path,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                })

                st.success(f"Part registered successfully! (ID {pid})")

        st.markdown("---")
        st.subheader("Registered parts list")

        parts = parts_table.all()
        if parts:
            df_parts = pd.DataFrame(
                [
                    {
                        "ID": p.doc_id,
                        "Part": p.get("name"),
                        "Model": p.get("machine_model", ""),
                        "Location": p.get("location_description", ""),
                        "Created on": p.get("created_at", "")
                    }
                    for p in parts
                ]
            )
            st.dataframe(df_parts, use_container_width=True)
        else:
            st.info("No parts registered yet.")

    # -------- TAB 2: SERIALS (Manual Entry) --------
    with tab2:
        st.subheader("🔢 Register part serial number (manual input)")

        mapping, labels = get_part_choices()
        if not labels:
            st.warning("No parts registered yet. Add at least one in the **'Machine Parts'** tab.")
            return

        part_label = st.selectbox("Choose the part:", labels)
        part_id = mapping[part_label]

        col1, col2 = st.columns(2)
        with col1:
            serial_text = st.text_input("Serial number*", placeholder="Ex: SN-394823984")
        with col2:
            technician = st.text_input("Technician", placeholder="Ex: Wagner")

        machine_tag = st.text_input("Machine ID / Tag (optional)", placeholder="Ex: SCO-001, KIOSK-22")
        serial_notes = st.text_area("Notes (optional)")

        if st.button("Save serial", type="primary", key="save_serial"):
            if not serial_text:
                st.error("Serial number is required.")
            else:
                serials_table.insert({
                    "part_id": part_id,
                    "serial_text": serial_text,
                    "technician": technician,
                    "machine_tag": machine_tag,
                    "notes": serial_notes,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "source": "manual"
                })

                st.success("Serial registered successfully!")

        st.markdown("---")
        st.subheader("Registered serial numbers")

        all_serials = serials_table.all()
        if all_serials:
            rows = []
            for s_doc in all_serials:
                part = parts_table.get(doc_id=s_doc["part_id"])
                rows.append({
                    "Serial ID": s_doc.doc_id,
                    "Part": part.get("name") if part else "Part not found",
                    "Model": part.get("machine_model", "") if part else "",
                    "Machine Tag": s_doc.get("machine_tag", ""),
                    "Serial": s_doc.get("serial_text", ""),
                    "Technician": s_doc.get("technician", ""),
                    "Registered on": s_doc.get("created_at", ""),
                    "Source": s_doc.get("source", ""),
                    "Notes": s_doc.get("notes", "")
                })
            df_serials = pd.DataFrame(rows)
            st.dataframe(df_serials, use_container_width=True)
        else:
            st.info("No serials registered yet.")



# ==========================
# 📷 MOBILE SERIAL SCANNER (html5-qrcode)
# ==========================

def page_serial_scanner():
    st.header("📷 Serial Scanner (Mobile)")

    mapping, labels = get_part_choices()
    if not labels:
        st.warning("No parts registered yet. Add at least one part in **'Parts & Serials'**.")
        return

    part_label = st.selectbox("Part / Component:", labels)
    part_id = mapping[part_label]

    col1, col2 = st.columns(2)
    with col1:
        machine_tag = st.text_input("Machine ID / Tag*", placeholder="Ex: SCO-001, KIOSK-22")
    with col2:
        technician = st.text_input("Technician*", value="Wagner")

    st.markdown("Point your mobile camera at the part’s barcode or QR code.")

    # 👇 Opens the camera and returns the scanned text
    code = qrcode_scanner(key="barcode_scanner")

    if code:
        st.success(f"Serial captured: **{code}**")
        st.session_state["scanned_code"] = code

    st.markdown("---")
    st.subheader("Save captured serial")

    if "scanned_code" in st.session_state:
        serial_notes = st.text_area("Notes (optional)", key="scanner_notes")

        if st.button("💾 Save this serial", type="primary"):
            if not machine_tag or not technician:
                st.error("Fill Machine Tag and Technician before saving.")
            else:
                serials_table.insert({
                    "part_id": part_id,
                    "serial_text": st.session_state["scanned_code"],
                    "technician": technician,
                    "machine_tag": machine_tag,
                    "notes": serial_notes,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "source": "mobile_scanner"
                })
                st.success("Serial saved to database!")
    else:
        st.info("No code detected yet. Aim the camera at the part's barcode.")

# ==========================
# SERIAL REPORT PAGE
# ==========================

def page_serial_report():
    st.header("📄 Serial Numbers Report – All Parts")

    all_serials = serials_table.all()
    if not all_serials:
        st.info("No serials registered yet.")
        return

    rows = []
    for s_doc in all_serials:
        part = parts_table.get(doc_id=s_doc["part_id"])
        rows.append({
            "Part": part.get("name") if part else "Part not found",
            "Machine Model": part.get("machine_model", "") if part else "",
            "Location in Machine": part.get("location_description", "") if part else "",
            "Machine Tag": s_doc.get("machine_tag", ""),
            "Serial Number": s_doc.get("serial_text", ""),
            "Technician": s_doc.get("technician", ""),
            "Registered on": s_doc.get("created_at", ""),
            "Source": s_doc.get("source", ""),
            "Notes": s_doc.get("notes", "")
        })

    df_report = pd.DataFrame(rows)

    st.subheader("Consolidated serial table")
    st.dataframe(df_report, use_container_width=True)

    csv_bytes = df_report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download CSV report",
        data=csv_bytes,
        file_name=f"serial_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    st.caption("Open in Excel / Google Sheets to format and print like a 'Component List'.")



# ==========================
# NAVIGATION
# ==========================

menu = st.sidebar.radio(
    "Navigation",
    [
        "📘 View Manual",
        "➕ Register Procedure",
        "🧩 Register Steps",
        "🔧 Parts & Serials",
        "📷 Serial Scanner (Mobile)",
        "📄 Serial Report",
    ]
)

if menu == "📘 View Manual":
    page_view_manual()
elif menu == "➕ Register Procedure":
    page_add_procedure()
elif menu == "🧩 Register Steps":
    page_add_steps()
elif menu == "🔧 Parts & Serials":
    page_parts_and_serials()
elif menu == "📷 Serial Scanner (Mobile)":
    page_serial_scanner()
elif menu == "📄 Serial Report":
    page_serial_report()
