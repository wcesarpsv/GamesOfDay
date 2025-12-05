from __future__ import annotations
import streamlit as st
from tinydb import TinyDB, Query
import os
from datetime import datetime
import pandas as pd

# 🔴 NOVO: imports para o scanner
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from pyzbar.pyzbar import decode
import cv2
import av

# ==========================
# CONFIGURAÇÕES INICIAIS
# ==========================

st.set_page_config(page_title="Manual Técnico – Procedimentos", layout="wide")

st.title("📘 Manual Técnico – Máquinas / Equipamentos")

DB_PATH = "manual_db.json"
IMAGES_DIR = "imagens"
os.makedirs(IMAGES_DIR, exist_ok=True)

db = TinyDB(DB_PATH)
procedures_table = db.table("procedures")
steps_table = db.table("steps")
parts_table = db.table("parts")
serials_table = db.table("serials")
Q = Query()

# ==========================
# FUNÇÕES AUXILIARES
# ==========================

def save_image(uploaded_file, prefix: str) -> str | None:
    """Salva imagem enviada e retorna o caminho relativo."""
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
        f"{p['name']} – {p.get('machine_model', 'Modelo não informado')} (ID {p.doc_id})": p.doc_id
        for p in parts
    }
    labels = list(mapping.keys())
    return mapping, labels

# ==========================
# PÁGINAS
# ==========================

def page_view_manual():
    st.header("📚 Visualizar Manual de Procedimentos")

    procs = procedures_table.all()
    if not procs:
        st.info("Nenhum procedimento cadastrado ainda. Vá em **'➕ Cadastrar Procedimento'** para adicionar o primeiro.")
        return

    # Filtro por categoria e por texto
    categorias = sorted(set(p.get("category", "Sem categoria") for p in procs))
    col1, col2 = st.columns([1, 2])
    with col1:
        cat_filter = st.selectbox("Filtrar por categoria:", ["Todas"] + categorias)
    with col2:
        text_filter = st.text_input("Buscar por nome / descrição:")

    for p in procs:
        cat_ok = (cat_filter == "Todas") or (p.get("category") == cat_filter)
        text_ok = True
        if text_filter:
            texto = (p.get("name", "") + " " + p.get("description", "")).lower()
            text_ok = text_filter.lower() in texto

        if not (cat_ok and text_ok):
            continue

        with st.expander(f"📘 {p['name']}  –  {p.get('category', 'Sem categoria')} (ID {p.doc_id})", expanded=False):
            st.markdown(f"**Descrição:** {p.get('description', 'Sem descrição')}")
            st.caption(f"Criado em: {p.get('created_at', 'Desconhecido')}")

            steps = steps_table.search(Q.procedure_id == p.doc_id)
            if not steps:
                st.warning("Nenhum passo cadastrado ainda para este procedimento.")
            else:
                steps_sorted = sorted(steps, key=lambda s: s.get("step_number", 0))
                for s_step in steps_sorted:
                    st.markdown(f"### Passo {s_step.get('step_number', '?')}")
                    st.write(s_step.get("text", ""))

                    img_path = s_step.get("image_path")
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)
                    st.markdown("---")


def page_add_procedure():
    st.header("➕ Cadastrar Novo Procedimento")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Nome do procedimento*", placeholder="Ex: PM – Preventive Maintenance")
        category = st.text_input("Categoria*", placeholder="Ex: Manutenção, Instalação, Software, Hardware")
    with col2:
        machine_model = st.text_input("Modelo do equipamento (opcional)", placeholder="Ex: SST RDL 39893")

    description = st.text_area(
        "Descrição geral do procedimento",
        placeholder="Descreva aqui o objetivo desse procedimento, contexto, tipo de máquina, etc."
    )

    if st.button("Salvar procedimento", type="primary"):
        if not name or not category:
            st.error("Nome e categoria são obrigatórios.")
        else:
            pid = procedures_table.insert({
                "name": name,
                "category": category,
                "machine_model": machine_model,
                "description": description,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
            st.success(f"Procedimento salvo com sucesso! (ID {pid})")

    st.markdown("---")
    st.subheader("📄 Procedimentos já cadastrados")
    procs = procedures_table.all()
    if procs:
        df = pd.DataFrame(
            [
                {
                    "ID": p.doc_id,
                    "Nome": p.get("name"),
                    "Categoria": p.get("category"),
                    "Modelo": p.get("machine_model", ""),
                    "Criado em": p.get("created_at", "")
                }
                for p in procs
            ]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Ainda não há procedimentos cadastrados.")


def page_add_steps():
    st.header("🧩 Cadastrar Passos para Procedimentos")

    mapping, labels = get_procedure_choices()
    if not labels:
        st.warning("Nenhum procedimento encontrado. Cadastre um em **'➕ Cadastrar Procedimento'** primeiro.")
        return

    selected_label = st.selectbox("Escolha o procedimento:", labels)
    selected_pid = mapping[selected_label]

    st.markdown(f"Selecionado: **{selected_label}**")

    step_text = st.text_area(
        "Descrição do passo",
        placeholder="Ex: Abrir a porta frontal da máquina e tirar foto geral do interior."
    )
    step_image = st.file_uploader("Imagem ilustrativa do passo (opcional)", type=["jpg", "jpeg", "png"])

    if st.button("Adicionar passo", type="primary"):
        if not step_text:
            st.error("A descrição do passo é obrigatória.")
        else:
            existing = steps_table.search(Q.procedure_id == selected_pid)
            if existing:
                next_number = max(s.get("step_number", 0) for s in existing) + 1
            else:
                next_number = 1

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
            st.success(f"Passo {next_number} adicionado ao procedimento!")

    st.markdown("---")
    st.subheader("📑 Passos deste procedimento")

    steps = steps_table.search(Q.procedure_id == selected_pid)
    if not steps:
        st.info("Ainda não há passos cadastrados para este procedimento.")
    else:
        steps_sorted = sorted(steps, key=lambda s: s.get("step_number", 0))
        for s_step in steps_sorted:
            with st.expander(f"Passo {s_step.get('step_number', '?')} – {s_step.get('text', '')[:40]}..."):
                st.write(s_step.get("text", ""))
                img_path = s_step.get("image_path")
                if img_path and os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                st.caption(f"Registrado em: {s_step.get('created_at', '')}")


def page_parts_and_serials():
    st.header("🔧 Cadastro de Peças e Seriais")

    tab1, tab2 = st.tabs(["📍 Peças na máquina", "🔢 Seriais das peças"])

    # ----------------- TAB 1: PEÇAS -----------------
    with tab1:
        st.subheader("📍 Cadastrar nova peça / componente")

        col1, col2 = st.columns(2)
        with col1:
            part_name = st.text_input("Nome da peça*", placeholder="Ex: Roller de entrada, Sensor óptico, Placa lógica")
            machine_model = st.text_input("Modelo da máquina", placeholder="Ex: SST RDL 39893")
        with col2:
            location_description = st.text_area(
                "Localização na máquina*",
                placeholder="Descreva onde essa peça fica na máquina (ex: 'Parte frontal, lado direito, atrás do painel X').",
                height=100
            )

        part_notes = st.text_area("Observações adicionais (opcional)")
        part_image = st.file_uploader("Foto da peça / localização (opcional)", type=["jpg", "jpeg", "png"])

        if st.button("Salvar peça", type="primary", key="save_part"):
            if not part_name or not location_description:
                st.error("Nome da peça e localização são obrigatórios.")
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
                st.success(f"Peça cadastrada com sucesso! (ID {pid})")

        st.markdown("---")
        st.subheader("Lista de peças cadastradas")

        parts = parts_table.all()
        if parts:
            df_parts = pd.DataFrame(
                [
                    {
                        "ID": p.doc_id,
                        "Peça": p.get("name"),
                        "Modelo": p.get("machine_model", ""),
                        "Localização": p.get("location_description", ""),
                        "Criado em": p.get("created_at", "")
                    }
                    for p in parts
                ]
            )
            st.dataframe(df_parts, use_container_width=True)
        else:
            st.info("Nenhuma peça cadastrada ainda.")

    # ----------------- TAB 2: SERIAIS -----------------
    with tab2:
        st.subheader("🔢 Registrar serial number de peça")

        mapping, labels = get_part_choices()
        if not labels:
            st.warning("Nenhuma peça cadastrada. Cadastre pelo menos uma peça na aba **'Peças na máquina'**.")
            return

        part_label = st.selectbox("Escolha a peça:", labels)
        part_id = mapping[part_label]

        col1, col2 = st.columns(2)
        with col1:
            serial_text = st.text_input("Serial number*", placeholder="Ex: SN-394823984")
        with col2:
            technician = st.text_input("Técnico responsável", placeholder="Ex: Wagner")

        machine_tag = st.text_input("ID / Tag da máquina (opcional)", placeholder="Ex: SCO-001, KIOSK-22")
        serial_notes = st.text_area("Observações (opcional)")

        if st.button("Salvar serial", type="primary", key="save_serial"):
            if not serial_text:
                st.error("O serial number é obrigatório.")
            else:
                serials_table.insert({
                    "part_id": part_id,
                    "serial_text": serial_text,
                    "technician": technician,
                    "machine_tag": machine_tag,
                    "notes": serial_notes,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                st.success("Serial registrado com sucesso!")

        st.markdown("---")
        st.subheader("Seriais registrados")

        all_serials = serials_table.all()
        if all_serials:
            rows = []
            for s_doc in all_serials:
                part = parts_table.get(doc_id=s_doc["part_id"])
                rows.append({
                    "ID Serial": s_doc.doc_id,
                    "Peça": part.get("name") if part else "Peça não encontrada",
                    "Modelo": part.get("machine_model", "") if part else "",
                    "Machine Tag": s_doc.get("machine_tag", ""),
                    "Serial": s_doc.get("serial_text", ""),
                    "Técnico": s_doc.get("technician", ""),
                    "Data Registro": s_doc.get("created_at", ""),
                    "Observações": s_doc.get("notes", "")
                })
            df_serials = pd.DataFrame(rows)
            st.dataframe(df_serials, use_container_width=True)
        else:
            st.info("Ainda não há seriais registrados.")


def page_serial_report():
    st.header("📄 Relatório de Serial Numbers – Todas as Peças")

    all_serials = serials_table.all()
    if not all_serials:
        st.info("Nenhum serial registrado ainda.")
        return

    rows = []
    for s_doc in all_serials:
        part = parts_table.get(doc_id=s_doc["part_id"])
        rows.append({
            "Peça": part.get("name") if part else "Peça não encontrada",
            "Modelo da Máquina": part.get("machine_model", "") if part else "",
            "Localização na Máquina": part.get("location_description", "") if part else "",
            "Machine Tag": s_doc.get("machine_tag", ""),
            "Serial Number": s_doc.get("serial_text", ""),
            "Técnico": s_doc.get("technician", ""),
            "Data Registro": s_doc.get("created_at", ""),
            "Observações": s_doc.get("notes", "")
        })

    df_report = pd.DataFrame(rows)

    st.subheader("Tabela consolidada de seriais")
    st.dataframe(df_report, use_container_width=True)

    csv_bytes = df_report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Baixar relatório em CSV (para imprimir / enviar)",
        data=csv_bytes,
        file_name=f"relatorio_seriais_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

    st.caption("Você pode abrir esse CSV no Excel ou Google Sheets e imprimir como relatório oficial.")


# ==========================
# 📷 PÁGINA DO SCANNER MOBILE
# ==========================

class SerialScanner(VideoProcessorBase):
    def __init__(self):
        self.last_serial = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        barcodes = decode(img)
        for b in barcodes:
            serial = b.data.decode("utf-8")
            self.last_serial = serial

            # desenha retângulo em volta do código
            x, y, w, h = b.rect
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def page_serial_scanner():
    st.header("📷 Scanner de Seriais (Mobile)")

    mapping, labels = get_part_choices()
    if not labels:
        st.warning("Nenhuma peça cadastrada. Cadastre pelo menos uma peça em **'Peças & Seriais'**.")
        return

    part_label = st.selectbox("Peça / componente:", labels)
    part_id = mapping[part_label]

    col1, col2 = st.columns(2)
    with col1:
        machine_tag = st.text_input("ID / Tag da máquina*", placeholder="Ex: SCO-001, KIOSK-22")
    with col2:
        technician = st.text_input("Técnico*", value="Wagner")

    st.markdown("Toque em **‘Start’** abaixo e aponte a câmera para o código de barras do componente.")

    ctx = webrtc_streamer(
        key="serial-scanner",
        video_processor_factory=SerialScanner,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        serial = ctx.video_processor.last_serial
        if serial:
            st.session_state["scanned_serial"] = serial

    st.markdown("---")
    st.subheader("Serial capturado")

    if "scanned_serial" in st.session_state:
        st.success(f"Serial lido: **{st.session_state['scanned_serial']}**")
        serial_notes = st.text_area("Observações (opcional)", key="scanner_notes")

        if st.button("💾 Salvar este serial", type="primary"):
            if not machine_tag or not technician:
                st.error("Preencha pelo menos Machine Tag e Técnico.")
            else:
                serials_table.insert({
                    "part_id": part_id,
                    "serial_text": st.session_state["scanned_serial"],
                    "technician": technician,
                    "machine_tag": machine_tag,
                    "notes": serial_notes,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "source": "camera_scanner"
                })
                st.success("Serial salvo no banco de dados!")
    else:
        st.info("Ainda nenhum código detectado. Aponte a câmera para o barcode.")


# ==========================
# NAVEGAÇÃO
# ==========================

menu = st.sidebar.radio(
    "Navegação",
    [
        "📘 Ver Manual",
        "➕ Cadastrar Procedimento",
        "🧩 Cadastrar Passos",
        "🔧 Peças & Seriais",
        "📷 Scanner de Seriais (Mobile)",
        "📄 Relatório de Seriais",
    ]
)

if menu == "📘 Ver Manual":
    page_view_manual()
elif menu == "➕ Cadastrar Procedimento":
    page_add_procedure()
elif menu == "🧩 Cadastrar Passos":
    page_add_steps()
elif menu == "🔧 Peças & Seriais":
    page_parts_and_serials()
elif menu == "📷 Scanner de Seriais (Mobile)":
    page_serial_scanner()
elif menu == "📄 Relatório de Seriais":
    page_serial_report()
