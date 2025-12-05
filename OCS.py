import streamlit as st
import sqlite3
import os

st.set_page_config(page_title="Manual Técnico – Procedimentos", layout="wide")
st.title("📘 Manual Técnico – Cadastro de Procedimentos")

db = sqlite3.connect("manual.db")
cursor = db.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS procedures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    category TEXT,
    description TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    procedure_id INTEGER,
    step_number INTEGER,
    text TEXT,
    image_path TEXT
)
""")

db.commit()

# ------------------------------
# Criar novo procedimento
# ------------------------------
st.header("➕ Criar novo procedimento")

name = st.text_input("Nome do procedimento")
category = st.selectbox("Categoria", [
    "Manutenção", "Instalação", "Software", "Hardware", "Diagnóstico"
])
description = st.text_area("Descrição geral")

if st.button("Salvar procedimento"):
    cursor.execute("INSERT INTO procedures (name, category, description) VALUES (?, ?, ?)",
                   (name, category, description))
    db.commit()
    st.success("Procedimento criado!")

st.markdown("---")

# ------------------------------
# Adicionar passos
# ------------------------------
st.header("📑 Adicionar passos a um procedimento")

procedures = cursor.execute("SELECT id, name FROM procedures").fetchall()
procedures_dict = {name: pid for pid, name in procedures}

selected = st.selectbox("Escolha o procedimento:", list(procedures_dict.keys()))

step_text = st.text_area("Descrição do passo")
step_image = st.file_uploader("Foto ilustrativa", type=["jpg","png"])

if st.button("Adicionar passo"):
    pid = procedures_dict[selected]

    # salvar foto localmente
    img_path = None
    if step_image:
        img_path = os.path.join("imagens", step_image.name)
        with open(img_path, "wb") as f:
            f.write(step_image.getbuffer())

    cursor.execute("""
    INSERT INTO steps (procedure_id, step_number, text, image_path)
    VALUES (?, 
        (SELECT IFNULL(MAX(step_number), 0) + 1 FROM steps WHERE procedure_id=?),
        ?, ?
    )
    """, (pid, pid, step_text, img_path))

    db.commit()
    st.success("Passo adicionado!")
