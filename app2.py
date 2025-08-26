try:
    # 📥 Load the CSV
    df = pd.read_csv(file_path)

    # 🧹 Remove colunas Unnamed e espaços
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = df.columns.str.strip()
    df = df.dropna(axis=1, how='all')

    # 📆 Corrigir coluna Date (mesmo se for string)
    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Date"] = df["Date"].dt.date
    else:
        st.error("❌ O arquivo não contém a coluna 'Date'.")
        st.stop()

    # 📅 Filtro pelo dia selecionado
    df_filtered = df[df["Date"] == selected_date]

    # 🚫 Corrigir liga antes do filtro
    if "League" in df_filtered.columns:
        df_filtered["League"] = df_filtered["League"].astype(str).str.strip()
        if EXCLUDED_LEAGUE_KEYWORDS:
            pattern = "|".join(map(re.escape, EXCLUDED_LEAGUE_KEYWORDS))
            df_filtered = df_filtered[~df_filtered["League"].str.contains(pattern, case=False, na=False)]

    # 👁️ Remove 'Date' column from display and reset index
    df_display = df_filtered.drop(columns=["Date"], errors="ignore")
    df_display.index = range(len(df_display))

    # ⚠️ Mostrar alerta se nada restar
    if df_filtered.empty:
        st.warning("⚠️ Nenhum jogo encontrado para a data selecionada após aplicar os filtros.")
    else:
        # 🔢 Calcula médias só das colunas que existem
        mean_cols = {col: df_display[col].mean() for col in ["M_HT_H", "M_HT_A", "M_H", "M_A"] if col in df_display.columns}

        # ✅ Cria estilo com segurança
        styled = (
            df_display.style
            .format({
                "Odd_H": "{:.2f}" if "Odd_H" in df_display.columns else None,
                "Odd_D": "{:.2f}" if "Odd_D" in df_display.columns else None,
                "Odd_A": "{:.2f}" if "Odd_A" in df_display.columns else None,
                "Diff_HT_P": "{:.2f}" if "Diff_HT_P" in df_display.columns else None,
                "Diff_Power": "{:.2f}" if "Diff_Power" in df_display.columns else None,
                "OU_Total": (lambda x: f"{x * 100:.2f}") if "OU_Total" in df_display.columns else None,
                "Goals_H_FT": (lambda x: f"{int(x)}") if "Goals_H_FT" in df_display.columns else None,
                "Goals_A_FT": (lambda x: f"{int(x)}") if "Goals_A_FT" in df_display.columns else None,
                "M_HT_H": (lambda x: arrow_trend(x, mean_cols["M_HT_H"])) if "M_HT_H" in mean_cols else None,
                "M_HT_A": (lambda x: arrow_trend(x, mean_cols["M_HT_A"])) if "M_HT_A" in mean_cols else None,
                "M_H": (lambda x: arrow_trend(x, mean_cols["M_H"])) if "M_H" in mean_cols else None,
                "M_A": (lambda x: arrow_trend(x, mean_cols["M_A"])) if "M_A" in mean_cols else None,
            })
            .background_gradient(cmap="RdYlGn", subset=[c for c in ["Diff_HT_P","Diff_Power"] if c in df_display.columns])
            .background_gradient(cmap="Blues", subset=[c for c in ["OU_Total"] if c in df_display.columns])
        )

        # ✅ Show styled table
        st.dataframe(styled, height=1200, use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ Arquivo `{filename}` não encontrado.")
except pd.errors.EmptyDataError:
    st.error(f"❌ O arquivo `{filename}` está vazio ou inválido.")
except Exception as e:
    st.error(f"⚠️ Erro inesperado: {e}")
