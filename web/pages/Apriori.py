import os
import pandas as pd
import streamlit as st
from PIL import Image


class Apriori:
    def __init__(self):
        pass

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _plot_top_rules_lift(df_display: pd.DataFrame, n: int = 15):
        """Barras horizontales: Top N reglas ordenadas por Lift."""
        import matplotlib.pyplot as plt

        df_top = df_display.head(n).copy()
        df_top["regla"] = df_top["antecedents"] + "  →  " + df_top["consequents"]

        fig, ax = plt.subplots(figsize=(10, max(4, n * 0.45)))
        bars = ax.barh(df_top["regla"], df_top["lift"], color="steelblue", edgecolor="white")
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.set_xlabel("Lift")
        ax.set_title(f"Top {n} Reglas por Lift")
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    @staticmethod
    def _plot_heatmap(df_display: pd.DataFrame, max_items: int = 15):
        """Heatmap antecedente vs consecuente con confianza como color."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        df_h = df_display.head(max_items * 2).copy()
        pivot = df_h.pivot_table(
            index="antecedents",
            columns="consequents",
            values="confidence",
            aggfunc="max",
        )

        if pivot.empty:
            st.info("No hay suficientes reglas para construir el heatmap.")
            return

        fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2),
                                        max(4, len(pivot.index) * 0.6)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Confianza"},
            ax=ax,
        )
        ax.set_title("Confianza por par Antecedente → Consecuente")
        ax.set_xlabel("Consecuente")
        ax.set_ylabel("Antecedente")
        plt.xticks(rotation=35, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    @staticmethod
    def _rules_to_display(rules: pd.DataFrame) -> pd.DataFrame:
        """Convierte frozensets a strings y redondea métricas."""
        if rules.empty:
            return rules
        df = rules.copy()
        df["antecedents"] = df["antecedents"].apply(lambda x: ", ".join(sorted(x)))
        df["consequents"] = df["consequents"].apply(lambda x: ", ".join(sorted(x)))
        for col in ["support", "confidence", "lift"]:
            if col in df.columns:
                df[col] = df[col].round(4)
        return df[["antecedents", "consequents", "support", "confidence", "lift"]]

    @staticmethod
    def _show_image(path: str, caption: str):
        if os.path.exists(path):
            st.image(Image.open(path), caption=caption, use_container_width=True)
        else:
            st.info(f"Imagen no disponible: `{path}`")

    # ── render principal ─────────────────────────────────────────────────
    def render(self):
        st.title("📘 Resultados — Apriori")

        results: dict = st.session_state.get("pipeline_results", {})
        output_dir: str = st.session_state.get("pipeline_output_dir", "outputs")

        if not results:
            st.warning("⚠️ No hay resultados disponibles. Genera el análisis primero en la sección **Datos**.")
            return

        # ── filtrar columnas con resultados apriori ───────────────────────
        cols_con_apriori = [
            col for col, algos in results.items() if "apriori" in algos
        ]
        if not cols_con_apriori:
            st.error("No se encontraron resultados de Apriori.")
            return

        # ── selector de columna ───────────────────────────────────────────
        col_sel = st.selectbox(
            "🗂️ Columna de transacción",
            options=cols_con_apriori,
            key="apriori_col_sel",
        )

        rec = results[col_sel]["apriori"]
        rules_raw: pd.DataFrame = rec.rules if hasattr(rec, "rules") else pd.DataFrame()

        # ── métricas rápidas ──────────────────────────────────────────────
        m1, m2, m3 = st.columns(3)
        m1.metric("Reglas generadas", len(rules_raw))
        if not rules_raw.empty:
            m2.metric("Confianza promedio", f"{rules_raw['confidence'].mean():.3f}")
            m3.metric("Lift promedio", f"{rules_raw['lift'].mean():.3f}")

        st.divider()

        # ── tabla de reglas ───────────────────────────────────────────────
        st.subheader("📋 Reglas de Asociación")

        if rules_raw.empty:
            st.warning("Sin reglas generadas para esta columna. Prueba reduciendo el soporte mínimo.")
        else:
            df_display = self._rules_to_display(rules_raw.sort_values("lift", ascending=False))

            # filtro interactivo
            with st.expander("🔍 Filtrar reglas", expanded=False):
                fc1, fc2, fc3 = st.columns(3)
                min_sup  = fc1.slider("Soporte mínimo",   0.0, 1.0, float(df_display["support"].min()),   0.01, key="ap_fsup")
                min_conf = fc2.slider("Confianza mínima",  0.0, 1.0, float(df_display["confidence"].min()), 0.01, key="ap_fconf")
                min_lift = fc3.slider("Lift mínimo",       1.0, float(df_display["lift"].max()) + 0.1,
                                      float(df_display["lift"].min()), 0.1, key="ap_flift")

            mask = (
                (df_display["support"]    >= min_sup) &
                (df_display["confidence"] >= min_conf) &
                (df_display["lift"]       >= min_lift)
            )
            df_filtrado = df_display[mask]
            st.markdown(f"**{len(df_filtrado)} reglas** cumplen los filtros")
            st.dataframe(df_filtrado, use_container_width=True, height=350)

        st.divider()

        # ── visualizaciones ───────────────────────────────────────────────
        st.subheader("📊 Visualizaciones")
        col_dir = os.path.join(output_dir, col_sel.replace(" ", "_"), "apriori")

        v1, v2 = st.columns(2)
        with v1:
            self._show_image(
                os.path.join(col_dir, "scatter_confidence_lift.png"),
                "Confianza vs Lift",
            )
        with v2:
            self._show_image(
                os.path.join(col_dir, "network_rules.png"),
                "Grafo de Reglas",
            )


# ── gráficos adicionales ──────────────────────────────────────────
        if not rules_raw.empty:
            st.divider()
            st.subheader("📊 Análisis adicional")

            n_top = st.slider(
                "Número de reglas a mostrar",
                min_value=5, max_value=min(30, len(df_display)),
                value=min(15, len(df_display)),
                step=1,
                key="top_n_apriori",
                )

            tab1, tab2 = st.tabs(["📊 Top reglas por Lift", "🟦 Heatmap Confianza"])
            with tab1:
                self._plot_top_rules_lift(df_display, n=n_top)

            with tab2:
                self._plot_heatmap(df_display, max_items=n_top)