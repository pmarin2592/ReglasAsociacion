"""
--------------
Clase Recommender: sistema de recomendación basado en reglas de asociación.
"""

import pandas as pd


class Recommender:
    """
    Genera recomendaciones a partir de reglas de asociación (Apriori o ECLAT).

    Parámetros
    ----------
    rules : pd.DataFrame
        DataFrame con columnas: antecedents, consequents, support,
        confidence, lift.
    algorithm_name : str
        Nombre del algoritmo fuente (para etiquetas en los reportes).
    """

    def __init__(self, rules: pd.DataFrame, algorithm_name: str = ""):
        if rules.empty:
            self.rules = rules
        else:
            self.rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        self.algorithm_name = algorithm_name

    # ------------------------------------------------------------------
    # Recomendación por ítem
    # ------------------------------------------------------------------
    def recommend(self, items: list[str],
                  top_n: int = 5,
                  min_lift: float = 1.0) -> pd.DataFrame:
        """
        Dado un conjunto de ítems actuales, devuelve recomendaciones.

        Parámetros
        ----------
        items : list[str]
            Ítems que ya tiene / seleccionó el usuario.
        top_n : int
            Número máximo de recomendaciones.
        min_lift : float
            Lift mínimo para filtrar reglas.
        """
        if self.rules.empty:
            return pd.DataFrame(columns=["recommendation", "confidence", "lift"])

        items_set = set(items)
        recommendations = []

        for _, row in self.rules.iterrows():
            ant = set(row["antecedents"])
            if ant.issubset(items_set) and row["lift"] >= min_lift:
                for item in row["consequents"]:
                    if item not in items_set:
                        recommendations.append({
                            "recommendation": item,
                            "confidence": round(row["confidence"], 4),
                            "lift": round(row["lift"], 4),
                            "support": round(row["support"], 4),
                        })

        if not recommendations:
            return pd.DataFrame(columns=["recommendation", "confidence", "lift", "support"])

        df_rec = (
            pd.DataFrame(recommendations)
            .sort_values(["lift", "confidence"], ascending=False)
            .drop_duplicates(subset="recommendation")
            .head(top_n)
            .reset_index(drop=True)
        )
        return df_rec

    # ------------------------------------------------------------------
    # Propuestas de negocio
    # ------------------------------------------------------------------
    def business_insights(self, top_n: int = 10) -> list[str]:
        """
        Genera propuestas de negocio textuales a partir de las
        mejores reglas.

        Parámetros
        ----------
        top_n : int
            Número de reglas a analizar.
        """
        if self.rules.empty:
            return ["No se generaron reglas suficientes para propuestas de negocio."]

        insights = []
        algo = f"[{self.algorithm_name}] " if self.algorithm_name else ""
        rules_top = self.rules.head(top_n)

        for _, row in rules_top.iterrows():
            ant = ", ".join(sorted(row["antecedents"]))
            con = ", ".join(sorted(row["consequents"]))
            conf = row["confidence"]
            lift = row["lift"]
            sup = row["support"]

            if lift > 2:
                strength = "muy fuerte"
            elif lift > 1.5:
                strength = "fuerte"
            else:
                strength = "moderada"

            insight = (
                f"{algo}Los clientes que adquieren [{ant}] tienen una probabilidad "
                f"del {conf*100:.1f}% de también elegir [{con}] "
                f"(soporte={sup:.3f}, lift={lift:.2f} — asociación {strength}). "
                f"→ Se recomienda crear un paquete combinado o campaña cruzada."
            )
            insights.append(insight)

        return insights

    # ------------------------------------------------------------------
    # Reporte completo
    # ------------------------------------------------------------------
    def print_report(self, top_n_rules: int = 10,
                     top_n_insights: int = 5) -> None:
        """Imprime un reporte completo de reglas y propuestas."""
        algo = self.algorithm_name or "Reglas de Asociación"
        print(f"\n{'='*70}")
        print(f"  REPORTE DE RECOMENDACIONES – {algo.upper()}")
        print(f"{'='*70}")

        if self.rules.empty:
            print("  Sin reglas disponibles.")
            return

        print(f"\n  Total de reglas: {len(self.rules)}")
        print(f"\n  Métricas promedio:")
        print(f"    Soporte    : {self.rules['support'].mean():.4f}")
        print(f"    Confianza  : {self.rules['confidence'].mean():.4f}")
        print(f"    Lift       : {self.rules['lift'].mean():.4f}")

        print(f"\n  Top {top_n_rules} Reglas (por Lift):")
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print(self.rules[cols].head(top_n_rules).to_string(index=False))

        print(f"\n  Propuestas de negocio:")
        for i, ins in enumerate(self.business_insights(top_n_insights), 1):
            print(f"\n  {i}. {ins}")
