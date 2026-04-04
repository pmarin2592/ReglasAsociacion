"""
----------------
Clase AprioriModel: encapsula el algoritmo Apriori para minería de
reglas de asociación usando mlxtend.
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

warnings.filterwarnings("ignore")


class AprioriModel:
    """
    Aplica el algoritmo Apriori y genera reglas de asociación.

    Parámetros
    ----------
    transactions : list[list[str]]
        Lista de transacciones (cada transacción es una lista de ítems).
    min_support : float
        Soporte mínimo (0-1).
    min_confidence : float
        Confianza mínima (0-1).
    output_dir : str
        Carpeta para guardar gráficas.
    """

    def __init__(self,
                 transactions: list[list[str]],
                 min_support: float = 0.3,
                 min_confidence: float = 0.5,
                 output_dir: str = "outputs/apriori"):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.df_encoded: pd.DataFrame | None = None
        self.frequent_itemsets: pd.DataFrame | None = None
        self.rules: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode(self) -> pd.DataFrame:
        """Convierte transacciones a matriz one-hot."""
        te = TransactionEncoder()
        te_array = te.fit(self.transactions).transform(self.transactions)
        df = pd.DataFrame(te_array, columns=te.columns_)
        self.df_encoded = df.replace(False, 0)
        return self.df_encoded

    # ------------------------------------------------------------------
    # Minería
    # ------------------------------------------------------------------
    def fit(self) -> "AprioriModel":
        """Ejecuta Apriori y genera reglas de asociación."""
        print(f"\n[Apriori] Ejecutando con soporte={self.min_support}, "
              f"confianza={self.min_confidence} ...")

        self._encode()

        self.frequent_itemsets = apriori(
            self.df_encoded,
            min_support=self.min_support,
            use_colnames=True,
            verbose=0,
        )

        if self.frequent_itemsets.empty:
            print("[Apriori] ⚠ No se encontraron itemsets frecuentes. "
                  "Pruebe con un soporte más bajo.")
            self.rules = pd.DataFrame()
            return self

        self.rules = association_rules(
            self.frequent_itemsets,
            metric="confidence",
            min_threshold=self.min_confidence,
        )

        print(f"[Apriori] Itemsets frecuentes: {len(self.frequent_itemsets)}")
        print(f"[Apriori] Reglas generadas  : {len(self.rules)}")
        return self

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------
    def get_rules(self) -> pd.DataFrame:
        """Devuelve el DataFrame de reglas ordenado por lift."""
        if self.rules is None:
            raise RuntimeError("Ejecute .fit() primero.")
        if self.rules.empty:
            return self.rules
        return self.rules.sort_values("lift", ascending=False).reset_index(drop=True)

    def get_itemsets(self) -> pd.DataFrame:
        """Devuelve los itemsets frecuentes."""
        if self.frequent_itemsets is None:
            raise RuntimeError("Ejecute .fit() primero.")
        return self.frequent_itemsets.sort_values("support", ascending=False)

    def print_top_rules(self, n: int = 10) -> None:
        """Imprime las n mejores reglas."""
        rules = self.get_rules()
        if rules.empty:
            print("[Apriori] Sin reglas para mostrar.")
            return
        print(f"\n--- TOP {n} REGLAS APRIORI ---")
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print(rules[cols].head(n).to_string(index=False))

    # ------------------------------------------------------------------
    # Visualizaciones
    # ------------------------------------------------------------------
    def plot_scatter(self, save: bool = True) -> None:
        """Scatter plot de Confianza vs Lift."""
        rules = self.get_rules()
        if rules.empty:
            return
        plt.figure(figsize=(8, 6))
        plt.scatter(rules["confidence"], rules["lift"], alpha=0.7, color="steelblue")
        plt.xlabel("Confianza")
        plt.ylabel("Lift")
        plt.title("Apriori – Confianza vs Lift")
        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "scatter_confidence_lift.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            print(f"[Apriori] Gráfica guardada: {path}")
        plt.show()
        plt.close()

    def plot_network(self, max_rules: int = 30, save: bool = True) -> None:
        """Grafo de reglas de asociación."""
        rules = self.get_rules().head(max_rules)
        if rules.empty:
            return

        G = nx.DiGraph()
        for _, row in rules.iterrows():
            ant = ", ".join(sorted(row["antecedents"]))
            con = ", ".join(sorted(row["consequents"]))
            G.add_edge(ant, con, weight=row["confidence"])

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, seed=42, k=2)
        weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_color="lightblue",
                               node_size=1500, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6,
                               edge_color="gray", arrows=True,
                               arrowsize=20)
        plt.title("Apriori – Grafo de Reglas de Asociación")
        plt.axis("off")
        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "network_rules.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            print(f"[Apriori] Gráfica guardada: {path}")
        plt.show()
        plt.close()
