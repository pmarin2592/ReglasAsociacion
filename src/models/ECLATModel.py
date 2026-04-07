"""
--------------
Clase ECLATModel: implementa el algoritmo ECLAT (Equivalence Class
Clustering and bottom-up Lattice Traversal) desde cero, usando
la representación vertical (TID-lists).
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
from itertools import combinations

warnings.filterwarnings("ignore")


class ECLATModel:
    """
    Aplica el algoritmo ECLAT para minería de reglas de asociación.

    Parámetros
    ----------
    transactions : list[list[str]]
        Lista de transacciones.
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
                 output_dir: str = "outputs/eclat"):
        self.transactions = transactions
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.n_transactions = len(transactions)
        self.frequent_itemsets: dict | None = None  # {frozenset: support_float}
        self.rules: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # ECLAT interno
    # ------------------------------------------------------------------
    def _build_vertical_db(self) -> dict:
        """Construye la base de datos vertical (ítem → conjunto de TIDs)."""
        vertical_db: dict[str, set] = defaultdict(set)
        for tid, transaction in enumerate(self.transactions):
            for item in transaction:
                vertical_db[item].add(tid)
        return dict(vertical_db)

    def _eclat_recursive(self, prefix: frozenset,
                         items: dict[str, set],
                         frequent: dict) -> None:
        """Recursión de ECLAT con TID-intersección."""
        items_list = list(items.items())
        for i, (item_i, tids_i) in enumerate(items_list):
            new_itemset = prefix | frozenset([item_i])
            support = len(tids_i) / self.n_transactions
            if support >= self.min_support:
                frequent[new_itemset] = support
                # Extender con los ítems restantes
                suffix: dict[str, set] = {}
                for item_j, tids_j in items_list[i + 1:]:
                    intersection = tids_i & tids_j
                    if len(intersection) / self.n_transactions >= self.min_support:
                        suffix[item_j] = intersection
                if suffix:
                    self._eclat_recursive(new_itemset, suffix, frequent)

    def _run_eclat(self) -> dict:
        """Ejecuta ECLAT y devuelve {frozenset: support}."""
        vertical_db = self._build_vertical_db()
        # Filtrar ítems individuales frecuentes
        frequent_singles = {
            item: tids for item, tids in vertical_db.items()
            if len(tids) / self.n_transactions >= self.min_support
        }
        frequent: dict[frozenset, float] = {}
        self._eclat_recursive(frozenset(), frequent_singles, frequent)
        return frequent

    # ------------------------------------------------------------------
    # Generación de reglas
    # ------------------------------------------------------------------
    def _generate_rules(self) -> list[dict]:
        """Genera reglas de asociación a partir de los itemsets frecuentes."""
        rules = []
        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue
            items = list(itemset)
            # Generar todas las posibles reglas antecedente → consecuente
            for r in range(1, len(items)):
                for antecedent in combinations(items, r):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    ant_support = self.frequent_itemsets.get(antecedent)
                    if ant_support is None or ant_support == 0:
                        continue
                    confidence = support / ant_support
                    if confidence >= self.min_confidence:
                        con_support = self.frequent_itemsets.get(consequent, 0)
                        lift = confidence / con_support if con_support > 0 else 0
                        rules.append({
                            "antecedents": antecedent,
                            "consequents": consequent,
                            "support": round(support, 4),
                            "confidence": round(confidence, 4),
                            "lift": round(lift, 4),
                        })
        return rules

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self) -> "ECLATModel":
        """Ejecuta ECLAT y genera reglas."""
        print(f"\n[ECLAT] Ejecutando con soporte={self.min_support}, "
              f"confianza={self.min_confidence} ...")

        self.frequent_itemsets = self._run_eclat()

        if not self.frequent_itemsets:
            print("[ECLAT] ⚠ No se encontraron itemsets frecuentes. "
                  "Pruebe con un soporte más bajo.")
            self.rules = pd.DataFrame()
            return self

        raw_rules = self._generate_rules()
        self.rules = pd.DataFrame(raw_rules) if raw_rules else pd.DataFrame()

        print(f"[ECLAT] Itemsets frecuentes: {len(self.frequent_itemsets)}")
        print(f"[ECLAT] Reglas generadas  : {len(self.rules)}")
        return self

    # ------------------------------------------------------------------
    # Resultados
    # ------------------------------------------------------------------
    def get_rules(self) -> pd.DataFrame:
        """Devuelve las reglas ordenadas por lift."""
        if self.rules is None:
            raise RuntimeError("Ejecute .fit() primero.")
        if self.rules.empty:
            return self.rules
        return self.rules.sort_values("lift", ascending=False).reset_index(drop=True)

    def get_itemsets_df(self) -> pd.DataFrame:
        """Devuelve los itemsets frecuentes como DataFrame."""
        if self.frequent_itemsets is None:
            raise RuntimeError("Ejecute .fit() primero.")
        rows = [{"itemset": set(k), "support": v}
                for k, v in self.frequent_itemsets.items()]
        return pd.DataFrame(rows).sort_values("support", ascending=False)

    def print_top_rules(self, n: int = 10) -> None:
        """Imprime las n mejores reglas."""
        rules = self.get_rules()
        if rules.empty:
            print("[ECLAT] Sin reglas para mostrar.")
            return
        print(f"\n--- TOP {n} REGLAS ECLAT ---")
        cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        print(rules[cols].head(n).to_string(index=False))

    # ------------------------------------------------------------------
    # Visualizaciones
    # ------------------------------------------------------------------
    def plot_scatter(self, save: bool = True) -> None:
        """Scatter plot Confianza vs Lift."""
        rules = self.get_rules()
        if rules.empty:
            return
        plt.figure(figsize=(8, 6))
        plt.scatter(rules["confidence"], rules["lift"],
                    alpha=0.6, c="darkorange", edgecolors="k", s=60)
        plt.xlabel("Confianza")
        plt.ylabel("Lift")
        plt.title("ECLAT – Confianza vs Lift")
        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "scatter_confidence_lift.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            print(f"[ECLAT] Gráfica guardada: {path}")
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
        nx.draw_networkx_nodes(G, pos, node_color="lightsalmon",
                               node_size=1500, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=8)
        nx.draw_networkx_edges(G, pos, width=weights, alpha=0.6,
                               edge_color="gray", arrows=True,
                               arrowsize=20)
        plt.title("ECLAT – Grafo de Reglas de Asociación")
        plt.axis("off")
        plt.tight_layout()
        if save:
            path = os.path.join(self.output_dir, "network_rules.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            print(f"[ECLAT] Gráfica guardada: {path}")
        plt.show()
        plt.close()
