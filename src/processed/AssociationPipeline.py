"""
pipeline.py
-----------
Clase AssociationPipeline: orquesta todo el flujo de minería de datos
desde la carga del CSV hasta las reglas de asociación y recomendaciones.

Uso básico
----------
>>> from src.processed.AssociationPipeline import AssociationPipeline
>>>
>>> pipeline = AssociationPipeline(
...     csv_path="data/raw/mi_dataset.csv",
...     transaction_columns=["columna_con_listas"],   # columnas tipo lista
...     output_dir="outputs",
... )
>>> pipeline.run()
"""

import os
import json
import pandas as pd
from narwhals import DataFrame

from src.data.DataLoader import DataLoader
from src.data.EDA import EDA
from src.data.Transformer import Transformer
from src.models.AprioriModel import AprioriModel
from src.models.ECLATModel import ECLATModel
from src.models.Recommender import Recommender


class AssociationPipeline:
    """
    Pipeline automatizado de minería de reglas de asociación.

    Parámetros
    ----------
    csv_path : str
        Ruta al CSV de entrada.
    transaction_columns : list[str] | None
        Columnas que ya contienen listas de ítems (separadas por coma o
        en formato literal de lista Python). Opcional si se usa combined_columns.
    columns_to_keep : list[str] | None
        Columnas que se conservan tras la limpieza.  None = todas.
    combined_columns : list[str] | None
        Columnas categóricas a concatenar en una sola columna de transacción
        llamada "items_combined". Se puede usar en lugar de transaction_columns
        o junto a él.
    min_support : float
        Soporte mínimo para Apriori y ECLAT.
    min_confidence : float
        Confianza mínima para las reglas.
    output_dir : str
        Directorio raíz donde se guardan todos los artefactos.
    run_apriori : bool
        Ejecutar Apriori.
    run_eclat : bool
        Ejecutar ECLAT.
    delimiter : str
        Separador del CSV.
    decimal : str
        Carácter decimal del CSV.
    encoding : str
        Codificación del CSV.
    cat_columns_to_plot : list[str] | None
        Columnas categóricas para graficar en el EDA.
    separator : str
        Separador usado dentro de las celdas de tipo lista.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transaction_columns: list[str] | None = None,
        columns_to_keep: list[str] | None = None,
        combined_columns: list[str] | None = None,
        min_support: float = 0.3,
        min_confidence: float = 0.5,
        output_dir: str = "outputs",
        run_apriori: bool = True,
        run_eclat: bool = True,
        delimiter: str = ",",
        decimal: str = ".",
        encoding: str = "utf-8",
        cat_columns_to_plot: list[str] | None = None,
        separator: str = ",",
        project_root_name: str = "ReglasAsociacion",
    ):
        self.df = df
        self.transaction_columns = transaction_columns or []
        self.columns_to_keep = columns_to_keep
        self.combined_columns = combined_columns
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.output_dir = output_dir
        self.run_apriori = run_apriori
        self.run_eclat = run_eclat
        self.delimiter = delimiter
        self.decimal = decimal
        self.encoding = encoding
        self.cat_columns_to_plot = cat_columns_to_plot
        self.separator = separator
        self.project_root_name = project_root_name

        os.makedirs(output_dir, exist_ok=True)

        # Estado interno
        self.df_clean: pd.DataFrame | None = None
        self.df_transformed: pd.DataFrame | None = None
        self.results: dict = {}   # columna → {apriori: Recommender, eclat: Recommender}

    # ------------------------------------------------------------------
    # Paso 1 – Carga y limpieza
    # ------------------------------------------------------------------
    def _step_load(self) -> pd.DataFrame:
        print("\n" + "▶" * 3 + " PASO 1: CARGA Y LIMPIEZA " + "◀" * 3)
        loader = DataLoader(
            df=self.df.copy(),
            project_root_name=self.project_root_name,
        )
        loader.load().clean(columns_to_keep=self.columns_to_keep)
        return loader.get_dataframe()

    # ------------------------------------------------------------------
    # Paso 2 – EDA
    # ------------------------------------------------------------------
    def _step_eda(self, df: pd.DataFrame) -> None:
        print("\n" + "▶" * 3 + " PASO 2: ANÁLISIS EXPLORATORIO (EDA) " + "◀" * 3)
        eda = EDA(df, output_dir=os.path.join(self.output_dir, "eda"))
        eda.run_full(cat_columns_to_plot=self.cat_columns_to_plot)

    # ------------------------------------------------------------------
    # Paso 3 – Transformación
    # ------------------------------------------------------------------
    def _step_transform(self, df: pd.DataFrame) -> Transformer:
        print("\n" + "▶" * 3 + " PASO 3: TRANSFORMACIÓN " + "◀" * 3)
        transformer = Transformer(df)

        # Parsear columnas de listas
        for col in self.transaction_columns:
            if col in df.columns:
                transformer.parse_list_column(col, separator=self.separator)

        # Construir columna combinada (opcional)
        if self.combined_columns:
            transformer.build_combined_column(
                self.combined_columns,
                new_col="items_combined",
                separator=self.separator,
            )
            transformer.parse_list_column("items_combined", separator=self.separator)
            if "items_combined" not in self.transaction_columns:
                self.transaction_columns.append("items_combined")

        return transformer

    # ------------------------------------------------------------------
    # Paso 4 – Modelos
    # ------------------------------------------------------------------
    def _step_models(self, transformer: Transformer) -> None:
        print("\n" + "▶" * 3 + " PASO 4: MODELOS DE ASOCIACIÓN " + "◀" * 3)

        for col in self.transaction_columns:
            if col not in transformer.get_dataframe().columns:
                print(f"[Pipeline] ⚠ Columna '{col}' no encontrada, se omite.")
                continue

            transactions = transformer.get_transactions(col)
            # Filtrar transacciones vacías
            transactions = [t for t in transactions if t]

            if not transactions:
                print(f"[Pipeline] ⚠ Sin transacciones en columna '{col}'.")
                continue

            print(f"\n  ── Columna: '{col}' ({len(transactions)} transacciones) ──")
            self.results[col] = {}

            col_dir = os.path.join(self.output_dir, col.replace(" ", "_"))

            # --- Apriori ---
            if self.run_apriori:
                apriori_dir = os.path.join(col_dir, "apriori")
                apriori_model = AprioriModel(
                    transactions=transactions,
                    min_support=self.min_support,
                    min_confidence=self.min_confidence,
                    output_dir=apriori_dir,
                )
                apriori_model.fit()
                apriori_model.print_top_rules()
                apriori_model.plot_scatter()
                apriori_model.plot_network()

                rec_a = Recommender(apriori_model.get_rules(), "Apriori")
                #rec_a.print_report()
                self.results[col]["apriori"] = rec_a

            # --- ECLAT ---
            if self.run_eclat:
                eclat_dir = os.path.join(col_dir, "eclat")
                eclat_model = ECLATModel(
                    transactions=transactions,
                    min_support=self.min_support,
                    min_confidence=self.min_confidence,
                    output_dir=eclat_dir,
                )
                eclat_model.fit()
                eclat_model.print_top_rules()
                eclat_model.plot_scatter()
                eclat_model.plot_network()

                rec_e = Recommender(eclat_model.get_rules(), "ECLAT")
                #rec_e.print_report()
                self.results[col]["eclat"] = rec_e

    # ------------------------------------------------------------------
    # Paso 5 – Exportar reglas a CSV
    # ------------------------------------------------------------------
    def _step_export(self) -> None:
        print("\n" + "▶" * 3 + " PASO 5: EXPORTACIÓN DE RESULTADOS " + "◀" * 3)
        for col, models in self.results.items():
            col_dir = os.path.join(self.output_dir, col.replace(" ", "_"))
            os.makedirs(col_dir, exist_ok=True)
            for algo, rec in models.items():
                if not rec.rules.empty:
                    path = os.path.join(col_dir, f"rules_{algo}.csv")
                    # Convertir frozensets a strings para CSV
                    df_export = rec.rules.copy()
                    df_export["antecedents"] = df_export["antecedents"].apply(
                        lambda x: ", ".join(sorted(x)))
                    df_export["consequents"] = df_export["consequents"].apply(
                        lambda x: ", ".join(sorted(x)))
                    df_export.to_csv(path, index=False)
                    print(f"[Pipeline] Reglas exportadas: {path}")

    # ------------------------------------------------------------------
    # Ejecución completa
    # ------------------------------------------------------------------
    def run(self, run_eda: bool = True) -> dict:
        """
        Ejecuta el pipeline completo.

        Parámetros
        ----------
        run_eda : bool
            Si True, ejecuta el EDA (Paso 2).

        Retorna
        -------
        dict : {columna: {algoritmo: Recommender}}
        """
        print("\n" + "=" * 70)
        print("  PIPELINE DE MINERÍA DE REGLAS DE ASOCIACIÓN")
        print("=" * 70)

        # 1. Carga
        self.df_clean = self._step_load()

        # 2. EDA
        if run_eda:
            self._step_eda(self.df_clean)

        # 3. Transformación
        transformer = self._step_transform(self.df_clean)
        self.df_transformed = transformer.get_dataframe()

        # 4. Modelos
        self._step_models(transformer)

        # 5. Exportar
        self._step_export()

        print("\n" + "=" * 70)
        print("  PIPELINE FINALIZADO ✓")
        print("=" * 70)
        return self.results

    # ------------------------------------------------------------------
    # Recomendación ad-hoc
    # ------------------------------------------------------------------
    def get_recommendations(self, column: str,
                             items: list[str],
                             algorithm: str = "apriori",
                             top_n: int = 5) -> pd.DataFrame:
        """
        Obtiene recomendaciones para un conjunto de ítems dados.

        Parámetros
        ----------
        column : str
            Columna de transacción usada.
        items : list[str]
            Ítems actuales del usuario.
        algorithm : str
            'apriori' o 'eclat'.
        top_n : int
            Número de recomendaciones.
        """
        if column not in self.results:
            raise ValueError(f"Columna '{column}' no procesada. Ejecute .run() primero.")
        if algorithm not in self.results[column]:
            raise ValueError(f"Algoritmo '{algorithm}' no disponible para '{column}'.")
        rec: Recommender = self.results[column][algorithm]
        return rec.recommend(items, top_n=top_n)
