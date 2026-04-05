import pandas as pd

from src.processed.AssociationPipeline import AssociationPipeline


class Pipeline:
    def __init__(self, df: pd.DataFrame, min_support: float, min_confidence: float):
        self.df = df
        self.min_support = min_support
        self.min_confidence = min_confidence

    def _run_laptops(self,df: pd.DataFrame):
        print("\n" + "#" * 70)
        print("#  DATASET: LAPTOPS")
        print("#" * 70)

        pipeline = AssociationPipeline(
            df=df,

            # Opción A: columnas que ya contienen listas de ítems
            # transaction_columns=["Sales Package", "Caracteristicas"],

            # Opción B: construir "Caracteristicas" combinando columnas sueltas
            # (con esto transaction_columns puede omitirse o dejarse vacío)
            combined_columns=[
                "Color", "Type", "Suitable For",
                "Processor Brand", "Processor Name",
                "RAM", "RAM Type", "Operating System"
            ],
            transaction_columns=["Sales Package"],  # puede omitirse si solo usas combined_columns

            # Columnas que se mantienen después de la limpieza
            columns_to_keep=[
                "name", "Sales Package", "Color", "Type", "Suitable For",
                "Processor Brand", "Processor Name", "RAM", "RAM Type",
                "Operating System",
            ],

            # Parámetros del modelo
            min_support=self.min_support,
            min_confidence=self.min_confidence,

            # Salida
            output_dir="outputs/laptops",
            run_apriori=True,
            run_eclat=True,

            # EDA: columnas categóricas a graficar
            cat_columns_to_plot=[
                "Processor Brand", "RAM Type", "Operating System", "Type"
            ],
        )

        results = pipeline.run()

        # Ejemplo de recomendación interactiva
        print("\n--- EJEMPLO DE RECOMENDACIÓN (Laptops) ---")
        try:
            recs = pipeline.get_recommendations(
                column="Sales Package",
                items=["Laptop Bag"],
                algorithm="apriori",
                top_n=5,
            )
            print("Dado que el cliente lleva: ['Laptop Bag']")
            print("Se recomienda también:")
            print(recs.to_string(index=False))
        except Exception as e:
            print(f"  (Sin recomendaciones disponibles: {e})")

        return results

    # ======================================================================
    # CONFIGURACIÓN – TURISMO
    # ======================================================================
    def _run_tourism(self,df: pd.DataFrame):
        print("\n" + "#" * 70)
        print("#  DATASET: TURISMO")
        print("#" * 70)

        pipeline = AssociationPipeline(
            df=df,

            # El EDA del proyecto parsea estas columnas como listas
            transaction_columns=["Interests", "Sites Visited"],

            # Columnas relevantes
            columns_to_keep=[
                "Tourist ID", "Age", "Interests", "Sites Visited",
                "Preferred Tour Duration", "Tour Duration",
            ],

            # Parámetros del modelo (valores más bajos por dataset de turismo)
            min_support=self.min_support,
            min_confidence=self.min_confidence,

            output_dir="outputs/tourism",
            run_apriori=True,
            run_eclat=True,

            cat_columns_to_plot=["Interests"],

            # Separador dentro de las celdas de lista
            separator=",",
        )

        results = pipeline.run()

        # Ejemplo de recomendación interactiva
        print("\n--- EJEMPLO DE RECOMENDACIÓN (Turismo) ---")
        try:
            recs = pipeline.get_recommendations(
                column="Interests",
                items=["Adventure"],
                algorithm="eclat",
                top_n=5,
            )
            print("Dado que el turista tiene interés en: ['Adventure']")
            print("Se recomienda también:")
            print(recs.to_string(index=False))
        except Exception as e:
            print(f"  (Sin recomendaciones disponibles: {e})")

        return results

    def execute(self, dataSet: str):
        # Ejecutar ambos
        if dataSet == "laptops":
            self._run_laptops(self.df)
        elif dataSet == "tourism":
            self._run_tourism(self.df)