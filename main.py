"""
main.py
-------
Punto de entrada del pipeline.
Demuestra el uso con los dos datasets del proyecto (Laptops y Turismo).
Puede adaptarse a cualquier CSV modificando los parámetros de configuración.

Ejecución
---------
    python main.py
"""

from src.processed.AssociationPipeline import AssociationPipeline


# ======================================================================
# CONFIGURACIÓN – LAPTOPS
# ======================================================================
def run_laptops():
    print("\n" + "#" * 70)
    print("#  DATASET: LAPTOPS")
    print("#" * 70)

    pipeline = AssociationPipeline(
        csv_path="data/raw/dataset_laptos.csv",

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
        min_support=0.3,
        min_confidence=0.5,

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
def run_tourism():
    print("\n" + "#" * 70)
    print("#  DATASET: TURISMO")
    print("#" * 70)

    pipeline = AssociationPipeline(
        csv_path="data/raw/tourism.csv",

        # El EDA del proyecto parsea estas columnas como listas
        transaction_columns=["Interests", "Sites Visited"],

        # Columnas relevantes
        columns_to_keep=[
            "Tourist ID", "Age", "Interests", "Sites Visited",
            "Preferred Tour Duration", "Tour Duration",
        ],

        # Parámetros del modelo (valores más bajos por dataset de turismo)
        min_support=0.2,
        min_confidence=0.2,

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


# ======================================================================
# CONFIGURACIÓN GENÉRICA – Cualquier CSV nuevo
# ======================================================================
def run_custom(
    csv_path: str,
    transaction_columns: list[str],
    columns_to_keep: list[str] | None = None,
    min_support: float = 0.3,
    min_confidence: float = 0.5,
    output_dir: str = "outputs/custom",
    run_apriori: bool = True,
    run_eclat: bool = True,
    cat_columns_to_plot: list[str] | None = None,
    separator: str = ",",
):
    """
    Función genérica para ejecutar el pipeline en cualquier CSV.

    Parámetros
    ----------
    csv_path : str
        Ruta al CSV.
    transaction_columns : list[str]
        Columnas con listas de ítems (separadas por `separator`).
    columns_to_keep : list[str] | None
        Columnas a conservar. None = todas.
    min_support : float
        Soporte mínimo.
    min_confidence : float
        Confianza mínima.
    output_dir : str
        Directorio de salida.
    run_apriori : bool
        ¿Ejecutar Apriori?
    run_eclat : bool
        ¿Ejecutar ECLAT?
    cat_columns_to_plot : list[str] | None
        Columnas categóricas para el EDA.
    separator : str
        Separador dentro de las celdas.
    """
    pipeline = AssociationPipeline(
        csv_path=csv_path,
        transaction_columns=transaction_columns,
        columns_to_keep=columns_to_keep,
        min_support=min_support,
        min_confidence=min_confidence,
        output_dir=output_dir,
        run_apriori=run_apriori,
        run_eclat=run_eclat,
        cat_columns_to_plot=cat_columns_to_plot,
        separator=separator,
    )
    return pipeline.run()


# ======================================================================
# ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    import sys

    # Por defecto ejecuta ambos datasets del proyecto.
    # Cambia las llamadas según el dataset que necesites.
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
        if dataset == "laptops":
            run_laptops()
        elif dataset == "tourism":
            run_tourism()
        else:
            print(f"Dataset '{dataset}' no reconocido. Use: laptops | tourism")
    else:
        # Ejecutar ambos
        run_laptops()
        run_tourism()
