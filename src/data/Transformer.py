"""
--------------
Clase Transformer: prepara columnas de tipo lista para minería de reglas.
"""

import ast
import pandas as pd


class Transformer:
    """
    Convierte columnas con valores de tipo lista (string separado por coma
    o literal de lista Python) en transacciones listas para Apriori/ECLAT.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame limpio.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ------------------------------------------------------------------
    # Parseo de columnas de lista
    # ------------------------------------------------------------------
    def parse_list_column(self, column: str,
                          separator: str = ",") -> "Transformer":
        """
        Convierte una columna de strings separados por `separator`
        (o literales de lista) en listas reales de Python.

        Parámetros
        ----------
        column : str
            Nombre de la columna a transformar.
        separator : str
            Separador usado dentro del string.
        """
        if column not in self.df.columns:
            raise ValueError(f"Columna '{column}' no encontrada en el DataFrame.")

        def _parse(value):
            if isinstance(value, list):
                return [str(v).strip() for v in value]
            if pd.isna(value):
                return []
            s = str(value).strip()
            # Si parece una lista Python, la evaluamos
            if s.startswith("["):
                try:
                    parsed = ast.literal_eval(s)
                    return [str(v).strip() for v in parsed]
                except Exception:
                    pass
            # Separamos por el delimitador
            return [v.strip() for v in s.split(separator) if v.strip()]

        self.df[column] = self.df[column].apply(_parse)
        print(f"[Transformer] Columna '{column}' convertida a listas.")
        return self

    # ------------------------------------------------------------------
    # Construcción de transacciones
    # ------------------------------------------------------------------
    def get_transactions(self, column: str) -> list[list[str]]:
        """
        Devuelve la lista de transacciones para una columna ya parseada.

        Parámetros
        ----------
        column : str
            Columna con listas de ítems.
        """
        if column not in self.df.columns:
            raise ValueError(f"Columna '{column}' no encontrada.")
        return self.df[column].tolist()

    # ------------------------------------------------------------------
    # Construcción de columna concatenada
    # ------------------------------------------------------------------
    def build_combined_column(self, columns: list[str],
                               new_col: str = "items",
                               separator: str = ",") -> "Transformer":
        """
        Concatena varias columnas categóricas en una sola columna de
        ítems separados por coma (útil cuando no hay una columna de lista
        predefinida).

        Parámetros
        ----------
        columns : list[str]
            Columnas a combinar.
        new_col : str
            Nombre de la nueva columna resultado.
        separator : str
            Separador entre ítems.
        """
        existing = [c for c in columns if c in self.df.columns]
        if not existing:
            raise ValueError("Ninguna de las columnas indicadas existe.")

        self.df[new_col] = self.df[existing].apply(
            lambda row: separator.join(str(v).strip() for v in row if pd.notna(v)),
            axis=1,
        )
        print(f"[Transformer] Columna combinada '{new_col}' creada desde: {existing}")
        return self

    # ------------------------------------------------------------------
    # Acceso
    # ------------------------------------------------------------------
    def get_dataframe(self) -> pd.DataFrame:
        """Devuelve el DataFrame transformado."""
        return self.df
