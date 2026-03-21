# Carga Librerías
import streamlit as st
import pandas as pd
import time
import hashlib
import io
from abc import ABC, abstractmethod
from typing import Optional

# Carga Clases
from src.utils.path import obtener_ruta_app


# ──────────────────────────────────────────────
# Clases de lectura de archivos
# ──────────────────────────────────────────────

class FileReader(ABC):
    """Clase base abstracta para lectores de archivo."""

    @abstractmethod
    def read(self, file, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        pass


class CSVReader(FileReader):
    """Lector para archivos CSV."""

    def read(self, file, **kwargs) -> pd.DataFrame:
        return pd.read_csv(file, **kwargs)

    def supported_extensions(self) -> list[str]:
        return [".csv"]


class ExcelReader(FileReader):
    """Lector para archivos Excel (.xlsx, .xls)."""

    def read(self, file, **kwargs) -> pd.DataFrame:
        xls = pd.ExcelFile(file)
        sheets = xls.sheet_names
        if len(sheets) > 1:
            selected = st.selectbox(
                "📋 Selecciona la hoja de Excel",
                options=sheets,
                key="sheet_selector"
            )
            return pd.read_excel(file, sheet_name=selected, **kwargs)
        return pd.read_excel(file, sheet_name=0, **kwargs)

    def supported_extensions(self) -> list[str]:
        return [".xlsx", ".xls"]


class FileReaderFactory:
    """Fábrica que retorna el lector correcto según la extensión del archivo."""

    _readers: list[FileReader] = [CSVReader(), ExcelReader()]

    @classmethod
    def get_reader(cls, filename: str) -> Optional[FileReader]:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
        for reader in cls._readers:
            if ext in reader.supported_extensions():
                return reader
        return None

    @classmethod
    def accepted_types(cls) -> list[str]:
        exts = []
        for r in cls._readers:
            exts.extend(r.supported_extensions())
        return exts


# ──────────────────────────────────────────────
# Clase de selección de columnas
# ──────────────────────────────────────────────

class ColumnSelector:
    """Gestiona la selección interactiva de columnas mediante combo box."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.all_columns = list(df.columns)

    def render(self) -> list[str]:
        st.markdown("### 🗂 Selecciona las columnas que deseas visualizar")

        if "selected_cols" not in st.session_state:
            st.session_state["selected_cols"] = []

        col1, col2 = st.columns([3, 1])
        with col1:
            available = [c for c in self.all_columns if c not in st.session_state["selected_cols"]]
            chosen = st.selectbox(
                "Columnas disponibles:",
                options=["— Elige una columna —"] + available,
                key="col_selectbox"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("➕ Agregar", use_container_width=True):
                if chosen != "— Elige una columna —" and chosen not in st.session_state["selected_cols"]:
                    st.session_state["selected_cols"].append(chosen)
                    st.rerun()

        if st.session_state["selected_cols"]:
            st.markdown("**Columnas seleccionadas:**")
            for col in list(st.session_state["selected_cols"]):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.markdown(f"- `{col}`")
                with c2:
                    if st.button("🗑", key=f"del_{col}"):
                        st.session_state["selected_cols"].remove(col)
                        st.rerun()

            if st.button("❌ Limpiar todas", use_container_width=True):
                st.session_state["selected_cols"] = []
                st.rerun()
        else:
            st.info("Aún no has agregado ninguna columna.")

        return st.session_state["selected_cols"]


# ──────────────────────────────────────────────
# Clase de visualización del DataFrame
# ──────────────────────────────────────────────

class DataFrameViewer:
    """Renderiza el DataFrame filtrado y sus estadísticas."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def render(self, selected_columns: list[str]):
        if not selected_columns:
            st.warning("⚠️ No has seleccionado ninguna columna.")
            return

        filtered = self.df[selected_columns]

        st.markdown(f"### 📊 Vista previa — {len(filtered):,} filas × {len(selected_columns)} columnas")
        st.dataframe(filtered, use_container_width=True, height=400)

        with st.expander("📈 Estadísticas descriptivas"):
            st.dataframe(filtered.describe(include="all"), use_container_width=True)

        self._download_button(filtered)

    def _download_button(self, df: pd.DataFrame):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Descargar selección como CSV",
            data=csv_buffer.getvalue(),
            file_name="datos_seleccionados.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ──────────────────────────────────────────────
# Clase principal de carga de datos
# ──────────────────────────────────────────────

class CargaDatos:
    def __init__(self):
        self.archivo_cargado = None
        self.analisis_realizado = False

    def _generar_hash_archivo(self, archivo_bytes):
        """Genera un hash MD5 del contenido del archivo para detectar cambios."""
        return hashlib.md5(archivo_bytes).hexdigest()

    def _obtener_tipo_dato_legible(self, tipo_pandas):
        """Convierte el tipo de dato de pandas a una descripción más legible."""
        tipo_str = str(tipo_pandas)
        if 'int' in tipo_str:
            return "🔢 Entero"
        elif 'float' in tipo_str:
            return "🔢 Decimal"
        elif 'bool' in tipo_str:
            return "☑️ Booleano"
        elif 'datetime' in tipo_str:
            return "📅 Fecha/Hora"
        elif 'object' in tipo_str:
            return "📝 Texto"
        elif 'category' in tipo_str:
            return "🏷️ Categoría"
        else:
            return f"❓ {tipo_str}"

    def _crear_tabla_con_tooltips(self, df):
        """Crea la tabla con tooltips en las columnas."""
        tooltips = {}
        for col in df.columns:
            tipo_dato = self._obtener_tipo_dato_legible(df[col].dtype)
            valores_nulos = df[col].isnull().sum()
            total_valores = len(df)
            porcentaje_nulos = (valores_nulos / total_valores) * 100
            tooltip_text = (
                f"{tipo_dato}\n"
                f"📊 {total_valores - valores_nulos} valores válidos\n"
                f"❌ {valores_nulos} valores nulos ({porcentaje_nulos:.1f}%)"
            )
            tooltips[col] = tooltip_text

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                col: st.column_config.Column(col, help=tooltips[col], width="medium")
                for col in df.columns
            }
        )

    def _leer_archivo(self, nombre: str, archivo_bytes: bytes, delimitador: str, decimal: str) -> pd.DataFrame:
        """Usa FileReaderFactory para leer el archivo según su extensión."""
        reader = FileReaderFactory.get_reader(nombre)
        if reader is None:
            raise ValueError(f"Formato de archivo no soportado: {nombre}")

        archivo_io = io.BytesIO(archivo_bytes)

        if nombre.lower().endswith(".csv"):
            return reader.read(archivo_io, delimiter=delimitador, decimal=decimal)
        else:
            return reader.read(archivo_io, decimal=decimal)

    def render(self):
        try:
            st.title("🗃️ Carga de Datos")

            accepted = FileReaderFactory.accepted_types()

            col1, col2 = st.columns([1, 2])
            with col1:
                delimitador = st.selectbox(
                    "Selecciona el delimitador del archivo:",
                    options=[",", ";", "\t"],
                    format_func=lambda x: f"{x}"
                )
                decimal = st.selectbox(
                    "Selecciona el decimal del archivo:",
                    options=[",", "."],
                    format_func=lambda x: f"{x}"
                )
            with col2:
                archivo = st.file_uploader(
                    "Sube tu archivo (.csv o .xlsx)",
                    type=[ext.lstrip(".") for ext in accepted],
                    key="uploader"
                )

            # Variables de estado previo
            prev_delimitador = st.session_state.get('delimitador')
            prev_decimal = st.session_state.get('decimal')
            prev_hash = st.session_state.get('file_hash')

            archivo_nuevo = False
            parametros_cambiados = False

            if archivo is not None:
                archivo_bytes = archivo.read()
                archivo.seek(0)

                archivo_hash = self._generar_hash_archivo(archivo_bytes)
                nombre = archivo.name

                if prev_hash != archivo_hash:
                    archivo_nuevo = True
                    st.session_state.archivo_bytes = archivo_bytes
                    st.session_state.file_name = nombre
                    st.session_state.file_hash = archivo_hash

            if prev_delimitador != delimitador or prev_decimal != decimal:
                parametros_cambiados = True

            if ('file_name' in st.session_state and
                    (archivo_nuevo or parametros_cambiados or 'df_cargado' not in st.session_state)):

                try:
                    nombre = st.session_state.file_name
                    archivo_bytes = st.session_state.archivo_bytes

                    df = self._leer_archivo(nombre, archivo_bytes, delimitador, decimal)

                    st.session_state.df_cargado = df
                    st.session_state.delimitador = delimitador
                    st.session_state.decimal = decimal
                    st.session_state.analisis_generado = False
                    st.session_state["selected_cols"] = []  # Resetear columnas al cargar nuevo archivo
                    if 'eda' in st.session_state:
                        del st.session_state.eda

                    if archivo_nuevo:
                        st.success(f"✅ Nuevo archivo '{nombre}' cargado exitosamente.")
                    elif parametros_cambiados:
                        st.success(f"✅ Datos actualizados con nuevos parámetros de lectura.")

                except Exception as e:
                    st.error(f"Error al procesar el archivo: {e}")
                    for key in ['df_cargado', 'eda', 'analisis_generado']:
                        if key in st.session_state:
                            del st.session_state[key]

            if 'df_cargado' in st.session_state:
                self.archivo_cargado = st.session_state.df_cargado

                st.subheader("Vista previa del dataset")

                df = st.session_state.df_cargado
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("📊 Filas", f"{len(df):,}")
                with col_stats2:
                    st.metric("📋 Columnas", len(df.columns))
                with col_stats3:
                    st.metric("📏 Tamaño", f"{df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

                self._crear_tabla_con_tooltips(df)

                st.divider()

                # Selección de columnas
                selector = ColumnSelector(df)
                selected_columns = selector.render()

                st.divider()

                # Visualización filtrada
                if selected_columns:
                    viewer = DataFrameViewer(df)
                    viewer.render(selected_columns)

                st.divider()

                col3, col4, col5 = st.columns([3, 2, 3])
                with col4:
                    if st.button("📊 Generar análisis"):
                        progreso = st.progress(0, text="Iniciando análisis...")
                        for i in range(1, 101):
                            time.sleep(0.02)
                            progreso.progress(i, text=f"Analizando datos... {i}%")

                        st.success("✅ Análisis completado exitosamente.")
                        self.analisis_realizado = True
                        st.session_state.analisis_generado = True
                        st.rerun()
            else:
                st.info("Por favor sube un archivo CSV o Excel (.xlsx).")

        except Exception as e:
            st.error(f"Ocurrió un error en la pantalla de datos: {e}")