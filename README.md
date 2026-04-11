# 📊 Aplicativo de Minería — Reglas de Asociación

Aplicación web interactiva construida con **Streamlit** para la minería de reglas de asociación usando los algoritmos **Apriori** y **ECLAT**. Diseñada para el análisis de datasets de laptops y turismo.

**Minería de Datos II – BD162 | Colegio Universitario de Cartago**  
**Grupo #3:** Pablo Marín Castillo · Alejandro Quesada Leiva

---

## 🗂️ Estructura del Proyecto

```
ReglasAsociacion/
│
├── app.py                          # Punto de entrada de la aplicación Streamlit
├── main.py                         # Entrada alternativa
├── logo-cuc.png                    # Logo institucional (sidebar)
├── README.md
│
├── data/
│   ├── raw/                        # Datasets originales sin procesar
│   │   ├── dataset_laptops.csv
│   │   └── tourism.csv
│   └── procesed/                   # Datos transformados/limpios
│       ├── datos_laptos.csv
│       └── datos_tourism.csv
│
├── notebooks/                      # Análisis exploratorios y prototipos
│   ├── EDA_Laptops.ipynb           # EDA del dataset de laptops
│   ├── EDA_Tourism.ipynb           # EDA del dataset de turismo
│   ├── Apriori_laptops.ipynb       # Prototipo Apriori sobre laptops
│   ├── Apriori_Tourism.ipynb       # Prototipo Apriori sobre turismo
│   ├── ECLAT_Laptops.ipynb         # Prototipo ECLAT sobre laptops
│   ├── ECLAT_Tourism.ipynb         # Prototipo ECLAT sobre turismo
│   └── Pipeline_Asociacion.ipynb   # Pipeline completo ejecutable como notebook
│
├── outputs/                        # Artefactos generados (gráficas y CSVs)
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── DataLoader.py           # Carga y limpieza de datos
│   │   ├── EDA.py                  # Análisis exploratorio automatizado
│   │   └── Transformer.py          # Transformación a transacciones
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── AprioriModel.py         # Algoritmo Apriori (mlxtend)
│   │   ├── ECLATModel.py           # Algoritmo ECLAT (implementación propia)
│   │   └── Recommender.py          # Sistema de recomendación y propuestas de negocio
│   │
│   ├── processed/
│   │   ├── __init__.py
│   │   ├── AssociationPipeline.py  # Pipeline completo de minería
│   │   └── Pipeline.py             # Orquestador por dataset
│   │
│   └── utils/
│       └── path.py                 # Utilidades de rutas relativas al proyecto
│
└── web/
    ├── asset/                      # Recursos estáticos web
    └── pages/
        ├── __init__.py
        ├── MenuPrincipal.py        # Navegación y layout del sidebar
        ├── CargaDatos.py           # Pantalla de carga y configuración
        ├── Apriori.py              # Pantalla de resultados Apriori
        └── ECLAT.py                # Pantalla de resultados ECLAT
```

---

## ⚙️ Requisitos

- Python 3.10+
- Las dependencias principales son:

```
streamlit
pandas
numpy
matplotlib
seaborn
networkx
mlxtend
Pillow
```

Instala todas las dependencias con:

```bash
pip install -r requirements.txt
```

---

## 🚀 Cómo ejecutar

Desde la raíz del proyecto:

```bash
streamlit run app.py
```

La aplicación abrirá automáticamente en el navegador en `http://localhost:8501`.

> **Nota:** Al iniciar por primera vez, se genera automáticamente un archivo `~/.streamlit/config.toml` con un límite de carga de 2000 MB.

---

## 📓 Notebooks

La carpeta `notebooks/` contiene los análisis exploratorios y prototipos que dieron origen a los módulos de producción. Son útiles para entender el razonamiento detrás de cada paso del pipeline.

| Notebook | Descripción |
|---|---|
| `EDA_Laptops.ipynb` | Análisis exploratorio del dataset de laptops: distribuciones, nulos, correlaciones y frecuencias de categorías. |
| `EDA_Tourism.ipynb` | Análisis exploratorio del dataset de turismo, incluyendo intereses y sitios visitados. |
| `Apriori_laptops.ipynb` | Prototipo del algoritmo Apriori aplicado sobre la columna `Sales Package` y columnas combinadas del dataset de laptops. |
| `Apriori_Tourism.ipynb` | Prototipo de Apriori sobre las columnas `Interests` y `Sites Visited` del dataset de turismo. |
| `ECLAT_Laptops.ipynb` | Implementación manual de ECLAT con TID-lists aplicada al dataset de laptops. |
| `ECLAT_Tourism.ipynb` | Implementación de ECLAT sobre las transacciones del dataset de turismo. |
| `Pipeline_Asociacion.ipynb` | Pipeline completo ejecutable como notebook. Reproduce el mismo flujo que la app Streamlit: carga → EDA → transformación → Apriori/ECLAT → recomendaciones. |

Para ejecutar los notebooks:

```bash
jupyter notebook notebooks/
```

Asegúrate de ejecutarlos desde la **raíz del proyecto** para que las importaciones de `src.*` funcionen correctamente. El `Pipeline_Asociacion.ipynb` agrega automáticamente la raíz al `sys.path`.

---

## 📋 Flujo de uso (Aplicación Web)

### 1. Carga de Datos
- Sube un archivo **CSV** o **Excel (.xlsx)**.
- Configura el delimitador y el separador decimal si es necesario.
- Previsualiza el dataset con métricas de calidad (filas, columnas, nulos, tamaño).
- Ajusta los parámetros del modelo:
  - **Soporte mínimo** — fracción mínima de transacciones que deben contener el ítemset.
  - **Confianza mínima** — probabilidad condicional mínima P(B|A).
  - **Lift mínimo** — fuerza de asociación (>1 indica asociación positiva).
- Haz clic en **"Generar análisis"** para ejecutar el pipeline.

### 2. Resultados Apriori
- Tabla de reglas de asociación con filtros interactivos (soporte, confianza, lift).
- Métricas resumen: total de reglas, confianza y lift promedio.
- Visualizaciones: scatter Confianza vs Lift, grafo de reglas, top reglas por lift y heatmap de confianza.

### 3. Resultados ECLAT
- Misma interfaz que Apriori, con resultados del algoritmo ECLAT.

> Las páginas **Apriori** y **ECLAT** permanecen bloqueadas hasta que se ejecute el análisis en la sección **Datos**.

---

## 🗃️ Datasets soportados

El pipeline detecta automáticamente el tipo de dataset por nombre de archivo:

| Nombre del archivo | Dataset | Columnas de transacción |
|---|---|---|
| Contiene `laptops` | Laptops | `Sales Package`, `items_combined` (Color, Type, Processor Brand, RAM, etc.) |
| Contiene `tourism` | Turismo | `Interests`, `Sites Visited` |

---

## 🏗️ Arquitectura del Pipeline

```
Archivo CSV/Excel
       │
       ▼
  DataLoader          → Carga y limpieza básica
       │
       ▼
     EDA              → Estadísticas, gráficas de distribución y correlación
       │
       ▼
  Transformer         → Parseo de listas y construcción de transacciones
       │
       ├──────────────────────────┐
       ▼                          ▼
 AprioriModel               ECLATModel
 (mlxtend)                  (TID-lists, implementación propia)
       │                          │
       ▼                          ▼
  Recommender             Recommender
       │                          │
       └──────────┬───────────────┘
                  ▼
          pipeline_results
          (guardado en session_state)
```

---

## 📦 Módulos principales

### `DataLoader`
Carga el DataFrame proporcionado y realiza limpieza básica: eliminación de duplicados, columnas con exceso de nulos y espacios en strings.

### `Transformer`
Convierte columnas de texto con valores separados por coma (o literales de lista Python) en listas reales de Python, aptas para los algoritmos de minería.

### `AprioriModel`
Encapsula el algoritmo Apriori usando `mlxtend`. Genera itemsets frecuentes y reglas de asociación, con visualizaciones de scatter y grafo de red.

### `ECLATModel`
Implementación propia de ECLAT usando representación vertical (TID-lists) e intersección de conjuntos. No depende de librerías externas de minería.

### `Recommender`
Dado un conjunto de ítems, busca reglas cuyo antecedente esté contenido en los ítems del usuario y devuelve los consecuentes como recomendaciones. También genera propuestas de negocio en texto natural.

### `AssociationPipeline`
Orquesta los cinco pasos del flujo: carga → EDA → transformación → modelos → exportación de reglas a CSV.

---

## 📁 Salidas generadas

Tras ejecutar el análisis, se crean los siguientes artefactos en la carpeta `outputs/`:

```
outputs/
└── <dataset>/
    ├── eda/
    │   ├── distributions.png
    │   ├── boxplots.png
    │   ├── correlation.png
    │   └── top_categories.png
    │
    └── <columna>/
        ├── apriori/
        │   ├── scatter_confidence_lift.png
        │   ├── network_rules.png
        │   └── rules_apriori.csv
        └── eclat/
            ├── scatter_confidence_lift.png
            ├── network_rules.png
            └── rules_eclat.csv
```

---

## 👥 Autores

Proyecto académico desarrollado para el curso **Minería de Datos II — Big Data CUC** (2026).

- **Pablo Marín Castillo**
- **Alejandro Quesada Leiva**
