# 🌸 Iris Classifier Pro

## Universidad de la Costa - Data Mining 2024
**Desarrollado por:** Alexander Gutierrez

---

## 📋 Descripción del Proyecto

**Iris Classifier Pro** es un proyecto avanzado de clasificación de especies de iris que implementa un modelo de Machine Learning usando Random Forest optimizado con GridSearchCV. El proyecto incluye un dashboard interactivo en Streamlit con visualizaciones profesionales y análisis exhaustivo del dataset.

### 🎯 Características Principales

- ✅ **Modelo Avanzado**: Random Forest con optimización GridSearchCV
- ✅ **Pipeline Completo**: Preprocesamiento, entrenamiento y evaluación
- ✅ **Dashboard Interactivo**: 4 tabs con diferentes análisis
- ✅ **Visualizaciones Profesionales**: Plotly, Seaborn, Matplotlib
- ✅ **Análisis 3D**: PCA para reducción dimensional
- ✅ **Predicción en Tiempo Real**: Sliders interactivos
- ✅ **Métricas Completas**: Accuracy, Precision, Recall, F1-Score

---

## 🏗️ Arquitectura del Proyecto

```
/mnt/okcomputer/output/
├── project.py          # Código principal del proyecto
├── requirements.txt    # Dependencias de Python
├── README.md          # Documentación del proyecto
└── data/              # Dataset integrado (sklearn)
```

---

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- 2GB de RAM mínimo

### Instalación Paso a Paso

1. **Clonar o descargar el proyecto**
   ```bash
   # Si está en un repositorio
   git clone [URL_DEL_REPOSITORIO]
   cd iris-classifier-pro
   ```

2. **Crear entorno virtual (recomendado)**
   ```bash
   python -m venv venv
   
   # En Windows
   venv\Scripts\activate
   
   # En Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**
   ```bash
   streamlit run project.py
   ```

5. **Abrir en navegador**
   - La aplicación se abrirá automáticamente en `http://localhost:8501`
   - O accede manualmente a esa URL

---

## 📊 Características del Dashboard

### 🎯 Tab 1: Dashboard Principal
- **Métricas de Evaluación**: Accuracy, Precision, Recall, F1-Score
- **Barras de Progreso Coloridas**: Visualización intuitiva de rendimiento
- **Importancia de Características**: Gráfico de barras interactivo
- **Matriz de Confusión**: Heatmap interactivo con Plotly
- **Hiperparámetros Óptimos**: Visualización de mejores parámetros

### 🔬 Tab 2: Análisis Exploratorio
- **Estadísticas Descriptivas**: Tabla completa con métricas estadísticas
- **Histogramas por Clase**: Distribuciones de características por especie
- **Scatter Matrix (Pairplot)**: Relaciones entre pares de características
- **Violin Plots**: Distribuciones detalladas con densidad

### 🌍 Tab 3: Visualización 3D
- **PCA Explicado**: Reducción dimensional manteniendo 95% de varianza
- **Gráfico 3D Interactivo**: Visualización en tres dimensiones
- **Componentes Principales**: PC1, PC2, PC3 con varianza explicada
- **Análisis de Varianza**: Tabla detallada de contribución

### 🔮 Tab 4: Predicción
- **Sliders Interactivos**: Controles para ingresar características
- **Resultado con Emoji**: Visualización gráfica de la predicción
- **Probabilidades**: Gráfico de barras con confianza por especie
- **Visualización 3D**: Nueva muestra en el espacio reducido

---

## 🧠 Modelo de Machine Learning

### Algoritmo: Random Forest Classifier
- **Tipo**: Ensemble Learning (Bosque de Árboles de Decisión)
- **Optimización**: GridSearchCV con validación cruzada (5-fold)
- **Preprocesamiento**: StandardScaler para normalización
- **Hiperparámetros Optimizados**:
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [3, 5, 7, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]

### Métricas de Rendimiento
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión por clase (weighted average)
- **Recall**: Sensibilidad por clase (weighted average)
- **F1-Score**: Media armónica de precision y recall

---

## 📈 Dataset: Iris Flower Dataset

### Características
- **Muestras**: 150 flores (50 por especie)
- **Especies**: Setosa, Versicolor, Virginica
- **Características**:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

### Origen
- **Fuente**: sklearn.datasets.load_iris()
- **Atribución**: Ronald Fisher (1936)
- **Tipo**: Dataset multivariado de clasificación

---

## 🛠️ Tecnologías Utilizadas

### Framework Principal
- **Streamlit**: 1.28.2 - Framework web para aplicaciones ML

### Ciencia de Datos
- **Pandas**: 2.1.3 - Manipulación de datos
- **NumPy**: 1.24.3 - Computación numérica
- **Scikit-learn**: 1.3.2 - Machine Learning

### Visualizaciones
- **Matplotlib**: 3.7.2 - Gráficos básicos
- **Seaborn**: 0.12.2 - Visualizaciones estadísticas
- **Plotly**: 5.17.0 - Gráficos interactivos

### Optimización
- **Joblib**: 1.3.2 - Paralelización y caching

---

## 🎨 Características de UI/UX

### Diseño Responsivo
- **Layout**: Wide mode para máximo aprovechamiento de pantalla
- **Sidebar**: Controles de predicción siempre visibles
- **Tabs**: Navegación intuitiva por secciones

### Visualizaciones
- **Colores Consistentes**: Paleta cromática armoniosa
- **Interactividad**: Todos los gráficos son interactivos
- **Responsive**: Adaptación a diferentes tamaños de pantalla

### Performance
- **Caching**: Sistema de caché avanzado con @st.cache_data
- **Optimización**: GridSearchCV para mejor rendimiento
- **Lazy Loading**: Carga diferida de componentes pesados

---

## 📚 Documentación del Código

### Funciones Principales

#### `load_and_explore_data()`
- **Propósito**: Cargar y explorar el dataset Iris
- **Retorno**: Tupla (X, y, iris_data)
- **Cache**: TTL de 3600 segundos

#### `create_ml_pipeline()`
- **Propósito**: Crear pipeline ML con preprocesamiento
- **Retorno**: Pipeline y grid de hiperparámetros
- **Cache**: Recurso persistente

#### `train_and_evaluate_model()`
- **Propósito**: Entrenar y evaluar el modelo
- **Retorno**: Diccionario con métricas y modelo
- **Optimización**: GridSearchCV con 5-fold CV

#### `create_3d_visualization()`
- **Propósito**: Crear gráfico 3D con PCA
- **Retorno**: Figura de Plotly 3D
- **Característica**: Muestra nueva muestra opcional

---

## 🤝 Contribuciones

### Cómo Contribuir
1. Fork del proyecto
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit de cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

### Reporte de Bugs
- **Issues**: Reportar problemas en el repositorio
- **Email**: alexander.gutierrez@ucosta.edu.co

---

## 📄 Licencia

Este proyecto es desarrollado para **Universidad de la Costa - Data Mining 2024**.

- **Propósito**: Uso académico y educativo
- **Distribución**: Prohibida comercialización
- **Créditos**: Debe mantener atribución al autor

---

## 📞 Contacto

### Autor
- **Nombre**: Alexander Gutierrez
- **Email**: alexander.gutierrez@ucosta.edu.co
- **LinkedIn**: [Alexander Gutierrez](https://linkedin.com/in/alexander-gutierrez)
- **GitHub**: [alexgutierrez](https://github.com/alexgutierrez)

### Universidad
- **Institución**: Universidad de la Costa
- **Programa**: Ingeniería de Sistemas
- **Asignatura**: Data Mining
- **Año**: 2024

---

## 🙏 Agradecimientos

- **Universidad de la Costa** por la oportunidad académica
- **Profesor de Data Mining** por la guía y conocimientos
- **Ronald Fisher** por el dataset Iris clásico
- **Streamlit Team** por el framework excepcional
- **Scikit-learn Community** por las herramientas ML

---

<div align="center">
    <h3>🌸 ¡Gracias por usar Iris Classifier Pro! 🌸</h3>
    <p><em>"La mejor manera de predecir el futuro es crearlo"</em></p>
</div>