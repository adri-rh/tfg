# 🚗 Estimación de la intención de los usuarios de la vía mediante redes neuronales basadas en grafos

**Trabajo Fin de Grado – Ingeniería de Computadores**  
**Universidad de Alcalá**

| Rol | Nombre |
| :--- | :--- |
| **Autor** | Adrián Rodríguez Hurtado |
| **Tutor** | Augusto Luis Ballardini |
| **Cotutor** | Angie Nataly Melo Castillo |

---

## 📌 Descripción del proyecto

Este repositorio contiene la implementación completa del TFG centrado en la **predicción de la intención de usuarios de la vía** utilizando modelos **Graph Neural Networks (GNN)**.

El objetivo principal es modelar escenas viales complejas utilizando grafos para capturar:

* Relaciones espaciales entre agentes.
* Dependencias temporales.
* Interacciones multipeatón.
* Influencia contextual en la intención de cruce.

Se evalúan distintas arquitecturas GNN y configuraciones de diseño del grafo para analizar su impacto en rendimiento y robustez.

---

## 🧠 Motivación del proyecto

En los **Sistemas de Conducción Automatizada (SCA)**, predecir correctamente la intención de los peatones es crítico para la seguridad vial. La intención de cruce no depende únicamente de la posición, velocidad o trayectoria, sino también de:

* Interacción con vehículos.
* Contexto urbano.
* Evolución temporal.
* Comportamiento colectivo.

> **El problema:** Los enfoques tradicionales de Machine Learning modelan entidades de forma independiente. Sin embargo, los entornos viales son inherentemente relacionales.

Los grafos permiten representar explícitamente a peatones como nodos, interacciones como aristas y la evolución temporal como conexiones entre instantes. Sobre esta representación, las **GNN** permiten aprender patrones estructurales complejos.

---

## 🏗️ Arquitectura del proyecto

### 🔹 Representación mediante grafos

Se diseñaron distintas estructuras para representar la escena:

1.  **Grafos espaciales:** Capturan relaciones entre características en un mismo instante.
2.  **Grafos temporales:** Conexiones entre nodos a lo largo del tiempo.
3.  **Grafos espaciotemporales:** Combinan estructura espacial + evolución temporal.

Se implementaron tres variantes principales de conexión temporal:
* Conexión temporal básica.
* Conexión temporal completa.
* Sliding Window.

**Modelado multipeatón:** Se incorporan múltiples peatones en un mismo grafo para modelar comportamiento colectivo e interacciones sociales en una misma escena.

---

## 🤖 Modelos evaluados

Implementados con **PyTorch** y **PyTorch Geometric**.

### Arquitecturas utilizadas
* **GCN** (Graph Convolutional Networks): Cada nodo puede actualizar su representación combinando las características propias con la de los vecinos inmediatos.
* **GAT** (Graph Attention Networks): Aprende coeficientes de atención que permiten ponderar dinámicamente la contribución de cada nodo vecino durante la agregación.
* **GraphConv**: Permite separar explícitamente la contribución del nodo central de la agregación de información de los vecinos.

### Métricas de comparación
* Accuracy.
* Estabilidad frente a semillas aleatorias.
* Sensibilidad al diseño del grafo.
* Impacto del modelado temporal y multipeatón.

---

## 📊 Datasets utilizados

### 🔹 JAAD (Joint Attention in Autonomous Driving)
Dataset real de escenas urbanas con peatones. Se trabajó con una versión adaptada y preprocesada para el modelado en grafos. Se construyeron dos tipos de representaciones:
* Dataset numérico.
* Dataset lingüístico.

### 🔹 MUTAG
Dataset clásico de clasificación de grafos utilizado para la validación preliminar de las arquitecturas.

---

## 🔬 Pipeline experimental

1.  Construcción del grafo a partir de anotaciones.
2.  Conversión a formato `torch_geometric.data.Data`.
3.  División **Train / Validation / Test**.
4.  Entrenamiento con múltiples semillas.
5.  Seguimiento experimental con **Weights & Biases**.
6.  Comparativa estadística.

---

## 📈 Resultados observados

* El **diseño del grafo** impacta significativamente en el rendimiento.
* La incorporación de **modelado temporal** mejora la capacidad predictiva.
* El enfoque **multipeatón** permite capturar interacciones sociales relevantes.

---

## ⚙️ Tecnologías utilizadas

- 🐍 **Python** – Lenguaje principal del desarrollo.
- 🔥 **PyTorch** – Framework de Deep Learning.  
  https://pytorch.org/
- 📈 **PyTorch Geometric** – Librería especializada en Graph Neural Networks.  
  https://pytorch-geometric.readthedocs.io/
- 📊 **Weights & Biases** – Seguimiento experimental y control de métricas.
- ⚙️ **KNIME** – Herramienta utilizada para análisis y procesamiento complementario.  
  https://www.knime.com/
- 🐙 **GitHub** – Control de versiones y gestión del repositorio.

---

## 📚 Recursos de apoyo y referencia

Durante el desarrollo del proyecto se utilizaron los siguientes recursos formativos y técnicos:

- 🎥 **Antonio Longa – PyTorch Geometric Tutorial (YouTube Playlist)**  
  https://www.youtube.com/playlist?list=PLIFCDJ0AZD7ecKcrrC1m8Jb-0Bfkn0PdK

- 💻 **Repositorio oficial de tutoriales de Antonio Longa**  
  https://github.com/AntonioLonga/PytorchGeometricTutorial

- 📘 **UVA Deep Learning Course – GNN Overview Notebook**  
  https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html

---

## 📂 Estructura del repositorio (orientativa)

```text
├── src/
│   ├── JAAD.py                #Script principal del pipeline
│   ├── annotations.py         #Transformación visual de las anotaciones a los frames
├── tests/
│   ├── mutagtest.py           #Validación de GNN con el dataset MUTAG
│   ├── test_pyg.py            #Creación manual de un grafo simple. Pruebas de funciones básicas de PyG
│   ├── test_pyg2.py           #Uso del dataset TUDataset (ENZYMES) y división manual en Train/Test
│   ├── test_pyg3.py           #Uso del dataset Planetoid (Cora) y análisis de entrenamiento/validación/test
│   ├── test_pyg4.py           #Ejemplo de batching con DataLoader y agregación por grafo usando scatter
│   ├── test_pyg5.py           #Implementación y entrenamiento de una GCN básica sobre el dataset Cora
│   ├── traingat.py            #Pruebas del modelo GAT
│   ├── traingcn.py            #Pruebas del modelo GCN
│   └── traingraphconv.py      #Pruebas del modelo GraphConv
```

---

## 🚀 Cómo ejecutar el proyecto

### 1️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2️⃣ Ejecutar el archivo JAAD

```bash
python JAAD.py
```

## 📌 Contribuciones del trabajo

- Comparativa sistemática de arquitecturas GNN en un contexto vial  
- Análisis del impacto del diseño estructural del grafo  
- Estudio de robustez frente a semillas aleatorias
- Implementación de modelado multipeatón  
- Análisis crítico de limitaciones del dataset  

---

## 🔮 Líneas futuras

- **Enriquecimiento del contexto del dataset**: Incorporar nuevas variables relevantes como señales del entorno, velocidad de los vehículos, geometría y distribución de la vía. Una mayor cantidad de información contextual permitiría reducir ambigüedades y mejorar la capacidad predictiva del modelo.

- **Arquitecturas más expresivas para modelado multipeatón**: Explorar modelos más avanzados como *Graph Transformers* o arquitecturas híbridas GNN capaces de capturar interacciones sociales complejas y dependencias a largo plazo entre peatones.

- **Generalización a nuevos datasets y escenarios**: Aprovechar el diseño modular del pipeline para aplicarlo a otras fuentes de datos más allá del dataset JAAD, manteniendo la estructura general del sistema.

- **Consolidación dentro de una línea de investigación amplia**: Reforzar la reutilización y adaptación del pipeline en trabajos futuros relacionados con la estimación de intención de usuarios de la vía mediante modelos basados en grafos.


---

## 📜 Referencia académica

Rodríguez Hurtado, A. (2026). *Estimación de la intención de los usuarios de la vía mediante redes neuronales basadas en grafos*. Trabajo Fin de Grado, Universidad de Alcalá.
