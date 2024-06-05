
# Embeddings aplicado a un buscador semántico

Este repositorio contiene el código y la implementación del proyecto de investigación "Embeddings aplicado a un buscador semántico", desarrollado en la Universidad Politécnica de Madrid por Luis Manobanda. El proyecto explora el uso de embeddings y arquitecturas avanzadas de procesamiento de lenguaje natural (NLP) para mejorar la precisión y relevancia de los resultados de búsqueda semántica.

## Descripción

Buscador semántico de películas implementado utilizando un modelo basado en BERT para la creación de los embeddings. Utiliza una base de datos vectorial Pinecone y una interfaz gráfica de [GRADIO](https://www.gradio.app/). Para utilizar la interfaz, se debe tener una cuenta en [Pinecone](https://www.pinecone.io/) (free) y agregar el API key.

## Contenido del Repositorio

- `src/`: Contiene el código fuente del buscador semántico.
- `data/`: Incluye el dataset utilizado, compuesto por 1000 películas del dataset open source IMDB.
- `notebooks/`: Jupyter notebooks con experimentos y análisis.

## Requisitos

Para instalar las dependencias necesarias, ejecute el siguiente comando:

```python
%%capture
!pip install -U sentence-transformers
!pip install pinecone-client
!pip install gradio
!pip install rank-bm25
```

## Uso

### Imports

```python
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from ast import literal_eval
import pinecone
from getpass import getpass
import gradio as gr
```

### Dataset

El dataset que se va a utilizar es una lista de películas de [IMDB](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows), que contiene las siguientes columnas:

- Poster\_Link: Enlace del póster de IMDb
- Series\_Title: Nombre de la película
- Released\_Year: Año de estreno
- Certificate: Certificado obtenido
- Runtime: Duración total
- Genre: Género
- IMDB\_Rating: Calificación en IMDb
- Overview: Mini historia/resumen
- Meta\_score: Puntuación obtenida
- Director: Nombre del director
- Star1, Star2, Star3, Star4: Nombres de las estrellas
- No\_of\_votes: Número total de votos
- Gross: Dinero recaudado

### Configuración de Pinecone

Para usar Pinecone, debe autenticarse con su API key:

```python
pinecone.init(api_key=getpass("Enter your Pinecone API key: "))
```

### Creación de Embeddings

Utilizando el framework `SentenceTransformers`, se crean los embeddings de las descripciones de las películas:

```python
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['Overview'].tolist(), convert_to_tensor=True)
```

### Implementación del Buscador Semántico

El buscador semántico utiliza los embeddings almacenados en Pinecone para realizar búsquedas eficientes y precisas:

```python
index = pinecone.Index("movie-search")

def search(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    results = index.query(query_embedding, top_k=top_k)
    return results
```

### Interfaz Gráfica con Gradio

Se implementa una interfaz gráfica simple usando Gradio para interactuar con el buscador semántico:

```python
def semantic_search(query):
    results = search(query)
    return results

iface = gr.Interface(fn=semantic_search, inputs="text", outputs="json")
iface.launch()
```

## Resultados

El buscador semántico permite obtener resultados más relevantes utilizando el contexto de las consultas. Se pueden observar ejemplos de las respuestas en el documento del proyecto.

## Conclusiones

- Los buscadores semánticos utilizan el contexto de la consulta para mejorar los resultados.
- Los embeddings permiten mantener el contexto de una oración, mejorando la precisión semántica.
- Sentence-BERT mejora la eficiencia de BERT utilizando medidas de similaridad.
- Las bases de datos vectoriales ofrecen ventajas en velocidad y manejo de embeddings para una rápida extracción y procesamiento.


## Enlaces Útiles

- [Repositorio GitHub](https://github.com/JaviMiot/buscador_semantico)
- [Documentación de SentenceTransformers](https://sbert.net/)
