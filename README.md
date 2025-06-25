# Trabajo Práctico

Integrantes:

- Sofía Javes
- Mariana Juarez Goldemberg
- María Sol Moon
- Macarena Vita Sánchez

## Cómo ejecutar

En la notebook (https://colab.research.google.com/drive/1fRu-xnUNaTSJipHjakoGuIdHGAlfz5tF?usp=sharing) se encuentra el desarrollo y análisis completo de cómo llegamos al modelo elegido.

Para ejecutar la app se necesita tener el paquete `streamlit` y el modelo ya descargados. Una vez que se tiene eso se ejecuta el siguiente comando para correr la app:

```bash
    streamlit run app.py
```

Para descargar el modelo se puede ejecutar la notebook o el archivo `modelo.py`. Ambos generan como salida el modelo ya entrenado y listo para utilizar. Para correr el archivo:

```bash
    python3 modelo.py ${path_dataset}
```

El dataset utilizado para entrenar el modelo es https://www.kaggle.com/datasets/payamamanat/imbd-dataset
