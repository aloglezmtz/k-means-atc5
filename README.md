# Actividad 5 · K-Means con el conjunto de datos Iris

Aplicación de K-means a Iris con los **seis pares de características** y k = 2, 3, 4 y 5 (variables estandarizadas; la especie no se usa para entrenar). Incluye una página interactiva que muestra el algoritmo en tiempo real.

**Autora:** Alondra Goretti González Martínez (218534681) · **Profesor:** Emilio Barajas González

## Contenido

| Archivo / carpeta | Descripción |
|---|---|
| `index.html` | Página interactiva (se publica con GitHub Pages) |
| `kmeans_iris.py` | Análisis completo en Python (24 análisis: 6 pares × 4 valores de k) |
| `iris.csv` | Base de datos Iris (150 flores) |
| `requirements.txt` | Dependencias de Python |
| `resultados/` | `metricas.csv` (inercia, silueta, ARI), `centroides.csv`, `asignaciones.csv` |
| `figs/` | Gráficas de grupos, método del codo, silueta y matriz comparativa |
| `informe/` | Informe en PDF (formato APA 7) |
| `.github/workflows/kmeans.yml` | Ejecuta el script en GitHub Actions |
| `.nojekyll` | Indica a GitHub Pages que sirva los archivos tal cual |

## Ejecutar en tu computadora

```bash
pip install -r requirements.txt
python kmeans_iris.py
```

Se crean/actualizan las carpetas `resultados/` y `figs/`. Para ver la página interactiva basta abrir `index.html` en el navegador.

## Publicar en GitHub Pages (paso a paso)

1. En GitHub, crea un repositorio nuevo (por ejemplo `kmeans-iris`) como **Public**.
2. Pulsa **Add file → Upload files**, arrastra **todo el contenido** de esta carpeta (incluida `.github`) y confirma con **Commit changes**.
3. Ve a **Settings → Pages**. En *Build and deployment* elige **Deploy from a branch**, rama `main` y carpeta `/ (root)`, y guarda.
4. Espera uno o dos minutos. La página quedará en `https://TU-USUARIO.github.io/kmeans-iris/`.

Si usas Git en la terminal:

```bash
git init
git add .
git commit -m "Actividad 5 K-Means"
git branch -M main
git remote add origin https://github.com/TU-USUARIO/kmeans-iris.git
git push -u origin main
```

## Ejecutar Python en GitHub (Actions)

Cada vez que subas cambios, la pestaña **Actions** ejecuta `kmeans_iris.py` y guarda `resultados/` y `figs/` como artefacto descargable (*resultados-kmeans*). También puedes lanzarlo a mano con **Run workflow**.

## Resultado principal

El par **longitud y ancho del pétalo** produce la agrupación más clara: con k = 3, silueta de 0.674 y ARI de 0.886 frente a las especies reales.
