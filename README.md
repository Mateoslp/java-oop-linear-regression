# Práctica 3 – Regresión Lineal con Programación Orientada a Objetos

Este repositorio contiene un framework orientado a objetos para entrenar y evaluar modelos de **regresión lineal simple** y **múltiple** en **Java**, desarrollado desde cero sin el uso de librerías externas.

El proyecto cumple con todos los requerimientos de la práctica:
- **Atributos:** `weights[]` y `bias`.
- **Métodos:** `fit()`, `predict()`, `score()` y `data scaling()`.
- **Extras:** Implementación de dos métodos de entrenamiento (Ecuación Normal y Descenso de Gradiente), escalado de características (StandardScaler), un lector de archivos CSV y utilidades para operaciones matriciales.

## Integrantes y Enlaces

- **Integrantes:** Juan Fernando Gómez Rivas e Ikker Mateo Gil Jordán
- **Repositorio GitHub:** https://github.com/Mateoslp/java-oop-linear-regression
- **Video en YouTube (≤ 10 min):** _(pegar URL del video)_

> En el video deben presentarse, explicar el código, las estrategias de desarrollo y mostrar una ejecución del programa.

## Estructura del Proyecto

El código fuente está organizado de la siguiente manera para promover la modularidad y la claridad:

```
src/
  edu/eafit/oop/lr/
    App.java                 # Demo CLI para entrenar, predecir y evaluar desde un CSV
    CSVReader.java           # Lector de CSV minimalista
    StandardScaler.java      # Implementa data_scaling() para escalar características
    RegressionModel.java     # Interfaz que define el contrato de un modelo de regresión
    LinearRegression.java    # Implementación del modelo con Ecuación Normal y Descenso de Gradiente
    Matrix.java              # Utilidades para operaciones matriciales (T, dot, inversa)
    VectorStats.java         # Funciones de ayuda para calcular métricas (MSE, R², MAE)
```

## Compilación y Ejecución (Java ≥ 11)

Sigue estos pasos para compilar y ejecutar el programa desde la terminal:

1.  **Compilar:**
    ```bash
    javac -d out $(find src -name "*.java")
    ```

2.  **Obtener ayuda:**
    ```bash
    java -cp out edu.eafit.oop.lr.App --help
    ```

## Estrategias de Entrenamiento Implementadas

El modelo de regresión lineal puede ser entrenado utilizando dos algoritmos diferentes, cada uno con sus propias características y casos de uso.

### 1. Ecuación Normal

Este método calcula los parámetros óptimos del modelo (θ, que incluye `bias` y `weights`) de forma analítica y directa, sin necesidad de iteraciones. La fórmula matemática utilizada es:

**θ = (XᵀX + λI)⁻¹ Xᵀy**

- **`X`** es la matriz de características (con una columna de unos para el término de sesgo o `bias`).
- **`y`** es el vector de la variable objetivo.
- **`(XᵀX)⁻¹`** es la inversa de la matriz `X` transpuesta multiplicada por `X`.
- **`λI`** es un término de regularización (Ridge) muy pequeño (`λ = 1e-8`) que se añade a la diagonal de `XᵀX`. Su propósito es garantizar que la matriz sea invertible, incluso si las características están correlacionadas, mejorando la estabilidad numérica.

En el código, esta lógica se encuentra en el método `fit()` de la clase `LinearRegression.java` cuando se selecciona el `TrainingMethod.NORMAL_EQUATION`.

### 2. Descenso de Gradiente (Gradient Descent)

A diferencia de la Ecuación Normal, el Descenso de Gradiente es un método iterativo que ajusta gradualmente los parámetros del modelo para minimizar el error. El algoritmo sigue estos pasos:

1.  Inicializa los pesos (`weights`) y el sesgo (`bias`) a cero.
2.  Calcula las predicciones del modelo (ŷ) con los parámetros actuales.
3.  Calcula el error (la diferencia entre las predicciones ŷ y los valores reales y).
4.  Ajusta los parámetros en la dirección opuesta al gradiente del error, utilizando la fórmula:
    **θ ← θ − α/m · Xᵀ(Xθ − y)**
5.  Repite los pasos 2-4 durante un número determinado de `epochs` (iteraciones) o hasta que el cambio en los parámetros sea insignificante.

- **`α` (alpha)** es la tasa de aprendizaje (`learningRate`), que controla el tamaño de los ajustes en cada iteración.
- **`m`** es el número de muestras de entrenamiento.

Este método es muy eficiente para conjuntos de datos con una gran cantidad de características y es fundamental para modelos más complejos como las redes neuronales. Para su correcto funcionamiento, es crucial escalar las características previamente.

## Resultados de las Pruebas

A continuación se muestran los resultados obtenidos al ejecutar el programa con los dos conjuntos de datos proporcionados.

### Ejemplo 1: Regresión Simple (`ice_cream.csv`)

Para este caso, se predice el número de ventas (`sales`) a partir de la temperatura (`temperature`).

**Comando:**
```bash
java -cp out edu.eafit.oop.lr.App \
  --file ice_cream.csv \
  --target-col sales \
  --method normal \
  --scale standard
```
*(Nota: El archivo `ice_cream.csv` no fue incluido en este análisis, pero el comando está estructurado según la especificación del proyecto).*

### Ejemplo 2: Regresión Múltiple (`student_exam_scores.csv`)

Aquí se predice la nota final de un examen (`exam_score`) basándose en cuatro características: horas de estudio, horas de sueño, porcentaje de asistencia y notas previas.

#### Ejecución con Ecuación Normal

**Comando:**
```bash
java -cp out edu.eafit.oop.lr.App \
  --file student_exam_scores.csv \
  --target-col exam_score \
  --method normal \
  --scale none
```

**Salida:**
```
== Model parameters ==
bias: 4.882963162607141
weights: [2.008432822165042, 0.4907198188164053, 0.09886024927236592, 0.05739097745778263]

== Predictions (first 5) ==
y_hat[0] = 30.519213 (y=29.300000)
y_hat[1] = 27.697529 (y=28.900000)
y_hat[2] = 40.528469 (y=46.000000)
y_hat[3] = 39.529888 (y=35.900000)
y_hat[4] = 34.026402 (y=30.000000)

== Scores ==
R2  : 0.771141
MSE : 12.001944
MAE : 2.760144
```

#### Ejecución con Descenso de Gradiente

**Comando:**
```bash
java -cp out edu.eafit.oop.lr.App \
  --file student_exam_scores.csv \
  --target-col exam_score \
  --method gd \
  --alpha 0.01 \
  --epochs 20000 \
  --scale standard
```

**Salida:**
```
== Model parameters ==
bias: 33.72250107384333
weights: [5.875200230248466, 1.4037597148818596, 0.8114141634629729, 0.4851253011386595]

== Predictions (first 5) ==
y_hat[0] = 30.519183 (y=29.300000)
y_hat[1] = 27.697561 (y=28.900000)
y_hat[2] = 40.528416 (y=46.000000)
y_hat[3] = 39.529849 (y=35.900000)
y_hat[4] = 34.026362 (y=30.000000)

== Scores ==
R2  : 0.771141
MSE : 12.001940
MAE : 2.760143

(Features were scaled with StandardScaler: mean=0, std=1)
```

## Desafíos y Soluciones

Durante el desarrollo, uno de los principales desafíos técnicos fue la implementación de la Ecuación Normal. La operación de inversión de la matriz `(XᵀX)⁻¹` es numéricamente inestable si la matriz es **singular** o casi singular (lo que ocurre cuando las características son linealmente dependientes).

- **Problema:** El programa arrojaba un error al intentar invertir una matriz singular.
- **Solución:** Se implementó una forma de regularización conocida como **Ridge (L2)**. Consiste en sumar una matriz identidad multiplicada por un valor muy pequeño (lambda, `λ`) a `XᵀX` antes de la inversión. Esto asegura que la matriz resultante sea siempre invertible, estabilizando la solución sin afectar significativamente el resultado. Esta solución se encuentra en la clase `LinearRegression.java`.

## Conclusiones

1.  **Efecto del Escalado de Características:** El escalado de características (usando `StandardScaler`) es fundamental para el Descenso de Gradiente. Sin él, el algoritmo converge muy lentamente o puede divergir, ya que las características con rangos de valores mayores dominan el cálculo del gradiente. En cambio, la Ecuación Normal no requiere escalado para funcionar, aunque puede beneficiar la estabilidad numérica.

2.  **Comparativa: Ecuación Normal vs. Descenso de Gradiente:** La Ecuación Normal es ideal para conjuntos de datos con un número de características bajo (ej. < 10,000), ya que ofrece una solución exacta y directa sin necesidad de ajustar hiperparámetros. Sin embargo, su coste computacional (O(n³)) la hace inviable para datasets más grandes. El Descenso de Gradiente es mucho más escalable y eficiente en esos casos, aunque requiere un ajuste cuidadoso de la tasa de aprendizaje y el número de épocas.

3.  **Importancia del Diseño Orientado a Objetos:** La estructura del proyecto, basada en interfaces (`RegressionModel`) y clases con responsabilidades únicas (`CSVReader`, `Matrix`, `StandardScaler`), demuestra las ventajas de la POO. Este diseño hace que el código sea más fácil de leer, mantener y extender. Por ejemplo, añadir un nuevo tipo de modelo (como Regresión Polinomial) sería tan simple como crear una nueva clase que implemente la interfaz `RegressionModel`.

## Transparencia sobre el Uso de IA

Se empleó asistencia de IA para proponer la estructura organizativa de este documento (`README.md`) y para sugerir la inclusión de la métrica MAE (Error Absoluto Medio) junto a R² y MSE, con el fin de ofrecer una evaluación más completa del rendimiento del modelo. El desarrollo del código, la lógica de los algoritmos y las conclusiones técnicas son trabajo original de los autores.
