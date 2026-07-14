# BusaOptions Pro 9.3

Mejoras sobre 9.2 (motor de pronóstico, aprendizaje y estrategias):

## Pronóstico de probabilidades (Sube/Baja/Lateral)
- El retorno medio histórico ya no se usa "crudo" como tendencia (drift): se
  le aplica *shrinkage* (control en la barra lateral) porque es un estimador
  muy ruidoso y antes hacía que rachas cortas se extrapolaran de forma exagerada.
- Volatilidad combinada: mezcla desvío simple + EWMA (más sensible a cambios
  recientes de volatilidad), en vez de sólo el desvío simple de la ventana.
- Sesgo técnico opcional y acotado (RSI, momentum 5 ruedas, distancia a
  EMA20/EMA50), activable/desactivable. Antes esos indicadores sólo se
  mostraban como "explicación" en Busa AI pero no influían en el número.

## Aprendizaje (Learning)
- Antes: un único factor, calculado sobre la "predicción dominante" de cada
  señal, con escalones fijos (n<5 → sin ajuste; si no, saltos discretos).
- Ahora: ajuste bayesiano (Beta-Binomial) **por clase** (Sube/Baja/Lateral por
  separado), con suavizado que evita sobrerreaccionar a pocas señales y que
  converge gradualmente a medida que se acumula historial evaluado.
- Nuevo panel de calibración: Brier score y accuracy por clase, visibles en
  la pestaña Busa AI.

## Elección de estrategias (Advisor)
- Antes: sólo se generaban estrategias "a favor" de la predicción dominante,
  y la "probabilidad de éxito" era una heurística lineal por distancia al
  break-even.
- Ahora: se generan candidatas de todos los tipos (calls, puts, spreads,
  straddle/strangle, butterfly) y se rankean por probabilidad de éxito +
  valor esperado, calculados integrando el payoff contra la misma
  distribución lognormal usada en el pronóstico — más un componente de
  liquidez (volumen + spread compra/venta).
- El límite de pérdida sobre capital (slider) ahora sí filtra estrategias;
  antes el parámetro se recibía pero nunca se aplicaba.
- Se corrigió el gráfico de payoff de la recomendación: antes fallaba
  silenciosamente para straddle/strangle, bear put spread y butterfly.

## Se mantiene sin cambios
- Integración con la API de IOL (`api_iol.py`), autenticación y endpoints.
- Estructura de despliegue en Streamlit Cloud / GitHub.
- Backup/restore del historial de Learning en CSV.
- Selectboxes con defaults robustos para móvil/cloud.
