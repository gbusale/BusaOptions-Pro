# BusaOptions Pro 9.6

## Novedades 9.6: gráficos más claros y profesionales
- Paleta de colores consistente estilo plataforma de trading (verde/rojo tipo
  TradingView en vez de los tonos genéricos anteriores).
- Fondo transparente en los 5 gráficos: se funden con el tema oscuro de la
  app en vez de mostrar un recuadro con borde propio.
- Título propio dentro de cada gráfico, grilla sutil, y título en los ejes
  (Precio (ARS), Volumen, RSI, MACD, ADX / DI).
- Tooltips prolijos con formato de miles y una sola etiqueta por serie
  (antes usaban el tooltip genérico de Plotly).
- Línea punteada de "último valor" con su etiqueta a la derecha, como en las
  plataformas profesionales (precio, RSI).
- Zonas de sobrecompra/sobreventa sombreadas en el RSI, no solo líneas.
- Precio ahora en velas con las bandas de Bollinger detrás (antes el precio
  era una línea simple).
- Cada sección (pronóstico, veredicto, y cada gráfico) va dentro de una
  tarjeta con borde (`st.container(border=True)`), para separar visualmente
  sin depender de tantos separadores.
- El veredicto técnico ahora tiene un ícono (▲ / ▼ / ➡) y tipografía más grande.

## Novedades 9.5: análisis técnico independiente, veredicto consolidado y volumen/ADX
- Cada indicador ahora tiene **su propio gráfico independiente y su propia
  explicación** (antes estaban todos apretados en un solo gráfico de 3 paneles):
  1. Precio en velas + Bollinger(20,2) + EMA20/EMA50
  2. Volumen (con promedio de 20 ruedas y color según suba/baja del día)
  3. RSI(14)
  4. MACD(12,26,9)
  5. ADX(14) + DI/-DI — indicador nuevo, ver más abajo
- **Veredicto técnico consolidado (SUBE/BAJA/LATERAL):** votación transparente
  entre RSI, MACD, posición vs. centro de Bollinger, tendencia (EMA50) y
  volumen como confirmación. Es independiente del pronóstico estadístico
  Busa AI (que sigue mostrado arriba, sin cambios) -- pueden coincidir o no,
  y el panel lo aclara explícitamente para no confundir ambos.
- **ADX(14) agregado:** RSI y MACD miden dirección/momentum pero no si hay
  una tendencia real. El ADX mide la fuerza de la tendencia; si está por
  debajo de 25, el veredicto técnico se marca como "poco confiable" (mercado
  en rango) en vez de sobre-interpretar una señal direccional débil.
- Gráficos más grandes, con leyenda propia por indicador (antes compartían
  una sola leyenda apretada arriba de todo).

## Novedades 9.4: tab "Probabilidades" como panel de análisis técnico experto
- La pestaña Probabilidades ahora muestra, para **GGAL e YPF a la vez** (no solo
  el activo elegido en la barra lateral), un panel completo:
  - Pronóstico Busa AI (Sube/Baja/Lateral) de cada activo.
  - Lectura narrativa tipo "experto en análisis técnico": RSI(14), MACD(12,26,9),
    Bandas de Bollinger(20,2) y posición respecto de la EMA50, calculados sobre
    el historial real (yfinance), no sobre valores aproximados.
  - Gráfico combinado de 3 paneles (precio+Bollinger+EMAs / RSI / MACD), estilo
    plotly_dark, igual de interactivo que la pestaña Velas.
- Estos cálculos son independientes del motor de pronóstico Sube/Baja/Lateral
  (que usa un modelo lognormal con shrinkage); acá se trata de los indicadores
  clásicos de gráfico que pidió el usuario para completar la lectura.

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
