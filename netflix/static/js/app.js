/**
 * NETFLIX REGRESIÓN LINEAL STUDIO - CLIENT LOGIC
 * Enfocado al 100% en el cálculo, visualización y evaluación de la Regresión Lineal.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Constantes del Modelo de Regresión Lineal 2D (Películas: Duración vs Edad)
  const REG_2D = {
    m: 0.0297,
    b: 11.02
  };

  // ==========================================
  // 1. Toast Notifications
  // ==========================================
  function showToast(message, type = "success") {
    const container = document.getElementById("toast-container");
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    const icon = type === "success" ? "fa-circle-check text-green" : "fa-circle-exclamation text-netflix";
    toast.innerHTML = `<i class="fa-solid ${icon}"></i> <span>${message}</span>`;

    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(100%)";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // ==========================================
  // 2. Carga y Actualización de Métricas de Regresión
  // ==========================================
  async function cargarMetricasRegresion() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();
      if (data.status === "success" && data.metricas_ml) {
        const ml = data.metricas_ml;

        if (ml.recta_2d) {
          REG_2D.m = ml.recta_2d.pendiente || 0.0297;
          REG_2D.b = ml.recta_2d.intercepto || 11.02;

          const eqFormula = document.getElementById("display-eq-formula");
          if (eqFormula) {
            eqFormula.innerHTML = `Edad Recomendada = (<span class="text-cyan">${REG_2D.m}</span> × Duración en min) + <span class="text-gold">${REG_2D.b}</span>`;
          }
          const spPen = document.getElementById("sp-pendiente");
          const spInt = document.getElementById("sp-intercepto");
          if (spPen) spPen.innerText = `+${REG_2D.m}`;
          if (spInt) spInt.innerText = `${REG_2D.b} años`;
        }

        const kpiR2 = document.getElementById("kpi-r2-score");
        const kpiR2Train = document.getElementById("kpi-r2-train");
        const kpiR2Test = document.getElementById("kpi-r2-test");
        const kpiMae = document.getElementById("kpi-mae");
        const kpiMaeDisp = document.getElementById("kpi-mae-display");
        const kpiRmse = document.getElementById("kpi-rmse");
        const kpiInt = document.getElementById("kpi-intercepto");

        if (kpiR2) kpiR2.innerText = ml.r2_test;
        if (kpiR2Train) kpiR2Train.innerText = ml.r2_train;
        if (kpiR2Test) kpiR2Test.innerText = ml.r2_test;
        if (kpiMae) kpiMae.innerText = `±${ml.mae_test} años`;
        if (kpiMaeDisp) kpiMaeDisp.innerHTML = `${ml.mae_test} <span class="unit">años</span>`;
        if (kpiRmse) kpiRmse.innerHTML = `${ml.rmse_test} <span class="unit">años</span>`;
        if (kpiInt) kpiInt.innerHTML = `${ml.intercepto} <span class="unit">años</span>`;
      }
    } catch (err) {
      console.warn("Usando parámetros locales de regresión:", err);
    }
  }

  // ==========================================
  // 3. Calculador de Regresión Lineal 2D (Tiempo Real)
  // ==========================================
  const sliderDuration = document.getElementById("input-simple-duration");
  const labelDurVal = document.getElementById("label-simple-dur-val");
  const mathX = document.getElementById("math-x-val");
  const mathMx = document.getElementById("math-mx-val");
  const mathY = document.getElementById("math-y-val");

  function resolverRegresion2D(duracionMin) {
    const mx = Number((REG_2D.m * duracionMin).toFixed(2));
    const rawY = Number((mx + REG_2D.b).toFixed(2));
    const edad = Math.max(0, Math.min(18, rawY));

    // Actualizar caja matemática
    if (labelDurVal) labelDurVal.innerText = `${duracionMin} min`;
    if (mathX) mathX.innerText = duracionMin;
    if (mathMx) mathMx.innerText = mx.toFixed(2);
    if (mathY) mathY.innerText = `${edad.toFixed(2)} años`;

    // Determinar categoría y badge
    actualizarClasificacionUI(edad);
  }

  function actualizarClasificacionUI(edad) {
    const badgeEl = document.getElementById("res-badge-large");
    const titleEl = document.getElementById("res-classification-title");
    const catEl = document.getElementById("res-category");
    const exactAgeEl = document.getElementById("res-exact-age");
    const fillEl = document.getElementById("gauge-bar-fill");
    const needleEl = document.getElementById("gauge-needle");

    let badge = "TP";
    let title = "TV-Y / G (Para todos los públicos)";
    let cat = "Público Infantil / Familiar (Todo espectador)";
    let color = "#00e676";

    if (edad >= 17) {
      badge = "+18";
      title = "NC-17 / TV-MA (Exclusivo Adultos)";
      cat = "Solo Adultos / Contenido Explícito (+18)";
      color = "#E50914";
    } else if (edad >= 15.5) {
      badge = "+16";
      title = "TV-MA / R (Maduro / Adolescentes Mayores)";
      cat = "Audiencia Madura (+16 a +17 años)";
      color = "#ff4444";
    } else if (edad >= 11) {
      badge = "+13";
      title = "PG-13 / TV-14 (Guía Parental Requerida)";
      cat = "Adolescentes y Adultos (+13 a +14 años)";
      color = "#ffb703";
    } else if (edad >= 6.5) {
      badge = "+7";
      title = "TV-PG / PG (+7 años en adelante)";
      cat = "Público Infantil Mayor (+7 a +9 años)";
      color = "#00d4ff";
    }

    if (badgeEl) {
      badgeEl.innerText = badge;
      badgeEl.style.backgroundColor = color;
      badgeEl.style.boxShadow = `0 0 22px ${color}88`;
    }
    if (titleEl) titleEl.innerText = title;
    if (catEl) catEl.innerText = cat;
    if (exactAgeEl) exactAgeEl.innerText = edad.toFixed(1);

    // Ajustar barra de aguja porcentual
    const pct = Math.min(100, Math.max(0, (edad / 18) * 100));
    if (fillEl) fillEl.style.width = `${pct}%`;
    if (needleEl) needleEl.style.left = `${pct}%`;
  }

  if (sliderDuration) {
    sliderDuration.addEventListener("input", (e) => {
      resolverRegresion2D(parseInt(e.target.value));
    });
  }

  // ==========================================
  // 4. Conmutador de Modo: Simple 2D vs Múltiple
  // ==========================================
  const btnModeSimple = document.getElementById("btn-mode-simple");
  const btnModeMultiple = document.getElementById("btn-mode-multiple");
  const secSimple = document.getElementById("section-mode-simple");
  const secMultiple = document.getElementById("section-mode-multiple");

  if (btnModeSimple && btnModeMultiple) {
    btnModeSimple.addEventListener("click", () => {
      btnModeSimple.classList.add("active");
      btnModeMultiple.classList.remove("active");
      secSimple.style.display = "block";
      secMultiple.style.display = "none";
      if (sliderDuration) resolverRegresion2D(parseInt(sliderDuration.value));
    });

    btnModeMultiple.addEventListener("click", () => {
      btnModeMultiple.classList.add("active");
      btnModeSimple.classList.remove("active");
      secSimple.style.display = "none";
      secMultiple.style.display = "block";
    });
  }

  // ==========================================
  // 5. Modo Regresión Múltiple (Scikit-Learn API)
  // ==========================================
  const radioPills = document.querySelectorAll(".radio-pill");
  const inputMultiYear = document.getElementById("input-multi-year");
  const labelMultiYear = document.getElementById("label-multi-year-val");
  const inputMultiDur = document.getElementById("input-multi-duration");
  const labelMultiDurVal = document.getElementById("label-multi-dur-val");
  const labelMultiDurText = document.getElementById("label-multi-dur-text");
  const btnSubmitMulti = document.getElementById("btn-submit-multiple");

  let selectedType = "Movie";

  radioPills.forEach(pill => {
    pill.addEventListener("click", () => {
      radioPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const radio = pill.querySelector("input[type='radio']");
      if (radio) {
        radio.checked = true;
        selectedType = radio.value;
      }

      if (selectedType === "Movie") {
        labelMultiDurText.innerText = "Duración (minutos)";
        inputMultiDur.min = 15;
        inputMultiDur.max = 240;
        inputMultiDur.value = 105;
        labelMultiDurVal.innerText = "105 min";
      } else {
        labelMultiDurText.innerText = "Temporadas (Seasons)";
        inputMultiDur.min = 1;
        inputMultiDur.max = 15;
        inputMultiDur.value = 2;
        labelMultiDurVal.innerText = "2 Temp.";
      }
    });
  });

  if (inputMultiYear) {
    inputMultiYear.addEventListener("input", (e) => {
      if (labelMultiYear) labelMultiYear.innerText = e.target.value;
    });
  }

  if (inputMultiDur) {
    inputMultiDur.addEventListener("input", (e) => {
      if (labelMultiDurVal) {
        labelMultiDurVal.innerText = selectedType === "Movie" ? `${e.target.value} min` : `${e.target.value} Temp.`;
      }
    });
  }

  if (btnSubmitMulti) {
    btnSubmitMulti.addEventListener("click", async () => {
      btnSubmitMulti.disabled = true;
      btnSubmitMulti.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Calculando...`;

      const payload = {
        type: selectedType,
        country: document.getElementById("select-country")?.value || "United States",
        genre: document.getElementById("select-genre")?.value || "Dramas",
        release_year: parseInt(inputMultiYear?.value || 2023),
        duration: parseInt(inputMultiDur?.value || 105)
      };

      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const json = await res.json();
        if (json.status === "success" && json.data) {
          const d = json.data;
          actualizarClasificacionUI(d.edad_recomendada);
          showToast(`Predicción calculada: ${d.edad_recomendada} años`);
        } else {
          showToast(json.message || "Error al calcular", "error");
        }
      } catch (err) {
        showToast("Error de conexión con el servidor", "error");
      } finally {
        btnSubmitMulti.disabled = false;
        btnSubmitMulti.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> <span>Calcular Regresión Múltiple</span>`;
      }
    });
  }

  // Inicializar cálculo en 100 min y cargar métricas de la API
  resolverRegresion2D(100);
  cargarMetricasRegresion();
});
