/* ═══════════════════════════════════════════════════════
   IrriSmart AI — Main JavaScript
   ═══════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {

  /* ── Sidebar Toggle ─────────────────────────────────── */
  const sidebar      = document.getElementById("sidebar");
  const toggleBtn    = document.getElementById("sidebarToggle");
  const topbar       = document.querySelector(".topbar");
  const mainContent  = document.querySelector(".main-content");
  const isMobile     = () => window.innerWidth <= 768;

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener("click", () => {
      if (isMobile()) {
        sidebar.classList.toggle("mobile-open");
      } else {
        sidebar.classList.toggle("collapsed");
        topbar.classList.toggle("expanded");
        mainContent.classList.toggle("expanded");
      }
    });
  }

  // Close sidebar on mobile when clicking outside
  document.addEventListener("click", e => {
    if (isMobile() && sidebar &&
        !sidebar.contains(e.target) &&
        !toggleBtn.contains(e.target)) {
      sidebar.classList.remove("mobile-open");
    }
  });

  /* ── Dark Mode Toggle ───────────────────────────────── */
  const themeToggle = document.getElementById("themeToggle");
  const html        = document.documentElement;

  // Persist theme
  const savedTheme = localStorage.getItem("theme") || "light";
  if (savedTheme === "dark") {
    html.setAttribute("data-theme", "dark");
    updateThemeIcon("dark");
  }

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const current = html.getAttribute("data-theme");
      const next    = current === "dark" ? "light" : "dark";
      html.setAttribute("data-theme", next);
      localStorage.setItem("theme", next);
      updateThemeIcon(next);
    });
  }

  function updateThemeIcon(theme) {
    if (!themeToggle) return;
    themeToggle.innerHTML = theme === "dark"
      ? '<i data-lucide="moon"></i>'
      : '<i data-lucide="sun"></i>';
    lucide.createIcons();
  }

  /* ── Auto-dismiss flash messages ───────────────────── */
  document.querySelectorAll(".flash").forEach(el => {
    setTimeout(() => {
      el.style.transition = "opacity .4s ease, transform .4s ease";
      el.style.opacity    = "0";
      el.style.transform  = "translateY(-6px)";
      setTimeout(() => el.remove(), 400);
    }, 5000);
  });

  /* ── Animate metric values ──────────────────────────── */
  const metricValues = document.querySelectorAll(".metric-value");
  metricValues.forEach(el => {
    const raw = parseFloat(el.textContent);
    if (!isNaN(raw)) {
      animateNumber(el, raw);
    }
  });

  function animateNumber(el, target) {
    const unit = el.querySelector(".metric-unit");
    const unitText = unit ? unit.textContent : "";
    const isInt    = Number.isInteger(target);
    const duration = 1000;
    const start    = performance.now();

    function update(now) {
      const pct = Math.min((now - start) / duration, 1);
      const ease = 1 - Math.pow(1 - pct, 3);
      const cur  = target * ease;
      const fmt  = isInt ? Math.round(cur) : cur.toFixed(4);
      el.innerHTML = `${fmt}<span class="metric-unit">${unitText}</span>`;
      if (pct < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
  }

  /* ── Tooltip helper ─────────────────────────────────── */
  document.querySelectorAll("[data-tooltip]").forEach(el => {
    const tip = document.createElement("div");
    tip.className   = "tooltip-bubble";
    tip.textContent = el.dataset.tooltip;
    document.body.appendChild(tip);

    el.addEventListener("mouseenter", e => {
      const r       = el.getBoundingClientRect();
      tip.style.top  = (r.top - 36 + window.scrollY) + "px";
      tip.style.left = (r.left + r.width/2) + "px";
      tip.style.transform = "translateX(-50%)";
      tip.style.opacity   = "1";
    });
    el.addEventListener("mouseleave", () => {
      tip.style.opacity = "0";
    });
  });

  /* ── Init ───────────────────────────────────────────── */
  lucide.createIcons();
});