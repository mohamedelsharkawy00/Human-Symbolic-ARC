/* Auto-size the left nav pane to fit its widest label, and add Expand/Collapse
   all buttons. Runs once on load (re-runs after fonts load). Measures every
   label's text width with an offscreen probe - this works even for labels in
   collapsed folders, whose live geometry would read zero. No fixed clamp. */
(function () {
  function sizeNav() {
    var sidebar = document.querySelector(".md-sidebar--primary");
    if (!sidebar) return;
    var inner = sidebar.querySelector(".md-sidebar__inner");
    var links = sidebar.querySelectorAll(".md-nav__item > .md-nav__link");
    if (!links.length || !inner) return;

    // Measure EVERY label's text width with an offscreen probe (using each
    // label's own font). This works even for labels inside collapsed folders -
    // those have a zero-size live rect, so reading their on-screen geometry
    // would skip them (that's why the longest one stayed clipped). The probe
    // measures text regardless of whether the item is currently visible.
    var probe = document.createElement("span");
    probe.style.cssText =
      "position:absolute;left:-9999px;top:0;visibility:hidden;white-space:nowrap;";
    document.body.appendChild(probe);

    // Indent (px) per nav depth, so a deep file's label gets its real left offset.
    function depthOf(el) {
      var d = 0, n = el;
      while (n && n !== sidebar) {
        if (n.classList && n.classList.contains("md-nav__list")) d++;
        n = n.parentElement;
      }
      return d; // 1 = top level, 2 = inside a folder, ...
    }

    var maxWidth = 0;
    sidebar
      .querySelectorAll(".md-nav__item > .md-nav__link .md-ellipsis")
      .forEach(function (span) {
        var cs = getComputedStyle(span);
        probe.style.font = cs.font;
        probe.style.letterSpacing = cs.letterSpacing;
        probe.textContent = span.textContent.trim();
        var indentPx = depthOf(span) * 0.6 * 16; // ~0.6rem per level
        var w = probe.offsetWidth + indentPx;
        if (w > maxWidth) maxWidth = w;
      });
    document.body.removeChild(probe);

    // text/indent + arrow + scrollbar/right margin.
    sidebar.style.width = maxWidth + 60 + "px";
    inner.style.paddingLeft = "0";
    inner.style.paddingRight = "0.4rem";
  }

  /* Add "Expand all / Collapse all" buttons that flip every folder toggle.
     Material renders each folder's open/closed state as a checkbox
     input.md-nav__toggle; checked = open. */
  function addExpandCollapse() {
    var nav = document.querySelector(".md-sidebar--primary .md-nav--primary");
    if (!nav || nav.querySelector(".nav-controls")) return;

    var toggles = function () {
      return document.querySelectorAll(
        ".md-sidebar--primary .md-nav__toggle.md-toggle"
      );
    };
    function setAll(open) {
      toggles().forEach(function (cb) {
        cb.checked = open;
      });
    }

    var bar = document.createElement("div");
    bar.className = "nav-controls";

    var expand = document.createElement("button");
    expand.type = "button";
    expand.textContent = "Expand all";
    expand.addEventListener("click", function () {
      setAll(true);
    });

    var collapse = document.createElement("button");
    collapse.type = "button";
    collapse.textContent = "Collapse all";
    collapse.addEventListener("click", function () {
      setAll(false);
    });

    bar.appendChild(expand);
    bar.appendChild(collapse);

    // Insert just under the sidebar title, above the list.
    var list = nav.querySelector(".md-nav__list");
    if (list) nav.insertBefore(bar, list);
    else nav.appendChild(bar);
  }

  function init() {
    addExpandCollapse();
    sizeNav();
    // Re-measure once web fonts have loaded - before that, the browser uses a
    // fallback font (often wider), which overshoots the pane and leaves a gap.
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(sizeNav);
    }
  }

  // Material's instant loading is OFF here, so a normal load fires once.
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
