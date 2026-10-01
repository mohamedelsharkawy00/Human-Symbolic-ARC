/* Auto-size the left nav pane to fit its widest label, and add Expand/Collapse
   all buttons. Nothing here is measured in fixed pixels: the browser has already
   laid the labels out at this screen's size, zoom and font, so we only ask it how
   much the widest one is being cut off by, and widen the pane by exactly that. */
(function () {
  // The thing worth protecting is the text you came to read, not the nav, so the
  // limit says what the text keeps rather than what the nav may take. Capping the
  // nav's share instead punished the narrow window, which is where the room was
  // most needed - and a window is narrow in these units whenever the screen is
  // scaled, since scaling hands the page fewer pixels to spend.
  var SHARE_LEFT_FOR_TEXT = 0.5;

  function widestAllowed() {
    return window.innerWidth * (1 - SHARE_LEFT_FOR_TEXT);
  }

  // A width you dragged holds for as long as you stay on the page and no longer:
  // a refresh is how you ask for the measured fit back. Kept as a share of the
  // window rather than pixels, so it keeps its proportion if the window moves to
  // the other screen. Zero means nobody has dragged, so the fit decides.
  var draggedShare = 0;

  function sizeNav() {
    var sidebar = document.querySelector(".md-sidebar--primary");
    if (!sidebar) return;
    var inner = sidebar.querySelector(".md-sidebar__inner");
    var spans = sidebar.querySelectorAll(".md-nav__item > .md-nav__link .md-ellipsis");
    if (!spans.length || !inner) return;

    inner.style.paddingLeft = "0";
    inner.style.paddingRight = "0.4rem";

    // The sidebar's own box is a few pixels tall, so the grip is given the
    // height of the scrolling list instead - the part you actually see.
    var grip = sidebar.querySelector(".nav-grip");
    var scrollwrap = sidebar.querySelector(".md-sidebar__scrollwrap");
    if (grip && scrollwrap) {
      grip.style.height = scrollwrap.getBoundingClientRect().height + "px";
    }

    // Your own width wins over the fitted one, until you double-click the grip.
    if (draggedShare) {
      sidebar.style.width = window.innerWidth * draggedShare + "px";
      return;
    }

    // Start from the theme's own width every time, so moving the window to the
    // smaller screen lets the pane shrink again instead of only ever growing.
    sidebar.style.width = "";

    // A label inside a closed folder has no size at all, so open every folder to
    // measure, then put each toggle back exactly as the reader left it.
    var toggles = sidebar.querySelectorAll(".md-nav__toggle.md-toggle");
    var wasOpen = [];
    toggles.forEach(function (toggle, i) {
      wasOpen[i] = toggle.checked;
      toggle.checked = true;
    });

    // Widen until nothing overruns, asking again after each try instead of
    // trusting one answer: growing the pane can bring a scrollbar in or out, and
    // whether a scrollbar steals width at all is the browser's choice, not ours.
    // Asking again costs one layout read and settles in a pass or two.
    var most = widestAllowed();
    for (var pass = 0; pass < 4; pass++) {
      var overflow = 0;
      spans.forEach(function (span) {
        var over = span.scrollWidth - span.clientWidth;
        if (over > overflow) overflow = over;
      });
      if (overflow <= 0) break;
      var wanted = sidebar.getBoundingClientRect().width + overflow;
      if (wanted >= most) {
        sidebar.style.width = most + "px";
        break;
      }
      sidebar.style.width = wanted + "px";
    }

    toggles.forEach(function (toggle, i) {
      toggle.checked = wasOpen[i];
    });
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
      // Opening folders can push the list long enough to need a scrollbar, and
      // that scrollbar takes width away from the labels - so re-fit on the
      // change itself rather than hoping the load-time fit still holds.
      sizeNav();
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

  /* A grip on the pane's right edge: drag to set the width yourself, double-click
     it to hand the width back to the automatic fit. */
  function addResizeGrip() {
    var sidebar = document.querySelector(".md-sidebar--primary");
    if (!sidebar || sidebar.querySelector(".nav-grip")) return;

    var grip = document.createElement("div");
    grip.className = "nav-grip";
    grip.title = "Drag to resize, double-click to fit automatically";
    sidebar.appendChild(grip);

    grip.addEventListener("pointerdown", function (down) {
      down.preventDefault();
      grip.setPointerCapture(down.pointerId);
      var startX = down.clientX;
      var startWidth = sidebar.getBoundingClientRect().width;

      function onMove(move) {
        var width = startWidth + (move.clientX - startX);
        // Wide enough to read, never wider than the window can spare.
        sidebar.style.width = Math.max(80, Math.min(width, widestAllowed())) + "px";
      }
      function onUp() {
        grip.removeEventListener("pointermove", onMove);
        grip.removeEventListener("pointerup", onUp);
        draggedShare = sidebar.getBoundingClientRect().width / window.innerWidth;
      }
      grip.addEventListener("pointermove", onMove);
      grip.addEventListener("pointerup", onUp);
    });

    grip.addEventListener("dblclick", function () {
      draggedShare = 0;
      sizeNav();
    });
  }

  function init() {
    addExpandCollapse();
    addResizeGrip();
    sizeNav();
    // Re-measure once web fonts have loaded - before that, the browser uses a
    // fallback font (often wider), which overshoots the pane and leaves a gap.
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(sizeNav);
    }
    // Dragging the window to the other screen resizes it, so re-fit on resize.
    // Settle first: a drag fires this continuously and each fit reads layout.
    var settle;
    window.addEventListener("resize", function () {
      clearTimeout(settle);
      settle = setTimeout(sizeNav, 150);
    });

    // Opening one folder by hand changes what is on show just as much as the
    // Expand all button does, so it gets the same re-fit.
    var sidebar = document.querySelector(".md-sidebar--primary");
    if (sidebar) {
      sidebar.addEventListener("change", function (event) {
        if (event.target.classList.contains("md-nav__toggle")) sizeNav();
      });
    }
  }

  // Material's instant loading is OFF here, so a normal load fires once.
  if (document.readyState !== "loading") init();
  else document.addEventListener("DOMContentLoaded", init);
})();
