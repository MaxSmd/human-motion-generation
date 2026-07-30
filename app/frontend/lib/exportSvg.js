// Export a live <svg> (e.g. a LineChart) as a standalone .svg or .png for
// dropping into a paper / slide. The on-page SVG styles colours through CSS
// custom properties (var(--muted), …) and Tailwind classes, none of which
// survive serialization — so we clone the node and bake the *computed* fill /
// stroke / font of every element into inline attributes first.

function bake(src, dst) {
  const cs = getComputedStyle(src);
  for (const prop of ["fill", "stroke", "stroke-width", "stroke-dasharray", "font-size", "font-family", "opacity"]) {
    const v = cs.getPropertyValue(prop);
    if (v && v !== "none" && v !== "normal") dst.setAttribute(prop, v);
  }
  const sc = src.children, dc = dst.children;
  for (let i = 0; i < sc.length; i++) if (dc[i]) bake(sc[i], dc[i]);
}

function serialize(svg, { background = "#070a11" } = {}) {
  const clone = svg.cloneNode(true);
  bake(svg, clone);
  const w = svg.viewBox?.baseVal?.width || svg.width.baseVal.value;
  const h = svg.viewBox?.baseVal?.height || svg.height.baseVal.value;
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("width", w);
  clone.setAttribute("height", h);
  // paint a background so it isn't transparent on white slides
  const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  rect.setAttribute("x", 0); rect.setAttribute("y", 0);
  rect.setAttribute("width", w); rect.setAttribute("height", h);
  rect.setAttribute("fill", background);
  clone.insertBefore(rect, clone.firstChild);
  return { xml: new XMLSerializer().serializeToString(clone), w, h };
}

function triggerDownload(href, filename) {
  const a = document.createElement("a");
  a.href = href; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
}

export function downloadSvg(svg, filename = "chart.svg", opts) {
  const { xml } = serialize(svg, opts);
  const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml" }));
  triggerDownload(url, filename);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function downloadPng(svg, filename = "chart.png", { scale = 2, ...opts } = {}) {
  const { xml, w, h } = serialize(svg, opts);
  const img = new Image();
  const url = URL.createObjectURL(new Blob([xml], { type: "image/svg+xml" }));
  img.onload = () => {
    const canvas = document.createElement("canvas");
    canvas.width = w * scale; canvas.height = h * scale;
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);
    canvas.toBlob((blob) => {
      const purl = URL.createObjectURL(blob);
      triggerDownload(purl, filename);
      setTimeout(() => URL.revokeObjectURL(purl), 1000);
    });
  };
  img.src = url;
}
