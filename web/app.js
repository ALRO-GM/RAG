// RAGPutin results site: a dependency-free single-page app.
// It only reads the JSON written by `python -m results_site build`
// (data/*.json) and the Markdown copies under docs/. No server code.
"use strict";

const view = document.getElementById("view");
const cache = new Map();

// ---------------------------------------------------------------- helpers

const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));

const num = (value, digits = 3) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";

const pct = (value, digits = 1) =>
  typeof value === "number" ? (value * 100).toFixed(digits) : "—";

const signed = (value, digits = 1) => {
  if (typeof value !== "number") return '<span class="muted">—</span>';
  const points = value * 100;
  const cls = points > 0.05 ? "pos" : points < -0.05 ? "neg" : "muted";
  return `<span class="${cls}">${points > 0 ? "+" : ""}${points.toFixed(digits)}</span>`;
};

const bar = (value, max = 1) => {
  const width = typeof value === "number" && max > 0
    ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return `<div class="bar" role="img" aria-label="${esc(num(value))}"><i style="width:${width}%"></i></div>`;
};

const params = (obj) => {
  const entries = Object.entries(obj || {});
  if (!entries.length) return '<span class="params">igual que la base</span>';
  return `<span class="params">${entries.map(([k, v]) =>
    `${esc(k)}=${esc(typeof v === "object" ? JSON.stringify(v) : v)}`).join(" · ")}</span>`;
};

async function load(path, kind = "json") {
  if (!cache.has(path)) {
    cache.set(path, fetch(path).then((r) => {
      if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
      return kind === "json" ? r.json() : r.text();
    }).catch((err) => { cache.delete(path); throw err; }));
  }
  return cache.get(path);
}

const render = (html) => { view.innerHTML = html; };
const empty = (text) => `<div class="card empty">${esc(text)}</div>`;

// Sortable table: columns [{key, label, num, html(row), sort(row)}].
function table(id, columns, rows, opts = {}) {
  const state = table.state[id] || (table.state[id] = { key: opts.sortKey, dir: -1 });
  const col = columns.find((c) => c.key === state.key);
  const sorted = col ? [...rows].sort((a, b) => {
    const x = (col.sort || ((r) => r[col.key]))(a);
    const y = (col.sort || ((r) => r[col.key]))(b);
    if (x == null) return 1;
    if (y == null) return -1;
    return (x > y ? 1 : x < y ? -1 : 0) * state.dir;
  }) : rows;
  const head = columns.map((c) => {
    const mark = c.key === state.key ? (state.dir < 0 ? " ↓" : " ↑") : "";
    return `<th class="${c.num ? "num " : ""}${c.sort !== false ? "sortable" : ""}" data-table="${id}" data-key="${c.key}">${esc(c.label)}${mark}</th>`;
  }).join("");
  const body = sorted.map((row) =>
    `<tr class="${opts.rowClass ? opts.rowClass(row) : ""}">${columns.map((c) =>
      `<td class="${c.num ? "num" : ""}">${c.html ? c.html(row) : esc(row[c.key])}</td>`).join("")}</tr>`).join("");
  return `<div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}
table.state = {};

view.addEventListener("click", (event) => {
  const th = event.target.closest("th.sortable");
  if (!th) return;
  const state = table.state[th.dataset.table] || (table.state[th.dataset.table] = {});
  state.dir = state.key === th.dataset.key ? -(state.dir || -1) : -1;
  state.key = th.dataset.key;
  route();
});

// ------------------------------------------------------------------ pages

async function pageHome() {
  const manifest = await load("data/manifest.json");
  const get = (name) => manifest.sections[name] ? load(`data/${name}.json`) : Promise.resolve(null);
  const [lab, beir, skills, hardware] = await Promise.all(
    ["lab", "beir", "skills", "hardware"].map(get));
  const final = lab && (lab.suites.find((s) => s.suite === "final") || lab.suites[0]);
  // The shipped configuration is the baseline of the final suite, not its
  // best trial (the reranker wins a little there but is off by default).
  const best = final && (final.trials.find((t) => t.id === final.baseline_id) || final.trials[0]);
  const shipped = beir && beir.averages.find((a) => /as shipped/i.test(a.config) && /hybrid/i.test(a.config));
  const run = hardware && hardware.latest;
  const gpuBest = run && run.results.find((r) => r.plan.device !== "cpu");
  const cpuBest = run && run.results.find((r) => r.plan.device === "cpu");
  const docs = manifest.docs.reduce((n, g) => n + g.items.length, 0);
  const kpi = (value, label, note = "", href = "") =>
    `<${href ? `a href="${href}"` : "div"} class="card kpi"><div class="value">${value}</div><div class="label">${label}</div>${note ? `<div class="note">${note}</div>` : ""}</${href ? "a" : "div"}>`;
  const metrics = best ? best.metrics : {};
  render(`
    <h1>Resultados</h1>
    <p class="lead">Todo lo que ha medido el proyecto: el laboratorio de recuperación sobre el corpus de vLLM,
      los benchmarks públicos (BEIR, CoIR), el analizador de soft skills y el hardware. Las pruebas se ejecutan
      en la estación de trabajo con GPU y esta web solo publica sus resultados.</p>
    <div class="grid">
      ${kpi(pct(metrics.docs && metrics.docs["recall@5"]), "Recall@5 docs (moulinette)", "configuración por defecto", "#/lab/final")}
      ${kpi(pct(metrics.code && metrics.code["recall@5"]), "Recall@5 código (moulinette)", "configuración por defecto", "#/lab/final")}
      ${kpi(shipped ? num(shipped.ndcg10) : "—", "nDCG@10 medio, sistema tal cual", shipped ? `${shipped.datasets} datasets públicos` : "", "#/beir")}
      ${kpi(run && run.gpu_speedup ? `${num(run.gpu_speedup, 1)}×` : "—", "GPU frente a la mejor CPU",
        gpuBest && cpuBest ? `${num(gpuBest.tokens_per_second, 1)} frente a ${num(cpuBest.tokens_per_second, 1)} tok/s` : "sin medición de GPU", "#/hardware")}
      ${kpi(lab ? lab.trial_count : 0, "Pruebas del laboratorio", lab ? `${lab.suites.length} suites` : "", "#/lab")}
      ${kpi(skills ? skills.trial_count : 0, "Pruebas de soft skills", `${manifest.rubrics} rúbricas`, "#/skills")}
      ${kpi(docs, "Documentos", `${manifest.docs.length} secciones`, "#/docs")}
    </div>`);
}

function suiteCards(summary, base) {
  return `<div class="grid">${summary.suites.map((s) => `
    <a class="card" href="${base}/${encodeURIComponent(s.suite)}">
      <strong>${esc(s.suite)}</strong>
      <div class="muted">${s.trials.length} pruebas</div>
      <div>mejor vs base: ${signed(s.best_delta)} <span class="muted">pts ${esc(summary.metric)}</span></div>
    </a>`).join("")}</div>`;
}

function trialsTable(summary, suite) {
  const datasets = summary.datasets;
  const max = Math.max(...suite.trials.map((t) => t.headline || 0), 0.0001);
  const filter = (pageTrials.filter || "").toLowerCase();
  const rows = suite.trials.filter((t) => !filter ||
    `${t.name} ${JSON.stringify(t.params)}`.toLowerCase().includes(filter));
  const columns = [
    { key: "name", label: "Prueba", html: (t) => `${esc(t.name)}${t.id === suite.baseline_id ? ' <span class="tag">base</span>' : ""}${t.id === suite.best_id ? ' <span class="tag">mejor</span>' : ""}<br>${t.id === suite.baseline_id ? params(t.params) : params(t.changed)}` },
    { key: "headline", label: `${summary.metric} medio`, num: true, html: (t) => `${num(t.headline)}${bar(t.headline, max)}` },
    { key: "delta", label: "Δ pts", num: true, html: (t) => signed(t.delta) },
    ...datasets.flatMap((d) => [
      { key: `${d}:r1`, label: `${d} R@1`, num: true, sort: (t) => t.metrics[d] && t.metrics[d]["recall@1"], html: (t) => num(t.metrics[d] && t.metrics[d]["recall@1"]) },
      { key: `${d}:r5`, label: `${d} R@5`, num: true, sort: (t) => t.metrics[d] && t.metrics[d]["recall@5"], html: (t) => num(t.metrics[d] && t.metrics[d]["recall@5"]) },
      { key: `${d}:mrr`, label: `${d} MRR`, num: true, sort: (t) => t.metrics[d] && t.metrics[d].mrr, html: (t) => num(t.metrics[d] && t.metrics[d].mrr) },
    ]),
    { key: "search_seconds_per_100", label: "s/100 búsquedas", num: true, html: (t) => num(t.search_seconds_per_100, 2) },
    { key: "index_seconds", label: "índice s", num: true, html: (t) => num(t.index_seconds, 1) },
  ];
  return rows.length
    ? table(`trials:${suite.suite}`, columns, rows, { sortKey: "headline", rowClass: (t) => t.id === suite.baseline_id ? "baseline" : "" })
    : empty("Ninguna prueba coincide con el filtro.");
}

async function pageTrials(file, base, title, lead, suiteName) {
  const manifest = await load("data/manifest.json");
  if (!manifest.sections[file]) return render(`<h1>${esc(title)}</h1>${empty("Aún no hay resultados publicados.")}`);
  const summary = await load(`data/${file}.json`);
  if (!suiteName) {
    return render(`<h1>${esc(title)}</h1><p class="lead">${lead}</p>${suiteCards(summary, base)}`);
  }
  const suite = summary.suites.find((s) => s.suite === suiteName);
  if (!suite) return render(`<h1>${esc(title)}</h1>${empty(`No existe la suite «${suiteName}».`)}`);
  render(`
    <p><a href="${base}">← ${esc(title)}</a></p>
    <h1>${esc(suite.suite)}</h1>
    <p class="lead">${suite.trials.length} pruebas. Δ = puntos de ${esc(summary.metric)} medio frente a la base
      (${suite.baseline == null ? "sin base" : num(suite.baseline)}). Pulsa una cabecera para ordenar.</p>
    <div class="toolbar"><input type="search" id="filter" placeholder="Filtrar por nombre o parámetro" value="${esc(pageTrials.filter || "")}"></div>
    <div id="trials">${trialsTable(summary, suite)}</div>`);
  const input = document.getElementById("filter");
  input.addEventListener("input", () => {
    pageTrials.filter = input.value;
    document.getElementById("trials").innerHTML = trialsTable(summary, suite);
  });
}

async function pageBeir() {
  const manifest = await load("data/manifest.json");
  if (!manifest.sections.beir) return render(`<h1>Benchmarks</h1>${empty("Aún no hay resultados publicados.")}`);
  const beir = await load("data/beir.json");
  const max = Math.max(...beir.averages.map((a) => a.ndcg10 || 0), 0.0001);
  const avg = table("beir:avg", [
    { key: "config", label: "Configuración" },
    { key: "ndcg10", label: "nDCG@10 medio", num: true, html: (a) => `${num(a.ndcg10)}${bar(a.ndcg10, max)}` },
    { key: "datasets", label: "datasets", num: true },
  ], beir.averages, { sortKey: "ndcg10" });
  const sections = beir.datasets.map((d) => {
    const top = Math.max(...d.rows.map((r) => r.ndcg10 || 0), 0.0001);
    const first = d.rows[0] || {};
    return `<h3>${esc(d.dataset)} <span class="muted">· ${esc(first.documents ?? "?")} documentos · ${esc(first.queries ?? "?")} consultas</span></h3>` +
      table(`beir:${d.dataset}`, [
        { key: "config", label: "Configuración", html: (r) => `${esc(r.config)}<br>${params(r.params)}` },
        { key: "ndcg10", label: "nDCG@10", num: true, html: (r) => `${num(r.ndcg10)}${bar(r.ndcg10, top)}` },
        { key: "recall10", label: "R@10", num: true, html: (r) => num(r.recall10) },
        { key: "recall100", label: "R@100", num: true, html: (r) => num(r.recall100) },
        { key: "search_seconds", label: "búsqueda s", num: true, html: (r) => num(r.search_seconds, 2) },
      ], d.rows, { sortKey: "ndcg10" });
  }).join("");
  render(`<h1>Benchmarks externos</h1>
    <p class="lead">${beir.row_count} ejecuciones sobre datasets públicos (BEIR y CoIR). La media de cada
      configuración solo cubre los datasets donde se ejecutó: mira la columna «datasets».</p>
    <h2>Media por configuración</h2>${avg}<h2>Por dataset</h2>${sections}`);
}

async function pageHardware() {
  const manifest = await load("data/manifest.json");
  if (!manifest.sections.hardware) {
    return render(`<h1>Hardware</h1>${empty("Sin mediciones. Ejecuta `python -m src system --benchmark` en la estación de trabajo.")}`);
  }
  const hw = await load("data/hardware.json");
  const run = hw.latest;
  const maxTok = Math.max(...run.results.map((r) => r.tokens_per_second || 0), 0.0001);
  const maxEmb = Math.max(...run.results.map((r) => r.embed_seconds || 0), 0.0001);
  const latest = table("hw:latest", [
    { key: "name", label: "Plan", sort: (r) => r.plan.name, html: (r) => `${esc(r.plan.name)} <span class="tag">${esc(r.plan.device)}</span><br><span class="params">${r.plan.threads} hilos${r.plan.cpus.length ? ` · CPUs ${esc(r.plan.cpus.join(","))}` : ""}</span>` },
    { key: "tokens_per_second", label: "generación tok/s", num: true, html: (r) => `${num(r.tokens_per_second, 1)}${bar(r.tokens_per_second, maxTok)}` },
    { key: "embed_seconds", label: "embeddings s", num: true, html: (r) => `${num(r.embed_seconds, 2)}${bar(r.embed_seconds, maxEmb)}` },
    { key: "cost", label: "coste s", num: true, html: (r) => num(r.cost, 2) },
  ], run.results, { sortKey: "tokens_per_second" });
  const history = table("hw:history", [
    { key: "created", label: "Fecha" },
    { key: "gpus", label: "GPU", sort: (r) => (r.gpus[0] || {}).name, html: (r) => esc(r.gpus.map((g) => g.name).join(", ") || "ninguna") },
    { key: "torch", label: "PyTorch" },
    { key: "gpu_speedup", label: "GPU / CPU", num: true, html: (r) => r.gpu_speedup ? `${num(r.gpu_speedup, 1)}×` : "—" },
  ], hw.runs, { sortKey: "created" });
  render(`<h1>Hardware</h1>
    <p class="lead">Las pruebas se calculan en la estación de trabajo. Cada <code>system --benchmark</code> mide
      los planes candidatos con los modelos reales (MiniLM para embeddings, Qwen3 para generar) y se guarda en el historial.</p>
    <div class="grid">
      <div class="card kpi"><div class="value">${run.gpu_speedup ? `${num(run.gpu_speedup, 1)}×` : "—"}</div><div class="label">GPU frente a la mejor CPU (coste total)</div></div>
      <div class="card kpi"><div class="value">${esc(run.gpus.map((g) => g.name).join(", ") || "sin GPU")}</div><div class="label">${esc(run.signature)}</div></div>
      <div class="card kpi"><div class="value">${esc(run.torch || "—")}</div><div class="label">PyTorch · ${esc(run.created)}</div></div>
    </div>
    <h2>Última medición</h2>${latest}<h2>Historial</h2>${history}`);
}

async function pageSkills(rest) {
  if (rest[0] === "rubric") return pageRubric(rest.slice(1).join("/"));
  const manifest = await load("data/manifest.json");
  if (rest[0]) {
    return pageTrials("skills", "#/skills", "Soft skills: laboratorio",
      "", decodeURIComponent(rest[0]));
  }
  const rubrics = await load("data/rubrics.json");
  const skills = manifest.sections.skills ? await load("data/skills.json") : null;
  render(`<h1>Soft skills</h1>
    <p class="lead">Analizador de competencias sobre el mismo núcleo RAG. <strong>Aviso:</strong> los datos de ejemplo son
      sintéticos y la rúbrica no la ha revisado aún ningún experto, así que estas cifras validan la herramienta, no el método.</p>
    <h2>Rúbricas</h2>
    <div class="grid">${rubrics.map((r) => `
      <a class="card" href="#/skills/rubric/${encodeURIComponent(r.path)}">
        <strong>${esc(r.name)}</strong> <span class="tag">${esc(r.language)}</span>
        <div class="muted">${r.competencies} competencias · ${r.indicators} indicadores (${r.positive}+ / ${r.negative}−)</div>
      </a>`).join("") || empty("Sin rúbricas.")}</div>
    <h2>Laboratorio de recuperación de evidencias</h2>
    ${skills ? suiteCards(skills, "#/skills") : empty("Aún no hay resultados publicados.")}`);
}

async function pageRubric(path) {
  const rubric = await load(`raw/${decodeURIComponent(path)}`);
  const comps = rubric.competencies.map((c) => `
    <details class="comp"><summary>${esc(c.id)} · ${esc(c.name)}</summary>
      <p>${esc(c.definition)}</p>
      <ul>${(c.indicators || []).map((i) => `<li><span class="${i.polarity === "positive" ? "pos" : "neg"}">${i.polarity === "positive" ? "+" : "−"}</span>
        <span class="params">${esc(i.id)}</span> ${esc(i.text)}</li>`).join("")}</ul>
    </details>`).join("");
  render(`<p><a href="#/skills">← Soft skills</a></p><h1>${esc(rubric.name)}</h1>
    <p class="lead">Versión ${esc(rubric.version)} · idioma ${esc(rubric.language)} · escala ${esc(JSON.stringify(rubric.scale))}</p>${comps}`);
}

async function pageDocs(path) {
  const manifest = await load("data/manifest.json");
  const all = manifest.docs.flatMap((g) => g.items);
  const current = path || (all[0] && all[0].path);
  const side = manifest.docs.map((g) => `<h4>${esc(g.title)}</h4>${g.items.map((d) =>
    `<a href="#/docs/${encodeURI(d.path)}" class="${d.path === current ? "active" : ""}">${esc(d.title)}</a>`).join("")}`).join("");
  let body = empty("Sin documentos.");
  if (current) {
    if (!all.some((d) => d.path === current)) {
      body = empty(`El documento «${current}» no está publicado.`);
    } else if (!window.marked || !window.DOMPurify) {
      body = `<pre>${esc(await load(`docs/${current}`, "text"))}</pre>`;
    } else {
      const html = window.DOMPurify.sanitize(window.marked.parse(await load(`docs/${current}`, "text")));
      body = rewriteLinks(html, current, all);
    }
  }
  render(`<div class="docs"><aside>${side}</aside><article class="md">${body}</article></div>`);
}

// Links between Markdown files become links inside the site; links to
// links to other repository files are left as they are.
function rewriteLinks(html, current, all) {
  const box = document.createElement("div");
  box.innerHTML = html;
  const known = new Set(all.map((d) => d.path));
  const dir = current.includes("/") ? current.slice(0, current.lastIndexOf("/") + 1) : "";
  box.querySelectorAll("a[href]").forEach((a) => {
    const href = a.getAttribute("href");
    if (/^[a-z]+:|^#|^\//i.test(href)) return;
    const target = normalize(dir + href.split("#")[0]);
    if (known.has(target)) a.setAttribute("href", `#/docs/${encodeURI(target)}`);
  });
  return box.innerHTML;
}

function normalize(path) {
  const out = [];
  for (const part of path.split("/")) {
    if (part === "..") out.pop();
    else if (part && part !== ".") out.push(part);
  }
  return out.join("/");
}

// ----------------------------------------------------------------- router

async function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/");
  const [section, ...rest] = parts;
  document.querySelectorAll("nav a").forEach((a) =>
    a.classList.toggle("active", a.dataset.route === (section || "")));
  try {
    if (!section) await pageHome();
    else if (section === "lab") {
      await pageTrials("lab", "#/lab", "Laboratorio",
        "Cada suite cambia uno o varios parámetros de la recuperación frente a una base y se mide con las preguntas públicas de la moulinette (docs y código).",
        rest[0] && decodeURIComponent(rest[0]));
    } else if (section === "beir") await pageBeir();
    else if (section === "skills") await pageSkills(rest);
    else if (section === "hardware") await pageHardware();
    else if (section === "docs") await pageDocs(decodeURI(rest.join("/")));
    else render(empty("Página no encontrada."));
  } catch (err) {
    render(`<h1>Error</h1><p class="error">${esc(err.message)}</p>`);
  }
}

async function footer() {
  try {
    const m = await load("data/manifest.json");
    document.getElementById("footer").innerHTML =
      `Generado ${esc(m.generated)}${m.commit ? ` · commit <code>${esc(m.commit)}</code>` : ""} · datos crudos: ${m.raw.map((p) =>
        `<a href="raw/${encodeURI(p)}">${esc(p.split("/").pop())}</a>`).join(", ")}`;
  } catch (err) { /* the page itself shows the error */ }
}

document.querySelector(".menu").addEventListener("click", (e) => {
  const nav = document.getElementById("nav");
  nav.classList.toggle("open");
  e.currentTarget.setAttribute("aria-expanded", nav.classList.contains("open"));
});
document.getElementById("nav").addEventListener("click", () =>
  document.getElementById("nav").classList.remove("open"));
window.addEventListener("hashchange", () => { route(); window.scrollTo(0, 0); });
window.addEventListener("DOMContentLoaded", () => { route(); footer(); });
