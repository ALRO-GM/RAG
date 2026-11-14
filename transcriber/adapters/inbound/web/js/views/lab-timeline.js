// Timeline of a lab run: one row per diarization track of the chosen
// denoised audio, the consensus on top and the reference labels when there
// are any; click to play from there, a cursor follows the audio.

import { fill, h, svg } from "../dom.js";
import { t } from "../i18n.js";
import { describeError } from "../errors.js";
import { formatTime } from "../lib/text.js";
import {
  allSpeakers, idPart, overlaps, scrollWindow, speakerColours, ticks, timeScale,
  visible, zoomWindow,
} from "../lib/lab.js";

const ROW = 26;
const GAP = 6;
const AXIS = 22;

/** The audio player shared by the timeline and the fragments list. */
export async function labPlayer(ctx, env) {
  let ready = true;
  try {
    await ctx.api.audioCookie();
  } catch (error) {
    if (error.kind === "unauthorized") throw error;
    ready = false;
  }
  const denoisers = env.status.request?.denoisers ?? ["none"];
  const audio = h("audio", { preload: "metadata", hidden: true });
  const error = h("p", { class: "error", role: "alert", hidden: ready }, t("player.no_audio"));
  const play = h("button", { type: "button", class: "primary" }, t("player.play"));
  const clock = h("span", { class: "time", role: "timer", "aria-live": "off" }, "0:00");
  const source = h("select", { id: "lab-audio" }, denoisers.map((d) => h("option", { value: d }, d === "none" ? t("lab.original") : d)));
  let stopAt = null;
  function use(denoiser) {
    if (!ready) return;
    const at = audio.currentTime || 0;
    audio.src = ctx.api.labAudioUrl(env.session, env.run, denoiser);
    if (at) audio.addEventListener("loadedmetadata", () => { audio.currentTime = at; }, { once: true });
  }
  source.addEventListener("change", () => use(source.value));
  play.addEventListener("click", () => (audio.paused ? start() : audio.pause()));
  audio.addEventListener("error", () => { if (audio.getAttribute("src")) error.hidden = false; });
  audio.addEventListener("play", () => { play.textContent = t("player.pause"); });
  audio.addEventListener("pause", () => { play.textContent = t("player.play"); });
  audio.addEventListener("timeupdate", () => {
    clock.textContent = formatTime(audio.currentTime);
    if (stopAt !== null && audio.currentTime >= stopAt) {
      stopAt = null;
      audio.pause();
    }
  });
  function start() {
    audio.play().catch(() => { error.hidden = false; });
  }
  use(denoisers[0]);
  return {
    audio,
    element: h("div", { class: "player" }, h("div", { class: "row wrap" }, play, clock,
      h("label", { for: "lab-audio" }, t("lab.listen")), source), error, audio),
    /** Play from `from`; stop at `until` when given. */
    playFrom(from, until = null) {
      stopAt = until;
      audio.currentTime = Math.max(0, from);
      start();
    },
    toggle() {
      if (audio.paused) start();
      else audio.pause();
    },
    stop() {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    },
  };
}

export async function timelineSection(body, env) {
  const { ctx, results } = env;
  let tracks;
  let labels = [];
  try {
    [tracks, labels] = await Promise.all([env.tracks(), ctx.api.labLabels(env.session).then((l) => l.labels).catch(() => [])]);
  } catch (error) {
    if (error.kind === "unauthorized") throw error;
    body.replaceChildren(h("p", { class: "error", role: "alert" }, describeError(error)));
    return null;
  }
  const ids = Object.keys(tracks.tracks);
  if (!ids.length) {
    body.replaceChildren(h("p", { class: "notice" }, t("lab.no_tracks")));
    return null;
  }
  const duration = results.duration || Math.max(...ids.flatMap((id) => tracks.tracks[id].map((s) => s.end)), 1);
  const denoisers = idPart(ids, 0);
  let denoiser = denoisers.includes("none") ? "none" : denoisers[0];
  let view = { start: 0, end: duration };
  const speakers = allSpeakers(tracks.tracks);
  const colours = speakerColours(speakers);
  const player = await labPlayer(ctx, env);
  const box = h("div", { class: "timeline-wrap" });
  const readout = h("p", { class: "meta", role: "status", "aria-live": "polite" });
  let cursor = null;
  let scale = null;
  let frame = null;

  function fillFor(speaker) {
    const slot = colours.get(speaker) ?? "anon-0";
    return slot.startsWith("anon") ? { class: `seg ${slot}`, fill: `url(#lab-hatch-${slot.slice(5)})` } : { class: `seg ${slot}` };
  }

  function rows() {
    const out = [{ id: "consensus", label: t("lab.consensus"), kind: "consensus", items: tracks.consensus }];
    if (labels.length) out.push({ id: "labels", label: t("lab.reference"), kind: "labels", items: labels });
    for (const id of ids.filter((x) => x.startsWith(`${denoiser}+`))) {
      out.push({ id, label: id.slice(denoiser.length + 1), kind: "track", items: tracks.tracks[id] });
    }
    return out;
  }

  function draw() {
    const width = Math.max(320, box.clientWidth || 900);
    const left = width < 600 ? 96 : 170;
    const all = rows();
    const height = AXIS + all.length * (ROW + GAP);
    scale = timeScale(view.start, view.end, width - left);
    const x = (time) => left + scale.x(time);
    const chart = svg("svg", { class: "timeline", viewBox: `0 0 ${width} ${height}`, width, height, role: "img", "aria-label": t("lab.timeline_aria") });
    const defs = svg("defs");
    for (let i = 0; i < 4; i++) {
      defs.append(svg("pattern", { id: `lab-hatch-${i}`, width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: `rotate(${45 + i * 30})` },
        svg("rect", { width: 6, height: 6, class: `hatch-bg anon-${i}` }), svg("line", { x1: 0, y1: 0, x2: 0, y2: 6, class: "hatch-line" })));
    }
    chart.append(defs);
    for (const tick of ticks(view.start, view.end, width < 600 ? 4 : 8)) {
      chart.append(svg("line", { x1: x(tick), x2: x(tick), y1: AXIS - 4, y2: height, class: "grid" }),
        svg("text", { x: x(tick), y: AXIS - 8, "text-anchor": "middle", class: "axis" }, formatTime(tick) + (view.end - view.start < 20 ? `.${Math.round((tick % 1) * 10)}` : "")));
    }
    all.forEach((row, r) => {
      const y = AXIS + r * (ROW + GAP);
      chart.append(svg("text", { x: left - 6, y: y + ROW * 0.68, "text-anchor": "end", class: `axis row-${row.kind}` }, row.label));
      chart.append(svg("rect", { x: left, y, width: width - left, height: ROW, class: "lane" }));
      const items = visible(row.items, view.start, view.end);
      for (const item of items) {
        const x0 = Math.max(left, x(item.start));
        const w = Math.max(1, Math.min(width, x(item.end)) - x0);
        if (row.kind === "track") {
          chart.append(svg("rect", { x: x0, y: y + 2, width: w, height: ROW - 4, ...fillFor(item.speaker) }, svg("title", {}, `${item.speaker} ${formatTime(item.start)}–${formatTime(item.end)}`)));
        } else if (row.kind === "consensus" || (row.kind === "labels" && (item.kind === "person" || item.kind === "both"))) {
          const who = item.speakers ?? [];
          if (!who.length) continue;
          const part = (ROW - 4) / who.length;
          who.forEach((speaker, i) => chart.append(svg("rect", {
            x: x0, y: y + 2 + i * part, width: w, height: part, ...fillFor(speaker),
          }, svg("title", {}, `${who.join(" + ")} ${formatTime(item.start)}–${formatTime(item.end)}`))));
          if (row.kind === "consensus" && item.agreed === false) chart.append(svg("rect", { x: x0, y: y + 2, width: w, height: ROW - 4, class: "tie" }));
        } else {
          chart.append(svg("rect", { x: x0, y: y + 2, width: w, height: ROW - 4, class: `seg lbl-${item.kind}` }, svg("title", {}, t(`lab.kind.${item.kind}`))));
        }
      }
      if (row.kind === "track") {
        for (const o of visible(overlaps(row.items), view.start, view.end)) {
          const x0 = Math.max(left, x(o.start));
          chart.append(svg("rect", { x: x0, y: y + ROW - 6, width: Math.max(2, Math.min(width, x(o.end)) - x0), height: 4, class: "ovl" }, svg("title", {}, t("lab.overlap"))));
        }
      }
    });
    cursor = svg("line", { x1: left, x2: left, y1: AXIS - 4, y2: height, class: "cursor" });
    chart.append(cursor);
    chart.addEventListener("click", (event) => {
      const bounds = chart.getBoundingClientRect();
      const px = ((event.clientX - bounds.left) / bounds.width) * width - left;
      if (px < 0) return;
      player.playFrom(scale.time(px));
    });
    box.replaceChildren(chart);
    moveCursor(left);
  }

  function moveCursor(left = box.clientWidth < 600 ? 96 : 170) {
    if (!cursor || !scale) return;
    const time = player.audio.currentTime || 0;
    const inside = time >= view.start && time <= view.end;
    const px = left + scale.x(time);
    cursor.setAttribute("x1", px);
    cursor.setAttribute("x2", px);
    cursor.classList.toggle("off", !inside);
    readout.textContent = t("lab.window", { from: formatTime(view.start), to: formatTime(view.end), at: formatTime(time) });
  }

  function tick() {
    moveCursor();
    const time = player.audio.currentTime;
    if (!player.audio.paused && (time > view.end || time < view.start)) {
      const span = view.end - view.start;
      view = scrollWindow({ start: time - span * 0.1, end: time + span * 0.9 }, 0, duration);
      draw();
    }
    frame = player.audio.paused ? null : requestAnimationFrame(tick);
  }
  player.audio.addEventListener("play", () => { if (frame === null) frame = requestAnimationFrame(tick); });
  player.audio.addEventListener("seeked", () => moveCursor());

  const zoom = (factor) => {
    const time = player.audio.currentTime;
    const centre = time >= view.start && time <= view.end ? time : (view.start + view.end) / 2;
    view = zoomWindow(view, factor, centre, duration, 2);
    draw();
  };
  const move = (fraction) => { view = scrollWindow(view, fraction, duration); draw(); };
  const button = (label, fn, aria) => h("button", { type: "button", onClick: fn, "aria-label": aria ?? null }, label);
  const dnSelect = h("select", { id: "lab-tl-dn" }, denoisers.map((d) => h("option", { value: d, selected: d === denoiser }, d === "none" ? t("lab.original") : d)));
  dnSelect.addEventListener("change", () => { denoiser = dnSelect.value; draw(); });

  const legend = h("ul", { class: "legend" }, speakers.map((s) => h("li", {},
    svg("svg", { width: 14, height: 14, "aria-hidden": "true" }, svg("rect", { width: 14, height: 14, ...fillFor(s) })),
    " ", s)), h("li", {}, svg("svg", { width: 14, height: 14, "aria-hidden": "true" }, svg("rect", { y: 9, width: 14, height: 4, class: "ovl" })), " ", t("lab.overlap")));

  fill(body,
    h("p", { class: "hint" }, t("lab.timeline_about")),
    player.element,
    h("div", { class: "row wrap" },
      h("label", { for: "lab-tl-dn" }, t("lab.rows_for")), dnSelect,
      button("−", () => zoom(2), t("lab.zoom_out")), button("+", () => zoom(0.5), t("lab.zoom_in")),
      button(t("lab.zoom_all"), () => { view = { start: 0, end: duration }; draw(); }),
      button("←", () => move(-0.5), t("lab.earlier")), button("→", () => move(0.5), t("lab.later"))),
    box, readout, legend);
  draw();
  const resize = () => draw();
  window.addEventListener("resize", resize);
  return () => {
    window.removeEventListener("resize", resize);
    if (frame !== null) cancelAnimationFrame(frame);
    player.stop();
  };
}
