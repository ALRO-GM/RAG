// DM-1 (#233): what the recording adds on top of the page: a visible
// mouse pointer that glides, a click ring, gentle scrolling and the
// title cards. Evaluated in the page over DevTools; it changes nothing
// of the app and is never served with it. Idempotent: evaluating it
// again keeps the existing elements.
(() => {
  if (window.__demo) return true;
  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  const frame = () => new Promise((done) => requestAnimationFrame(done));
  const layer = (name) => {
    const el = document.createElement('div');
    el.dataset.demo = name;
    Object.assign(el.style, { position: 'fixed', pointerEvents: 'none', zIndex: '2147483646' });
    document.documentElement.appendChild(el);
    return el;
  };

  const pointer = layer('pointer');
  Object.assign(pointer.style, { left: '0', top: '0', width: '28px', height: '28px', zIndex: '2147483647', opacity: '0', transition: 'opacity 300ms ease' });
  pointer.innerHTML = '<svg width="28" height="28" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">'
    + '<path d="M5 3 L5 22 L10 17.5 L13.6 25 L17 23.4 L13.5 16 L20 16 Z" fill="#111827" stroke="#ffffff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  const at = { x: 0, y: 0 };
  const place = () => { pointer.style.transform = `translate(${at.x - 5}px, ${at.y - 3}px)`; };

  const card = layer('card');
  Object.assign(card.style, {
    inset: '0', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
    gap: '18px', background: 'var(--bg, #f3f4f6)', color: 'var(--text, #111827)', textAlign: 'center',
    fontFamily: 'var(--font, sans-serif)', opacity: '0', transition: 'opacity 700ms ease', padding: '0 80px',
  });

  const scroller = (el) => {
    for (let node = el; node && node !== document.body; node = node.parentElement) {
      const style = getComputedStyle(node);
      if (/(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight + 2) return node;
    }
    return document.scrollingElement;
  };

  window.__demo = {
    // The centre of the smallest visible element under `scope` whose text is `text`
    // (or that matches `selector`), in CSS pixels; null when there is none.
    find(text, scope = 'main', selector = '*') {
      const root = document.querySelector(scope) || document.body;
      let best = null;
      for (const el of root.querySelectorAll(selector)) {
        const own = (el.innerText || '').trim();
        if (text !== null && own !== text) continue;
        const r = el.getBoundingClientRect();
        if (!r.width || !r.height) continue;
        if (!best || r.width * r.height < best.r.width * best.r.height) best = { el, r };
      }
      if (!best) return null;
      return { x: best.r.left + best.r.width / 2, y: best.r.top + best.r.height / 2, top: best.r.top, bottom: best.r.bottom };
    },
    show(x, y) { at.x = x; at.y = y; place(); pointer.style.opacity = '1'; return true; },
    async move(x, y, ms) {
      const from = { ...at };
      const start = performance.now();
      for (;;) {
        const t = Math.min(1, (performance.now() - start) / ms);
        at.x = from.x + (x - from.x) * ease(t);
        at.y = from.y + (y - from.y) * ease(t);
        place();
        if (t >= 1) return true;
        await frame();
      }
    },
    ring() {
      const ring = layer('ring');
      Object.assign(ring.style, { left: `${at.x - 18}px`, top: `${at.y - 18}px`, width: '36px', height: '36px', borderRadius: '50%', border: '3px solid rgba(59, 73, 160, 0.7)', zIndex: '2147483646' });
      ring.animate([{ transform: 'scale(0.4)', opacity: 1 }, { transform: 'scale(1.3)', opacity: 0 }], { duration: 650, easing: 'ease-out' }).finished.then(() => ring.remove());
      return true;
    },
    async scrollTo(el, y, ms) {
      const box = scroller(el);
      const from = box.scrollTop;
      const to = Math.max(0, Math.min(y, box.scrollHeight - box.clientHeight));
      const start = performance.now();
      for (;;) {
        const t = Math.min(1, (performance.now() - start) / ms);
        box.scrollTop = from + (to - from) * ease(t);
        if (t >= 1) return box.scrollTop;
        await frame();
      }
    },
    // A full-screen card: title, line and a small note; shown or faded out.
    card(visible, title = '', line = '', note = '') {
      if (visible) {
        card.replaceChildren();
        const parts = [[title, '52px', '650', '1'], [line, '26px', '400', '0.8'], [note, '17px', '500', '0.55']];
        for (const [text, size, weight, alpha] of parts) {
          if (!text) continue;
          const p = document.createElement('div');
          Object.assign(p.style, { fontSize: size, fontWeight: weight, opacity: alpha, letterSpacing: size === '52px' ? '-0.02em' : 'normal', maxWidth: '1100px' });
          p.textContent = text;
          card.appendChild(p);
        }
      }
      if (visible) pointer.style.opacity = '0';
      card.style.opacity = visible ? '1' : '0';
      return true;
    },
    cardNow() { card.style.transition = 'none'; card.style.opacity = '1'; void card.offsetWidth; card.style.transition = 'opacity 700ms ease'; return true; },
    colours() {
      const probe = document.createElement('div');
      probe.style.color = 'var(--text)';
      probe.style.background = 'var(--bg)';
      probe.style.border = '1px solid var(--line, #e5e6ea)';
      document.body.appendChild(probe);
      const s = getComputedStyle(probe);
      const out = { text: s.color, bg: s.backgroundColor, line: s.borderTopColor };
      probe.remove();
      return out;
    },
  };
  return true;
})();
