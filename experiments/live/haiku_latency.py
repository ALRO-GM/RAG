"""Latency of Haiku through `claude -p` for live call assist (synthetic data).

Modes: A cold json one-shot, B cold stream-json one-shot (partial messages),
C persistent stream-json process (input and output). Usage:
  python haiku_latency.py [n_a n_b n_c]   -> results.json next to this file
"""
import json, os, statistics, subprocess, sys, tempfile, time
from pathlib import Path

OUT = Path(__file__).with_name("results.json")
ENV = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL") if k in os.environ}
SYSTEM = (
    "Eres un asistente en directo para un agente de atención al cliente que habla por teléfono. "
    "Recibes la transcripción reciente de la llamada y fragmentos recuperados de la base de conocimiento de la empresa. "
    "Tu tarea: sugerir la siguiente respuesta del agente, en español, en un máximo de 40 palabras, en tono cordial, "
    "profesional y empático. Reglas: 1) Usa solo datos presentes en los fragmentos; si falta información, sugiere "
    "preguntar o ofrecer escalar el caso. 2) No inventes plazos, precios ni políticas. 3) No repitas lo que el agente ya dijo. "
    "4) Si el cliente está molesto, reconoce primero la emoción y luego propone una solución concreta. 5) Si detectas una "
    "petición de baja, ofrece primero una alternativa de retención sin presionar. 6) Si el cliente da un dato que requiere "
    "verificación, pide confirmación de forma amable. 7) Devuelve únicamente el texto de la sugerencia, sin comillas, sin "
    "listas y sin explicaciones. 8) Si el interlocutor es un candidato en una entrevista de trabajo, adapta la sugerencia "
    "al entrevistador: una pregunta de seguimiento breve basada en lo último que dijo el candidato. "
)
TOPICS = [
    ("el router lleva tres días sin conectar y he reiniciado dos veces", "Cliente enfadado por avería de fibra"),
    ("quiero dar de baja la línea porque me ha subido la factura", "Cliente pide baja"),
    ("me han cobrado dos veces la cuota de marzo", "Cobro duplicado"),
    ("llevo cinco años en logística y dirigí un equipo de doce personas", "Entrevista: experiencia de liderazgo"),
    ("mi mayor defecto es que me cuesta delegar", "Entrevista: debilidades"),
    ("no me llegó el paquete y el seguimiento lleva una semana parado", "Paquete retrasado"),
    ("quiero cambiar de tarifa pero no sé cuál me conviene", "Cambio de tarifa"),
    ("dejé mi anterior empleo porque la empresa cerró", "Entrevista: motivo de salida"),
    ("la aplicación se cierra cada vez que intento pagar", "Fallo de la app"),
    ("necesito una factura con los datos de mi empresa", "Solicitud de factura"),
    ("me prometieron un descuento y no aparece en el recibo", "Descuento prometido"),
    ("he gestionado presupuestos de doscientos mil euros", "Entrevista: presupuestos"),
    ("llevo una hora esperando al técnico y nadie ha venido", "Técnico no acude"),
    ("¿pueden devolverme el dinero del pedido cancelado?", "Reembolso"),
    ("quisiera saber si puedo trabajar en remoto dos días", "Entrevista: teletrabajo"),
]
LINES = [
    "Agente: Buenas tardes, le atiende Marta, ¿en qué puedo ayudarle?",
    "Cliente: Hola, llamo por un problema que me trae de cabeza.",
    "Agente: Cuénteme, estoy aquí para ayudarle.",
    "Cliente: Es la tercera vez que llamo por esto y nadie me da una solución.",
    "Agente: Lamento las molestias, vamos a revisarlo ahora mismo.",
    "Cliente: Se lo agradezco, pero ya he perdido mucho tiempo.",
    "Agente: ¿Me confirma su nombre y el número de cliente, por favor?",
    "Cliente: Claro, es el 48213 a nombre de Carmen Soler Vidal.",
    "Agente: Gracias, un momento mientras consulto la ficha.",
]
CHUNKS = [
    "[KB-101] Averías: si el equipo no sincroniza tras reinicio, abrir incidencia técnica; la visita del técnico se agenda en franjas de mañana o tarde y puede solicitarse prioridad si hay más de 72 horas sin servicio. El cliente recibe un SMS de confirmación con la franja asignada y puede reprogramarla una vez sin coste adicional.",
    "[KB-212] Facturación: los cobros duplicados se reembolsan en un plazo de 5 a 7 días hábiles tras abrir la reclamación; el agente debe verificar los dos movimientos con fecha e importe antes de tramitarla. Los descuentos comerciales figuran en la ficha de contrato y se aplican desde el ciclo siguiente a su alta.",
    "[KB-330] Retención: ante una solicitud de baja, ofrecer revisión de tarifa o permanencia reducida sin penalización antes de tramitarla; si el cliente insiste, derivar al equipo de bajas, que contacta en 24 horas. En entrevistas internas, las preguntas de seguimiento deben centrarse en logros medibles, tamaño del equipo y resultados concretos.",
]


def user_msg(i: int) -> str:
    said, topic = TOPICS[i % len(TOPICS)]
    lines = LINES[:8] + [f"Cliente: {said}"]
    return (f"Transcripción reciente ({topic}):\n" + "\n".join(lines) +
            "\n\nConocimiento recuperado:\n" + "\n\n".join(CHUNKS) +
            "\n\nSugiere la siguiente respuesta del agente (máx. 40 palabras).")


def base_cmd(fmt: str, extra=()):
    return ["claude", "-p", "--safe-mode", "--setting-sources", "", "--strict-mcp-config",
            "--tools", "", "--disable-slash-commands", "--no-session-persistence",
            "--output-format", fmt, "--model", "haiku", "--system-prompt", SYSTEM, *extra]


def mode_a(n):
    res = []
    for i in range(n):
        with tempfile.TemporaryDirectory() as d:
            t = time.perf_counter()
            p = subprocess.run(base_cmd("json"), input=user_msg(i), capture_output=True,
                               text=True, cwd=d, env=ENV)
            wall = time.perf_counter() - t
        try:
            j = json.loads(p.stdout)
        except Exception:
            res.append({"wall_s": wall, "error": p.stdout[:200] + p.stderr[:200]}); continue
        res.append({"wall_s": wall, "duration_ms": j.get("duration_ms"),
                    "duration_api_ms": j.get("duration_api_ms"), "usage": j.get("usage"),
                    "result_words": len(str(j.get("result", "")).split())})
    return res


def text_delta(ev):
    e = ev.get("event") if ev.get("type") == "stream_event" else None
    if e and e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta":
        return True
    return False


def mode_b(n):
    res = []
    for i in range(n):
        with tempfile.TemporaryDirectory() as d:
            t = time.perf_counter()
            p = subprocess.Popen(base_cmd("stream-json", ["--verbose", "--include-partial-messages"]),
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, cwd=d, env=ENV)
            p.stdin.write(user_msg(i)); p.stdin.close()
            first = init = done = None; usage = None; api = None
            for line in p.stdout:
                now = time.perf_counter() - t
                try: ev = json.loads(line)
                except Exception: continue
                if ev.get("type") == "system" and init is None: init = now
                if first is None and text_delta(ev): first = now
                if ev.get("type") == "result":
                    done = now; usage = ev.get("usage"); api = ev.get("duration_api_ms")
            p.wait()
            res.append({"init_s": init, "first_token_s": first, "done_s": done,
                        "wall_s": time.perf_counter() - t, "duration_api_ms": api, "usage": usage})
    return res


def mode_c(n):
    with tempfile.TemporaryDirectory() as d:
        t0 = time.perf_counter()
        p = subprocess.Popen(base_cmd("stream-json", ["--input-format", "stream-json", "--verbose",
                                                      "--include-partial-messages"]),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, cwd=d, env=ENV)
        res = []; init_s = None
        for i in range(n):
            msg = {"type": "user", "message": {"role": "user", "content": user_msg(i)}}
            t = time.perf_counter()
            p.stdin.write(json.dumps(msg) + "\n"); p.stdin.flush()
            first = done = None; usage = None; api = None
            for line in p.stdout:
                now = time.perf_counter() - t
                try: ev = json.loads(line)
                except Exception: continue
                if ev.get("type") == "system" and init_s is None:
                    init_s = time.perf_counter() - t0
                if first is None and text_delta(ev): first = now
                if ev.get("type") == "result":
                    done = now; usage = ev.get("usage"); api = ev.get("duration_api_ms"); break
            res.append({"first_token_s": first, "done_s": done, "duration_api_ms": api, "usage": usage})
        p.stdin.close(); p.wait(timeout=20)
    return res, init_s


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs: return None
    q = lambda f: xs[min(len(xs) - 1, int(round(f * (len(xs) - 1))))]
    return {"n": len(xs), "median": statistics.median(xs), "p90": q(0.9), "min": xs[0], "max": xs[-1]}


if __name__ == "__main__":
    na, nb, nc = (int(x) for x in sys.argv[1:4]) if len(sys.argv) > 3 else (12, 12, 12)
    out = {"claude_version": subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()}
    out["A"] = mode_a(na); out["B"] = mode_b(nb); out["C"], out["C_init_s"] = mode_c(nc)
    out["summary"] = {
        "A_wall_s": stats([r.get("wall_s") for r in out["A"]]),
        "A_duration_ms": stats([r.get("duration_ms") for r in out["A"]]),
        "A_duration_api_ms": stats([r.get("duration_api_ms") for r in out["A"]]),
        "B_first_token_s": stats([r["first_token_s"] for r in out["B"]]),
        "B_done_s": stats([r["done_s"] for r in out["B"]]),
        "B_init_s": stats([r["init_s"] for r in out["B"]]),
        "C_first_token_s": stats([r["first_token_s"] for r in out["C"]]),
        "C_done_s": stats([r["done_s"] for r in out["C"]]),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["summary"], indent=1)); print("C start-up (to first system event):", out["C_init_s"])
    print("usage A0:", out["A"][0].get("usage"))
