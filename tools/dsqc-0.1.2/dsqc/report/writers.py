import csv
import html
import json
import os
from collections import Counter

from ..model import SEVERITIES

SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_outputs(out, ds, results, run_info, thumbs=None):
    os.makedirs(os.path.join(out, "tables"), exist_ok=True)
    findings = [f for r in results for f in r.findings]
    for r in results:
        for tname, (fields, rows) in r.tables.items():
            if r.name == "c07_split_version" and tname == "manifest":
                write_csv(os.path.join(out, "manifest.csv"), fields, rows)
            else:
                write_csv(os.path.join(out, "tables", f"{r.name}__{tname}.csv"), fields, rows)
    write_csv(os.path.join(out, "findings.csv"),
              ["id", "check", "code", "severity", "n_images", "images", "message", "data"],
              [{"id": f.id, "check": f.check, "code": f.code, "severity": f.severity, "n_images": len(f.images),
                "images": "|".join(f.images), "message": f.message,
                "data": json.dumps(f.data, ensure_ascii=False, default=str)} for f in findings])
    summary = {
        "dataset": ds.name, "root": ds.root, "layout": ds.layout, "split_mode": ds.split_mode,
        "primary_source": ds.primary, "sources": ds.sources,
        "checks": {r.name: {"title": r.title, "stats": r.stats, "notes": r.notes,
                            "findings_by_code": dict(Counter(f.code for f in r.findings)),
                            "findings_by_severity": dict(Counter(f.severity for f in r.findings))}
                   for r in results},
        "findings_total": len(findings),
        "findings_by_severity": dict(Counter(f.severity for f in findings)),
    }
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)
    write_html(os.path.join(out, "report.html"), ds, results, run_info, summary, thumbs or {})
    return summary


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e3e6;--card:#f7f7f8;--high:#c0392b;--medium:#b9770e;--low:#2e6da4;--info:#6b6b70}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17171a;--fg:#ececef;--mut:#9a9aa2;--line:#33333a;--card:#1f1f23;--high:#ff6b5e;--medium:#f0b44c;--low:#6aa8e8;--info:#9a9aa2}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#ececef;--mut:#9a9aa2;--line:#33333a;--card:#1f1f23;--high:#ff6b5e;--medium:#f0b44c;--low:#6aa8e8;--info:#9a9aa2}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;margin:0;padding:24px 16px;max-width:1200px;margin:auto}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:32px 0 8px;padding-top:8px;border-top:1px solid var(--line)}
.mut{color:var(--mut)} table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top} th{color:var(--mut);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums} .wrap{overflow-x:auto}
.sev{font-weight:600} .high{color:var(--high)} .medium{color:var(--medium)} .low{color:var(--low)} .info{color:var(--info)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;margin:8px 0}
img.th{max-width:100%;height:auto;display:block;margin:4px 0}
code{font-size:12px}
"""


def _e(x):
    return html.escape(str(x))


def write_html(path, ds, results, run_info, summary, thumbs):
    cfg = run_info.get("config", {})
    max_rows = int(cfg.get("max_rows_html", 200))
    p = [f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>dsqc report · {_e(ds.name)}</title><style>{CSS}</style></head><body>"]
    p.append(f"<h1>Dataset quality report · {_e(ds.name)}</h1>")
    fp = summary["checks"].get("c07_split_version", {}).get("stats", {}).get("fingerprint_sha256", "")
    p.append(f"<div class='mut'>root <code>{_e(ds.root)}</code> · layout {_e(ds.layout)} · split {_e(ds.split_mode)} · "
             f"primary source {_e(ds.primary)} · dsqc {_e(run_info.get('dsqc_version'))} · {_e(run_info.get('finished'))}<br>"
             f"fingerprint <code>{_e(fp)}</code></div>")
    sev = summary["findings_by_severity"]
    p.append("<div class='card'>Findings: " + " · ".join(
        f"<span class='sev {s}'>{s} {sev.get(s, 0)}</span>" for s in SEVERITIES) +
        "<br><span class='mut'>A finding marks a disagreement or an anomaly to review; it does not state which label is correct.</span></div>")
    for r in results:
        p.append(f"<h2>{_e(r.name)} — {_e(r.title)}</h2>")
        if r.stats and r.name != "c00_inventory":
            p.append("<div class='wrap'><table><tr><th>statistic</th><th>value</th></tr>" + "".join(
                f"<tr><td>{_e(k)}</td><td class='n'>{_e(v)}</td></tr>" for k, v in r.stats.items()) + "</table></div>")
        if r.name == "c00_inventory" and "inventory" in r.tables:
            p.append("<div class='wrap'><table><tr><th>level</th><th>count</th><th>definition</th></tr>" + "".join(
                f"<tr><td>{_e(x['layer'])}</td><td class='n'>{_e(x['count'])}</td><td class='mut'>{_e(x['definition'])}</td></tr>"
                for x in r.tables["inventory"][1]) + "</table></div>")
        for n in r.notes:
            p.append(f"<div class='mut'>note: {_e(n)}</div>")
        fs = sorted(r.findings, key=lambda f: (SEV_RANK.get(f.severity, 9), f.code, f.images[:1]))
        if fs:
            shown = fs[:max_rows]
            p.append(f"<div class='mut'>{len(fs)} findings" + (f", first {len(shown)} shown (all in findings.csv)" if len(fs) > len(shown) else "") + "</div>")
            p.append("<div class='wrap'><table><tr><th>severity</th><th>code</th><th>images</th><th>message</th></tr>")
            for f in shown:
                img = thumbs.get(f.id)
                ims = "<br>".join(_e(x) for x in f.images[:6]) + (f"<br>… +{len(f.images) - 6}" if len(f.images) > 6 else "")
                p.append(f"<tr><td class='sev {f.severity}'>{f.severity}</td><td><code>{_e(f.code)}</code></td>"
                         f"<td><code>{ims}</code></td><td>{_e(f.message)}"
                         + (f"<img class='th' loading='lazy' src='{_e(img)}' alt='{_e(f.code)}'>" if img else "") + "</td></tr>")
            p.append("</table></div>")
    p.append("<h2>Run information</h2><div class='wrap'><table>" + "".join(
        f"<tr><td>{_e(k)}</td><td><code>{_e(json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v)}</code></td></tr>"
        for k, v in run_info.items()) + "</table></div></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(p))
