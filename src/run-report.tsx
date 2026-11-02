import React, { useMemo, useState } from 'react';
import { Activity, Crosshair, Flag, Gauge, Radar, Route, Table2, X } from 'lucide-react';
import { ROBOTS, type Replay } from './replay';
import { BODY_RADIUS, computeRunMetrics, mapBounds, NEAR_MISS_M, type Phase, type RunMetrics } from './run-metrics';
import './run-report.css';

/** Phase colours: categorical (fixed order) for navigate / manipulate, status red for blocked, muted for idle. Validated for the dark surface. */
const PHASE: Record<Phase['kind'], { color: string; label: string }> = { navigate: { color: '#3987e5', label: 'Navigate' }, manipulate: { color: '#d95926', label: 'Manipulate' }, blocked: { color: '#e66767', label: 'Blocked' }, idle: { color: '#6f7a76', label: 'Idle' } };
const STATUS = { done: { color: '#1baf7a', mark: '✓', label: 'Done' }, failed: { color: '#e66767', mark: '✗', label: 'Failed' }, skipped: { color: '#6f7a76', mark: '–', label: 'Skipped' } } as const;
const fmt = (v: number, d = 2) => Number.isFinite(v) ? v.toFixed(d) : '–';
const deg = (r: number) => `${(r * 180 / Math.PI).toFixed(0)}°`;

function Tile({ label, value, unit, hint, tone }: { label: string; value: string; unit?: string; hint?: string; tone?: 'good' | 'warn' | 'bad' }) {
  return <div className={`rr-tile ${tone ?? ''}`}><span className="rr-tile-label">{label}</span><span className="rr-tile-value">{value}{unit && <small>{unit}</small>}</span>{hint && <span className="rr-tile-hint">{hint}</span>}</div>;
}
function ScoreRing({ score, status }: { score: number; status: RunMetrics['status'] }) {
  const r = 46, c = 2 * Math.PI * r, color = status === 'completed' ? STATUS.done.color : status === 'failed' ? STATUS.failed.color : '#eda100';
  return <svg className="rr-ring" viewBox="0 0 120 120" role="img" aria-label={`Run score ${score} of 100`}>
    <circle cx="60" cy="60" r={r} fill="none" stroke="rgba(255,255,255,.08)" strokeWidth="9" />
    <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="9" strokeLinecap="round" strokeDasharray={`${c * score / 100} ${c}`} transform="rotate(-90 60 60)" />
    <text x="60" y="58" textAnchor="middle" className="rr-ring-value">{score}</text><text x="60" y="76" textAnchor="middle" className="rr-ring-label">RUN SCORE</text>
  </svg>;
}
/** Shared time axis: phase lanes, step outcomes and event ticks. */
function Timeline({ m, hover, setHover }: { m: RunMetrics; hover: number | null; setHover: (t: number | null) => void }) {
  const W = 640, H = 92, L = 8, R = 8, x = (t: number) => L + (t / m.duration) * (W - L - R);
  const events = (m as unknown as { events?: { time: number; label: string }[] }).events ?? [];
  return <svg viewBox={`0 0 ${W} ${H}`} className="rr-timeline" onMouseMove={e => { const b = e.currentTarget.getBoundingClientRect(); setHover(Math.max(0, Math.min(m.duration, ((e.clientX - b.left) / b.width * W - L) / (W - L - R) * m.duration))); }} onMouseLeave={() => setHover(null)} role="img" aria-label="Phase timeline">
    <defs><pattern id="rr-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" fill={PHASE.blocked.color} /><line x1="0" y1="0" x2="0" y2="6" stroke="#0f1715" strokeWidth="2" /></pattern></defs>
    <text x={L} y={12} className="rr-axis-label">PHASES</text>
    {m.phases.map((p, i) => <g key={i}><rect x={x(p.start) + 1} y={16} width={Math.max(1, x(p.end) - x(p.start) - 2)} height={18} rx="3" fill={p.kind === 'blocked' ? 'url(#rr-hatch)' : PHASE[p.kind].color} opacity=".9"><title>{`${p.label} · ${p.start.toFixed(1)}–${p.end.toFixed(1)} s`}</title></rect></g>)}
    <text x={L} y={50} className="rr-axis-label">STEPS</text>
    {m.steps.map((s, i) => s.start !== undefined && s.end !== undefined ? <g key={i}><rect x={x(s.start) + 1} y={54} width={Math.max(2, x(s.end) - x(s.start) - 2)} height={10} rx="2" fill={STATUS[s.status].color} opacity=".85"><title>{`${s.name} · ${STATUS[s.status].label}`}</title></rect></g> : null)}
    {events.map((e, i) => <line key={i} x1={x(e.time)} x2={x(e.time)} y1={68} y2={76} stroke="#c3c2b7" strokeWidth="1" />)}
    {Array.from({ length: 6 }, (_, i) => i * m.duration / 5).map((t, i) => <text key={i} x={x(t)} y={88} textAnchor={i === 0 ? 'start' : i === 5 ? 'end' : 'middle'} className="rr-tick">{t.toFixed(0)}s</text>)}
    {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={14} y2={78} stroke="#f0efec" strokeWidth="1" strokeDasharray="3 2" />}
  </svg>;
}
/** One-series line chart with a threshold band and a crosshair tooltip driven by the shared hover time. */
function Series({ m, field, title, unit, threshold, color, hover, setHover, baseline }: { m: RunMetrics; field: 'clearance' | 'speed' | 'arm'; title: string; unit: string; threshold?: number; color: string; hover: number | null; setHover: (t: number | null) => void; baseline?: number }) {
  const W = 640, H = 120, L = 40, R = 8, T = 18, B = 20;
  const values = m.samples.map(s => s[field]).filter(Number.isFinite);
  if (!values.length) return <div className="rr-chart rr-empty"><span>{title}</span><p>No reviewed geometry in this rollout.</p></div>;
  const lo = Math.min(baseline ?? 0, ...values, threshold ?? Infinity), hi = Math.max(...values, threshold ?? -Infinity, .05);
  const x = (t: number) => L + (t / m.duration) * (W - L - R), y = (v: number) => T + (1 - (v - lo) / (hi - lo || 1)) * (H - T - B);
  const path = m.samples.filter(s => Number.isFinite(s[field])).map((s, i) => `${i ? 'L' : 'M'}${x(s.t).toFixed(1)},${y(s[field]).toFixed(1)}`).join(' ');
  const idx = hover === null ? null : Math.round(hover / m.duration * (m.samples.length - 1)), at = idx === null ? null : m.samples[idx];
  const ticks = [lo, (lo + hi) / 2, hi];
  return <div className="rr-chart">
    <div className="rr-chart-head"><span>{title}</span>{at && Number.isFinite(at[field]) && <b style={{ color }}>{fmt(at[field])} {unit} <small>@ {at.t.toFixed(1)} s</small></b>}</div>
    <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={e => { const b = e.currentTarget.getBoundingClientRect(); setHover(Math.max(0, Math.min(m.duration, ((e.clientX - b.left) / b.width * W - L) / (W - L - R) * m.duration))); }} onMouseLeave={() => setHover(null)} role="img" aria-label={`${title} over time`}>
      {ticks.map((v, i) => <g key={i}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="rgba(255,255,255,.07)" /><text x={L - 6} y={y(v) + 3} textAnchor="end" className="rr-tick">{fmt(v, 1)}</text></g>)}
      {threshold !== undefined && <g><rect x={L} y={y(threshold)} width={W - L - R} height={Math.max(0, y(lo) - y(threshold))} fill={STATUS.failed.color} opacity=".12" /><line x1={L} x2={W - R} y1={y(threshold)} y2={y(threshold)} stroke={STATUS.failed.color} strokeDasharray="4 3" strokeWidth="1" /><text x={W - R - 4} y={y(threshold) - 4} textAnchor="end" className="rr-tick" fill={STATUS.failed.color}>near-miss {threshold} {unit}</text></g>}
      <path d={path} fill="none" stroke={color} strokeWidth="2" strokeLinejoin="round" />
      {at && Number.isFinite(at[field]) && <g><line x1={x(at.t)} x2={x(at.t)} y1={T} y2={H - B} stroke="#f0efec" strokeWidth="1" strokeDasharray="3 2" /><circle cx={x(at.t)} cy={y(at[field])} r="4" fill={color} stroke="#0f1715" strokeWidth="2" /></g>}
      <text x={L} y={H - 6} className="rr-tick">0s</text><text x={W - R} y={H - 6} textAnchor="end" className="rr-tick">{m.duration.toFixed(0)}s</text>
    </svg>
  </div>;
}
/** Top-down map: reviewed landmark boxes, the route coloured by phase, start/end and the first blocked point. */
function RouteMap({ m, hover }: { m: RunMetrics; hover: number | null }) {
  const b = mapBounds(m), W = 320, H = 300, sx = (W - 20) / (b.maxX - b.minX), sy = (H - 20) / (b.maxY - b.minY), s = Math.min(sx, sy);
  const X = (x: number) => 10 + (x - b.minX) * s + ((W - 20) - (b.maxX - b.minX) * s) / 2, Y = (y: number) => H - 10 - (y - b.minY) * s - ((H - 20) - (b.maxY - b.minY) * s) / 2;
  const idx = hover === null ? null : Math.round(hover / m.duration * (m.samples.length - 1)), at = idx === null ? null : m.samples[idx];
  return <svg viewBox={`0 0 ${W} ${H}`} className="rr-map" role="img" aria-label="Top-down route map">
    {m.landmarks.filter(l => l.kind !== 'tall' && l.kind !== 'door').map(l => <g key={l.id}><rect x={X(l.position[0] - l.size[0] / 2)} y={Y(l.position[1] + l.size[1] / 2)} width={l.size[0] * s} height={l.size[1] * s} fill="rgba(255,255,255,.05)" stroke="rgba(255,255,255,.28)" strokeWidth="1" rx="2" /><text x={X(l.position[0])} y={Y(l.position[1]) + 3} textAnchor="middle" className="rr-map-label">{l.label.split(' (')[0]}</text></g>)}
    {m.landmarks.filter(l => l.kind === 'door').map(l => <rect key={l.id} x={X(l.position[0] - l.size[0] / 2)} y={Y(l.position[1] + l.size[1] / 2)} width={l.size[0] * s} height={l.size[1] * s} fill="none" stroke={STATUS.done.color} strokeDasharray="3 2" rx="2" />)}
    {m.phases.map((p, i) => { const pts = m.samples.filter(q => q.t >= p.start && q.t <= p.end); return pts.length > 1 ? <polyline key={i} points={pts.map(q => `${X(q.x).toFixed(1)},${Y(q.y).toFixed(1)}`).join(' ')} fill="none" stroke={PHASE[p.kind].color} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" /> : null; })}
    <circle cx={X(m.samples[0].x)} cy={Y(m.samples[0].y)} r="5" fill="#0f1715" stroke={STATUS.done.color} strokeWidth="2" />
    <circle cx={X(m.samples.at(-1)!.x)} cy={Y(m.samples.at(-1)!.y)} r="5" fill={m.status === 'completed' ? STATUS.done.color : '#eda100'} stroke="#0f1715" strokeWidth="2" />
    {m.firstFailure && <g><circle cx={X(m.firstFailure.position[0])} cy={Y(m.firstFailure.position[1])} r="9" fill="none" stroke={STATUS.failed.color} strokeWidth="2" /><text x={X(m.firstFailure.position[0])} y={Y(m.firstFailure.position[1]) - 13} textAnchor="middle" className="rr-map-label" fill={STATUS.failed.color}>blocked</text></g>}
    {at && <circle cx={X(at.x)} cy={Y(at.y)} r={Math.max(4, BODY_RADIUS[m.robot] * s)} fill="rgba(240,239,236,.18)" stroke="#f0efec" strokeWidth="1.5" />}
  </svg>;
}
function RadarChart({ m }: { m: RunMetrics }) {
  const W = 300, H = 280, cx = W / 2, cy = H / 2 + 6, R = 96, n = m.capabilities.length;
  const pt = (i: number, r: number) => { const a = -Math.PI / 2 + i * 2 * Math.PI / n; return [cx + Math.cos(a) * r, cy + Math.sin(a) * r] as const; };
  const poly = m.capabilities.map((c, i) => pt(i, R * Math.max(.04, c.score)).join(',')).join(' ');
  return <svg viewBox={`0 0 ${W} ${H}`} className="rr-radar" role="img" aria-label="Capability profile">
    {[.25, .5, .75, 1].map(f => <polygon key={f} points={m.capabilities.map((_, i) => pt(i, R * f).join(',')).join(' ')} fill="none" stroke="rgba(255,255,255,.1)" />)}
    {m.capabilities.map((c, i) => { const [x, y] = pt(i, R); const [lx, ly] = pt(i, R + 22); return <g key={c.axis}><line x1={cx} y1={cy} x2={x} y2={y} stroke="rgba(255,255,255,.1)" /><text x={lx} y={ly + 4} textAnchor="middle" className="rr-radar-label"><title>{c.detail}</title>{c.axis}</text></g>; })}
    <polygon points={poly} fill="rgba(27,175,122,.22)" stroke={STATUS.done.color} strokeWidth="2" strokeLinejoin="round" />
    {m.capabilities.map((c, i) => { const [x, y] = pt(i, R * Math.max(.04, c.score)); return <circle key={c.axis} cx={x} cy={y} r="4" fill={STATUS.done.color} stroke="#0f1715" strokeWidth="2"><title>{`${c.axis}: ${(c.score * 100).toFixed(0)} · ${c.detail}`}</title></circle>; })}
  </svg>;
}
export function RunReport({ replay, onClose }: { replay: Replay; onClose: () => void }) {
  const m = useMemo(() => { const r = computeRunMetrics(replay); return Object.assign(r, { events: replay.events ?? [] }); }, [replay]);
  const [hover, setHover] = useState<number | null>(null), [table, setTable] = useState(false);
  const robot = ROBOTS.find(r => r.id === replay.robot)!, clearanceTone = m.minClearance === null ? undefined : m.minClearance < NEAR_MISS_M ? 'bad' : m.minClearance < .25 ? 'warn' : 'good';
  return <div className="rr-root" data-status={m.status}>
    <header className="rr-head">
      <ScoreRing score={m.score} status={m.status} />
      <div className="rr-title">
        <span className="rr-eyebrow"><Activity size={13} /> POST-RUN TELEMETRY · {m.planner.toUpperCase()} PLANNER · {replay.source === 'demo' ? 'ILLUSTRATIVE' : 'REVIEWED GEOMETRY'}</span>
        <h3>{robot.name} <span className={`rr-pill ${m.status}`}>{m.status}</span></h3>
        <p>{m.task ? `“${m.task}”` : 'Simulation replay'}{m.summary ? ` — ${m.summary}` : ''}</p>
        <div className="rr-legend">{(Object.keys(PHASE) as Phase['kind'][]).map(k => <span key={k}><i style={{ background: PHASE[k].color }} />{PHASE[k].label}</span>)}<span><i style={{ background: STATUS.done.color }} />Step done</span><span><i style={{ background: STATUS.failed.color }} />Step failed</span></div>
      </div>
      <div className="rr-actions"><button className="rr-button" onClick={() => setTable(v => !v)} aria-pressed={table}><Table2 size={14} />{table ? 'Charts' : 'Data table'}</button><button className="rr-button rr-close" onClick={onClose} aria-label="Close run report"><X size={16} /></button></div>
    </header>
    <section className="rr-tiles">
      <Tile label="Steps" value={`${m.steps.filter(s => s.status === 'done').length}/${m.steps.length}`} hint={`${m.steps.filter(s => s.status === 'failed').length} failed · ${m.steps.filter(s => s.status === 'skipped').length} skipped`} tone={m.steps.some(s => s.status === 'failed') ? 'bad' : 'good'} />
      <Tile label="Path length" value={fmt(m.pathLength)} unit="m" hint={`${fmt(m.straightLine)} m start→end`} />
      <Tile label="Path efficiency" value={(m.pathEfficiency * 100).toFixed(0)} unit="%" hint="straight-line ÷ travelled per leg" tone={m.pathEfficiency < .8 ? 'warn' : 'good'} />
      <Tile label="Min clearance" value={m.minClearance === null ? '–' : (m.minClearance * 100).toFixed(0)} unit="cm" hint={`${m.nearMisses} near-miss · ${m.nearMissTime.toFixed(1)} s`} tone={clearanceTone} />
      <Tile label="Mean · peak speed" value={`${fmt(m.meanSpeed)}·${fmt(m.peakSpeed)}`} unit="m/s" hint={`${(m.movingFraction * 100).toFixed(0)}% of time moving · ${m.stops} stops`} />
      <Tile label="Heading change" value={deg(m.totalTurn)} hint="integrated |yaw rate|" />
      <Tile label="Jerk RMS" value={fmt(m.jerkRms, 2)} unit="m/s³" hint={`accel RMS ${fmt(m.accelRms)} m/s²`} tone={m.jerkRms > 6 ? 'warn' : 'good'} />
      <Tile label="Arm duty" value={(m.armDuty * 100).toFixed(0)} unit="%" hint={`peak extension ${(m.armPeak * 100).toFixed(0)}% · ${m.manipulationTime.toFixed(1)} s manipulating`} />
    </section>
    {table ? <section className="rr-table-wrap"><table className="rr-table"><thead><tr><th>t (s)</th><th>x (m)</th><th>y (m)</th><th>speed (m/s)</th><th>yaw rate (rad/s)</th><th>accel (m/s²)</th><th>arm</th><th>clearance (m)</th></tr></thead><tbody>
      {m.samples.filter((_, i) => i % 10 === 0).map(s => <tr key={s.t}><td>{s.t.toFixed(1)}</td><td>{fmt(s.x)}</td><td>{fmt(s.y)}</td><td>{fmt(s.speed)}</td><td>{fmt(s.yawRate)}</td><td>{fmt(s.accel)}</td><td>{fmt(s.arm)}</td><td>{fmt(s.clearance)}</td></tr>)}
    </tbody></table></section> : <>
      <section className="rr-grid">
        <div className="rr-panel rr-span2"><h4><Gauge size={14} /> Timeline</h4><Timeline m={m} hover={hover} setHover={setHover} />
          <Series m={m} field="clearance" title="Clearance to reviewed geometry" unit="m" threshold={NEAR_MISS_M} color="#3987e5" hover={hover} setHover={setHover} />
          <Series m={m} field="speed" title="Base speed" unit="m/s" color="#d95926" hover={hover} setHover={setHover} baseline={0} />
        </div>
        <div className="rr-panel"><h4><Radar size={14} /> Capability profile</h4><RadarChart m={m} /><ul className="rr-caps">{m.capabilities.map(c => <li key={c.axis}><span>{c.axis}</span><i><b style={{ width: `${c.score * 100}%` }} /></i><small>{c.detail}</small></li>)}</ul></div>
        <div className="rr-panel rr-feedback"><h4><Crosshair size={14} /> Feedback loop · next iteration</h4>
          <ul>{m.feedback.map((f, i) => <li key={i} className={f.kind}><span className="rr-fb-kind">{f.kind === 'fix' ? 'RE-PLAN' : f.kind === 'switch' ? 'SWITCH ROBOT' : 'PROMOTE'}</span><b>{f.title}</b><p>{f.detail}</p></li>)}</ul>
          <p className="rr-note">Verdict and step notes come from the planner; kinematics and clearance are recomputed here from the replayed trajectory at 10 Hz against the same landmark boxes the planner used.</p>
        </div>
        <div className="rr-panel rr-span2"><h4><Flag size={14} /> Step outcomes</h4><ol className="rr-steps">{m.steps.map((s, i) => <li key={i} className={s.status}><span className="rr-mark" style={{ color: STATUS[s.status].color }}>{STATUS[s.status].mark}</span><div><b>{s.name}</b>{s.start !== undefined && <small>{s.start.toFixed(1)}–{s.end!.toFixed(1)} s</small>}{s.note && <p>{s.note}</p>}</div></li>)}</ol></div>
        <div className="rr-panel rr-span2 rr-route"><h4><Route size={14} /> Route · top-down</h4><RouteMap m={m} hover={hover} /><p className="rr-note">Boxes are the reviewed collision landmarks; clearance is measured from the robot body ({(BODY_RADIUS[m.robot]*100).toFixed(0)} cm radius), not the planner's inflated footprint. Hover the timeline to place the body.</p></div>
      </section>
    </>}
  </div>;
}
