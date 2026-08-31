'use client';

import { useMemo, useState } from 'react';
import Image from 'next/image';

type Metric = 'Recall@10' | 'NDCG@10';

const baselineRows = [
  { model: 'Popularity', recall: 0.0219, ndcg: 0.0421, kind: 'Non-personalized' },
  { model: 'Target BPR', recall: 0.0207, ndcg: 0.0315, kind: 'Target-only' },
  { model: 'LightGCN', recall: 0.0391, ndcg: 0.0611, kind: 'Graph CF' },
  { model: 'SASRec tuned', recall: 0.0127, ndcg: 0.0194, kind: 'Sequential' },
  { model: 'EMCDR', recall: 0.0197, ndcg: 0.0346, kind: 'Cross-domain' },
  { model: 'Creator-Bridge', recall: 0.1023, ndcg: 0.1640, kind: 'Our reference' },
  { model: 'Bridge + GRPO', recall: 0.1136, ndcg: 0.1717, kind: 'Our policy' },
];

const pipeline = [
  ['01', 'Time-safe event gate', 'Source events are clipped at each user’s target-train cutoff.'],
  ['02', 'Creator-Bridge recall', 'Short-video creator affinity transfers into the live-author space.'],
  ['03', 'Group-relative policy', 'G candidate slates produce within-user relative advantages.'],
  ['04', 'Constrained reranking', 'PPO clip and reference KL bound movement from the recall model.'],
  ['05', 'Full-sort audit', 'Accuracy, cold buckets, exposure and proxy rewards stay separate.'],
];

export default function Home() {
  const [metric, setMetric] = useState<Metric>('Recall@10');
  const [section, setSection] = useState<'results' | 'method' | 'audit'>('results');
  const rows = useMemo(
    () => baselineRows.map((row) => ({ ...row, value: metric === 'Recall@10' ? row.recall : row.ndcg })),
    [metric],
  );
  const maximum = Math.max(...rows.map((row) => row.value));

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="LiveBridge home">
          <span className="brandMark">LB</span>
          <span>LiveBridge<span className="accent">—GRPO</span></span>
        </a>
        <nav aria-label="Dashboard sections">
          {(['results', 'method', 'audit'] as const).map((item) => (
            <button key={item} className={section === item ? 'navActive' : ''} onClick={() => setSection(item)}>
              {item}
            </button>
          ))}
        </nav>
        <span className="status"><i /> 3-seed verified</span>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow"><span>PUBLIC DATA</span> KuaiLive-M3 · Short Video → Live Stream</div>
        <h1>Transfer intent.<br /><em>Optimize the slate.</em></h1>
        <p className="heroCopy">
          A time-safe cross-domain recommender that bridges short-video creator interest into live-stream recall,
          then applies constrained group-relative policy optimization to a multi-step slate.
        </p>
        <div className="heroActions">
          <a className="primary" href="#benchmark">Explore benchmark</a>
          <a className="secondary" href="#audit">Read claim boundary</a>
        </div>
        <div className="signalRail" aria-label="Key project facts">
          <div><strong>+191%</strong><span>Recall@10 vs LightGCN</span></div>
          <div><strong>3 seeds</strong><span>42 · 43 · 44</span></div>
          <div><strong>Full-sort</strong><span>No sampled-negative metric</span></div>
          <div><strong>RTX 5090</strong><span>Reproducible GPU pipeline</span></div>
        </div>
      </section>

      <section className="panel benchmark" id="benchmark">
        <div className="sectionHead">
          <div>
            <span className="kicker">01 / BENCHMARK</span>
            <h2>Strong baselines, one protocol</h2>
          </div>
          <div className="segmented" role="group" aria-label="Metric selector">
            {(['Recall@10', 'NDCG@10'] as Metric[]).map((item) => (
              <button key={item} onClick={() => setMetric(item)} className={metric === item ? 'selected' : ''}>{item}</button>
            ))}
          </div>
        </div>
        <p className="sectionIntro">Mean over three independently sampled 1% user splits. Error bars are reported in the repository artifact.</p>
        <div className="chart" role="img" aria-label={`${metric} model comparison`}>
          {rows.map((row) => (
            <div className={`barRow ${row.model.includes('GRPO') ? 'ours' : ''}`} key={row.model}>
              <div className="barLabel"><strong>{row.model}</strong><span>{row.kind}</span></div>
              <div className="barTrack"><i style={{ width: `${Math.max(3, row.value / maximum * 100)}%` }} /></div>
              <code>{row.value.toFixed(4)}</code>
            </div>
          ))}
        </div>
        <div className="finding">
          <span>Primary finding</span>
          <p>Bridge + GRPO raises {metric} from <b>{metric === 'Recall@10' ? '0.0391' : '0.0611'}</b> to <b>{metric === 'Recall@10' ? '0.1136' : '0.1717'}</b> versus the strongest non-project baseline, LightGCN.</p>
        </div>
        <div className="policyStrip" aria-label="GRPO policy gains over Creator-Bridge">
          <div><span>Recall@10</span><strong>+11.0%</strong><small>vs Bridge reference</small></div>
          <div><span>NDCG@10</span><strong>+4.7%</strong><small>all 3 seeds positive</small></div>
          <div><span>Long-tail share</span><strong>+75.0%</strong><small>0.3093 → 0.5411</small></div>
          <div><span>Catalog coverage</span><strong>+6.3%</strong><small>with lower exposure Gini</small></div>
        </div>
      </section>

      <section className="methodGrid" id="method">
        <div className="sectionHead wide">
          <div><span className="kicker">02 / SYSTEM</span><h2>From event logs to a constrained policy</h2></div>
          <p>Every arrow has a matching file, command and audit field.</p>
        </div>
        <div className="pipeline">
          {pipeline.map(([number, title, copy]) => (
            <article key={number}>
              <span>{number}</span><div><h3>{title}</h3><p>{copy}</p></div>
            </article>
          ))}
        </div>
        <aside className="formulaCard">
          <span className="kicker">POLICY OBJECTIVE</span>
          <div className="formula">L = −E[min(rA, clip(r)A)] + β KL(πθ ‖ πref)</div>
          <p>The main policy combines discounted logged relevance, watch-duration proxy, source affinity and long-tail exposure. A separately reported extension adds public author-profile affinity; it is not silently mixed into the headline result.</p>
          <div className="chips"><span>PPO clip</span><span>Reference KL</span><span>Profile extension</span><span>Group advantage</span></div>
        </aside>
      </section>

      <section className="panel agentic" id="agentic">
        <div className="sectionHead">
          <div><span className="kicker">03 / AGENTIC ROUTER V2</span><h2>Turn sparse routing feedback into a learnable action group</h2></div>
          <span className="stamp">3-SEED PILOT PASSED</span>
        </div>
        <p className="sectionIntro">
          V1 exposed a real failure: sparse sampled trajectories collapsed toward a fixed route. V2 adds causal consumed-content feedback,
          enumerates all five tool routes as the GRPO group, and reports only a small directional 10% pilot lift—the paired-user confidence interval still crosses zero.
        </p>
        <div className="policyStrip">
          <div><span>Session return</span><strong>+0.60%</strong><small>10% users · 3 seeds</small></div>
          <div><span>Recall@10</span><strong>+0.73%</strong><small>directional, not significant</small></div>
          <div><span>Policy P95</span><strong>0.85ms</strong><small>cached route policy</small></div>
          <div><span>Invalid exposure</span><strong>0%</strong><small>after 10%/30% failure injection</small></div>
        </div>
        <figure className="architectureFigure">
          <Image src="/livebridge_agentic_serving.png" width={1600} height={940} alt="LiveBridge Agentic RL offline, asynchronous control and real-time serving architecture" />
          <figcaption>Training and policy refresh stay outside the request path; cached routes still pass through live-status filtering and deterministic Bridge fallback.</figcaption>
        </figure>
      </section>

      <section className="panel audit" id="audit">
        <div className="sectionHead">
          <div><span className="kicker">04 / EVIDENCE AUDIT</span><h2>What the result does—and does not—claim</h2></div>
          <span className="stamp">LEAKAGE CHECKED</span>
        </div>
        <div className="auditGrid">
          <article className="pass"><span>PASS</span><h3>Temporal source gate</h3><p>Photo events after each user’s target-train cutoff are excluded before model training.</p></article>
          <article className="pass"><span>PASS</span><h3>Untouched test split</h3><p>GRPO learns from validation logged positives; the test split is opened only for final evaluation.</p></article>
          <article className="warn"><span>BOUNDARY</span><h3>No unbiased OPE</h3><p>The public subset has no action propensity for this policy. Reward is an offline proxy—not online CTR or revenue.</p></article>
          <article className="pass"><span>PASS</span><h3>Parseable artifacts</h3><p>Metrics, per-user rows, seeds, configs, policy weights and audit metadata are written to JSON/CSV.</p></article>
        </div>
      </section>

      <footer>
        <div><strong>LiveBridge—GRPO</strong><span>Built on public KuaiLive-M3</span></div>
        <p>Offline recommendation research project · Results are reproducible, bounded and interview-ready.</p>
      </footer>
    </main>
  );
}
