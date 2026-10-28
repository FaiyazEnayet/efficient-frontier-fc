"""Presentation-only postprocessor. Reads fresh source HTML; never changes research calculations.

Usage: python portfolio_ui/renderer.py --repo NAME --root /path/to/repo --input docs/dashboard.html --output docs/dashboard.html
Requires beautifulsoup4. The Sukuk alternative uses only stdlib plus beautifulsoup4.
"""
import argparse,base64,calendar,copy,datetime,hashlib,html,json,math,re,sys
from pathlib import Path
from bs4 import BeautifulSoup
from configs import CONFIGS
from opening import opening
HERE=Path(__file__).parent
DEC=json.JSONDecoder()
def extract(raw,repo):
 soup=BeautifulSoup(raw,'html.parser');figs=[]
 for sc in soup.find_all('script'):
  t=sc.string or ''
  if len(t)>500000:continue
  for m in re.finditer(r'Plotly\.newPlot\(\s*',t):
   args=[];pos=m.end()
   try:
    for i in range(4):
     while t[pos] in ' \r\n\t,':pos+=1
     v,n=DEC.raw_decode(t[pos:]);args.append(v);pos+=n
   except (ValueError,IndexError):continue
   fm=re.search(r"Plotly\.addFrames\(\s*['\"]"+re.escape(args[0])+r"['\"],\s*",raw)
   figs.append(dict(id=args[0],data=args[1],layout=args[2],config=args[3],frames=DEC.raw_decode(raw[fm.end():])[0] if fm else []))
 d=dict(repo=repo,figures=figs,tables=[],recommendations=[])
 for tb in soup.find_all('table'):
  prev=tb.find_previous(['h2','h3']);headers=[x.get_text(' ',strip=True) for x in tb.select('thead th')]
  rows=[[x.get_text(' ',strip=True) for x in tr.find_all('td')] for tr in tb.select('tbody tr')]
  d['tables'].append(dict(title=prev.get_text(' ',strip=True) if prev else 'Results',headers=headers,rows=rows))
 for card in soup.select('.rec-card'):
  for attr in card.select('[style]'):del attr['style']
  d['recommendations'].append(str(card).replace('—',':'))
 if repo=='formulaforge':
  sc=next(s.string for s in soup.find_all('script') if s.string and 'const DATA = ' in s.string)
  d['formula_data']=DEC.raw_decode(sc.split('const DATA = ',1)[1])[0];d['formula_order']=DEC.raw_decode(sc.split('const ORDER = ',1)[1])[0]
  d['formula_data'].pop('cse214',None);d['formula_order']=[k for k in d['formula_order'] if k!='cse214']
 return d,soup
def jsonfile(root,p):return json.loads((root/p).read_text(encoding='utf8'))
def enrich(d,soup,root):
 repo=d['repo'];c=CONFIGS[repo];d['design']=c;d['metrics']=[]
 if repo=='neural-cppi':
  # The committed HTML and the original generator use different table treatments.
  for f in d['figures'][4:]:
   for t in f['data']:
    if t.get('type')=='table':
     v=t['cells']['values'];d['tables'].append(dict(title=f['layout'].get('title',{}).get('text','Experiment'),headers=t['header']['values'],rows=[list(row) for row in zip(*v)]))
  d['figures']=d['figures'][:4]
 if repo=='neural-cppi':
  t=next((t for t in d['tables'] if 'Single asset' in t['title'] and '63d' in t['title']),None)
  if t:
   r=next(r for r in t['rows'] if r[0]=='learned');d['metrics']=[[r[1],'learned mean wealth','63d · synthetic experiment'],[r[2],'terminal wealth std','same held-out experiment'],[f'{float(r[3])*100:.1f}%','terminal floor breaches','finite sample, not a guarantee']]
 if repo=='FlowForge':
  m=jsonfile(root,'demo/metrics.json');d['metrics_raw']=m
  d['metrics']=[[f"{m['models']['ddpm']['mean_w1']:.3f}",'DDPM mean W1','lower is better'],[f"{m['models']['vae']['mean_w1']:.3f}",'VAE baseline W1'],[f"{m['models']['ddpm']['auc_test']:.3f}",'DDPM discriminator AUC','0.5 = indistinguishable'],[str(m['n_test']),'held-out test windows']]
 if repo=='grind-bandit':
  for n in soup.select('.stats .stat')[:6]:
   b=n.find('b');v=b.get_text(' ',strip=True) if b else '';b.extract() if b else None;d['metrics'].append([v,n.get_text(' ',strip=True)])
 if repo=='studyforge':
  d['metrics']=[[n.select_one('.num').get_text(),n.select_one('.lbl').get_text()] for n in soup.select('.cards .card') if n.select_one('.num')]
 if repo=='deeparcade':
  d['evaluation']=jsonfile(root,'runs/eval.json')
  from game_theme import theme
  game_path=root/'docs/game.html'
  if not game_path.exists():game_path=HERE/'game-snapshot.html'
  d['game_html']=theme(game_path.read_text(encoding='utf8'))
 if repo=='wattwise':
  if len(d['figures'])==1:
   # The original generator emits a multi-panel figure; the committed HTML splits it.
   f=d['figures'][0];groups={}
   for tr in f['data']:groups.setdefault(tr.get('xaxis','domain' if tr.get('type')=='pie' else 'x'),[]).append(copy.deepcopy(tr))
   parts=[]
   for key,traces in groups.items():
    lay=copy.deepcopy(f['layout']);lay.pop('annotations',None);lay.pop('shapes',None)
    for axis in list(lay):
     if re.match(r'^[xy]axis\d*$',axis):del lay[axis]
    if key!='domain':
     xkey=key.replace('x','xaxis');xa=copy.deepcopy(f['layout'].get(xkey,{}));xa.pop('domain',None);xa['anchor']='y';lay['xaxis']=xa
     ys=list(dict.fromkeys(t.get('yaxis','y') for t in traces))
     for i,y in enumerate(ys):
      ya=copy.deepcopy(f['layout'].get(y.replace('y','yaxis'),{}));ya.pop('domain',None);ya['anchor']='x'
      if i:ya['overlaying']='y';ya['side']='right'
      else:ya.pop('overlaying',None)
      lay['yaxis'+(str(i+1) if i else '')]=ya
     for t in traces:t['xaxis']='x';t['yaxis']='y'+(str(ys.index(t.get('yaxis','y'))+1) if ys.index(t.get('yaxis','y')) else '')
    else:
     for t in traces:t.pop('domain',None)
    parts.append(dict(data=traces,layout=lay,frames=[]))
   d['figures']=parts
  m=jsonfile(root,'demo/metrics.json');d['metrics_raw']=m;d['metrics']=[[f"{m['kwh_30d']:.1f}",'kWh · trailing 30 days','synthetic profile'],[f"${m['opt_savings']:.2f}",'modeled shifting savings','6.21% of flexible-load cost'],[f"${m['forecast_bill']:.2f}",'October bill estimate',f"80% band ${m['forecast_bill_lo']:.2f}–${m['forecast_bill_hi']:.2f}"],[f"${m['solar_monthly_saving']:.2f}",'standalone solar savings','6 kW · source scenario']]
  # Exact source tariff curve; weekdays use the source weekday curve, weekends the source off-peak price.
  p=d['figures'][0]['data'][2]['y'];prices=p if isinstance(p,list) else list(__import__('struct').unpack('<'+'d'*(len(base64.b64decode(p['bdata']))//8),base64.b64decode(p['bdata'])))
  d['tariff']=dict(data=[dict(type='heatmap',z=[prices]*5+[[min(prices)]*24]*2,x=list(range(24)),y=['Mon','Tue','Wed','Thu','Fri','Sat','Sun'],hovertemplate='%{y} %{x}:00<br>$%{z:.4f}/kWh<extra></extra>')],layout=dict(xaxis=dict(title=dict(text='Hour of day')),yaxis=dict(title=dict(text='Day')),showlegend=False))
 if repo=='freefood-radar':
  m=jsonfile(root,'models/metrics.json');events=jsonfile(root,'data/seed_events.json');d['metrics']=[[str(len(events)),'seeded event records','demonstration only'],[str(len(d['figures'][1]['data'][0]['cells']['values'][0])),'ranked listings','source top 15'],[f"{m['precision']:.1%}",'classifier precision','held-out labeled text'],[f"{m['recall']:.1%}",'classifier recall','held-out labeled text']]
 if repo=='sukuklab':
  anchor=datetime.date(2026,1,15);dates=[datetime.date(2026+i//2,(7 if i%2==0 else 1),15) for i in range(10)]
  # Explicit semiannual dates, matching Dates.Month increments in Julia.
  dates=[datetime.date(2026+(6+i*6)//12,(6+i*6)%12+1,15) for i in range(10)]
  cfs=[dict(date=x.isoformat(),amount=25.3+(1000 if i==9 else 0),t=(x-anchor).days/365) for i,x in enumerate(dates)]
  raw=jsonfile(root,'data/iilm_2026-01.json');obs=[]
  def collect(v):
   if isinstance(v,dict):
    if 'months' in v and 'profit_rate' in v:obs.append(v)
    for x in v.values():collect(x)
   elif isinstance(v,list):
    for x in v:collect(x)
  collect(raw);ca=datetime.date(2026,1,13);pillars=[];zeros=[]
  for o in obs:
   mm=ca.month-1+o['months'];mat=datetime.date(ca.year+mm//12,mm%12+1,ca.day);t=(mat-ca).days/365;pillars.append(t);zeros.append(math.log1p(o['profit_rate']*t)/t)
  structures=[t for t in d['tables'] if t['title'].startswith(('Ijara','Murabaha','Wakala'))]
  d['sukuk']=dict(cashflows=cfs,pillars=pillars,zeros=zeros,structures=structures)
  d['metrics']=[['13 Jan 2026','curve observation anchor'],['4','IILM maturity pillars'],['5.06%','example profit rate'],['1,000','example face · currency units']]
 d['provenance']=dict(files={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in c['sources'] if (root/p).is_file()},classification=c['classification'])
 return d
def page_parts(d):
 c=d['design'];cols=c['colors'];tokens=';'.join('--'+k+':'+v for k,v in zip(['bg','surface','text','muted','accent','line'],cols));tokens+=';--radius:3px;--display:"'+c['display']+'";--body:"'+c['body']+'";--mono:"'+c['mono']+'"'
 fonts=json.loads((HERE/'fonts.json').read_text(encoding='utf8'));font='\n'.join(fonts.get(f,'') for f in dict.fromkeys([c['display'],c['body'],c['mono']]));css=(HERE/'base.css').read_text(encoding='utf8');js=(HERE/'runtime.js').read_text(encoding='utf8')
 source='https://github.com/FaiyazEnayet/'+d['repo'];limits=''.join('<li>'+html.escape(x)+'</li>' for x in c['limits']);files=''.join('<li><code>'+html.escape(p)+'</code></li>' for p in c['sources'])
 refs={'neural-cppi':[('Wealth & floor','wealth'),('Policy','policy'),('Evidence','evidence')],'FlowForge':[('Window explorer','explorer'),('Discriminator','discriminator'),('Evaluation','evaluation')],'toxicflow':[('Session','session'),('Signals','signals'),('Simulation','simulation')],'efficient-frontier-fc':[('Budget & squad','frontier'),('Players','players'),('Backtest','backtest')],'sukuklab':[('Price & yield','valuation'),('Curve','curve'),('Cash flows','cashflows')],'formulaforge':[('Formula atlas','atlas'),('Leverage','leverage'),('Coverage','coverage')],'grind-bandit':[('Skill board','mastery'),('Recommendation','recommendation'),('Forgetting','forgetting')],'studyforge':[('Calendar','calendar'),('Readiness','readiness'),('Recall','recall'),('Extra hour','allocation')],'deeparcade':[('Evaluation','evaluation'),('Learning','learning'),('Strategy','strategy')],'wattwise':[('Load','load'),('Forecast','forecast'),('Scenarios','scenarios'),('Savings','savings')],'freefood-radar':[('Field map','map'),('Quest listings','events'),('Classifier','model'),('Activity','activity')]}
 nav=''.join(f'<a href="#{a}">{t}</a>' for t,a in refs[d['repo']])+'<a href="#sources">Sources & limits</a>'
 head=f'''<!doctype html><html lang="en" style='{tokens};color-scheme:{'light' if cols[0][1:3] in ['ee','f1','f2','f3','f4'] else 'dark'}'><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="{cols[0]}"><title>{html.escape(c['name'])} | {d['repo']}</title><style>{font}</style><style>{css}</style></head><body data-repo="{d['repo']}"><a class="skip" href="#content">Skip to dashboard</a><div class="shell"><header class="masthead"><div class="identity"><svg class="seal" viewBox="0 0 32 32" aria-hidden="true"><path d="M16 2 29 9v14l-13 7L3 23V9ZM16 8v16M8 12l16 8M24 12 8 20" fill="none" stroke="currentColor" stroke-width="1"/></svg><b>{html.escape(d['repo'])}</b> / RESEARCH PORTFOLIO</div><div class="edition">FAIYAZ ENAYET<br>LOCAL EDITION · 2026</div></header><div class="hero"><div><div class="eyebrow">{html.escape(c['brief'])}</div><h1>{html.escape(c['name'])}</h1><p class="question">{html.escape(c['question'])}</p></div><div class="brief"><b>Evidence, with its boundaries.</b>{html.escape(c['classification'])}<br><span class="badge">SELF-CONTAINED SNAPSHOT</span></div></div><nav class="nav" aria-label="Dashboard sections">{nav}</nav><main id="content"></main><div id="status" class="status" role="status" aria-live="polite">Preparing local charts…</div><section class="source" id="sources"><h2>Evidence & limitations</h2><p>{html.escape(c['classification'])}. This page preserves the repository’s recorded results; it does not fetch live data or rerun research models.</p><ul>{limits}</ul><details><summary>Inspect data lineage and assumptions</summary><p>Source: <a href="{source}">FaiyazEnayet/{d['repo']}</a>. Recorded chart arrays are extracted from the committed dashboard. Additional views use the explicit source calculations documented in DESIGN-AND-VALIDATION.md. Research code and evaluation methods are preserved.</p><ul>{files}</ul><p>Source-file SHA-256 checksums are embedded in this file’s evidence payload. The accompanying validation report distinguishes automated checks from visual review.</p></details></section><footer class="footer"><span>{html.escape(c['name'])} · Faiyaz Enayet</span><span>Designed for inspection. Built from source evidence.</span></footer></div>'''
 head=re.sub(r'<header class="masthead">.*?<main id="content"></main>',opening(d,nav)+'<main id="content"></main>',head,flags=re.S)
 return head,js
def render_five_leagues(root):
 root=Path(root);r=jsonfile(root,'runs/five_leagues/results.json')
 if r.get('feature_version')!=4:raise ValueError('The v2.0.0 dashboard requires FEATURE_VERSION 4 results')
 names=['English Premier League','La Liga','Serie A','Bundesliga','Ligue 1']
 if list(r['leagues'])!=names or r['budgets_eur']!=[300000,500000,700000]:raise ValueError('Unexpected five-league release contract')
 config={k:copy.deepcopy(CONFIGS['efficient-frontier-fc'][k]) for k in ['name','display','body','mono','colors','series']}
 sources=['runs/five_leagues/results.json','runs/five_leagues/report.md']
 d=dict(repo='efficient-frontier-fc',release='v2.0.0',design=config,five_leagues=r,report_source=sources[1],provenance={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sources})
 d['figures']=[dict(data=[dict(type='scatter',name=n,x=r['budgets_eur'],y=[x['exp_vaep_per90'] for x in r['leagues'][n]['frontier']]) for n in names]),dict(data=[dict(type='bar',name=n,x=r['budgets_eur'],y=[x['optimizer']-x['value'] for x in r['leagues'][n]['backtest']]) for n in names]),dict(data=[dict(type='scatter',name=n,x=[v['model']['brier']['mean'] for v in r['leagues'][n]['scoreboard_test'].values()],y=list(r['leagues'][n]['scoreboard_test'])) for n in names])]
 fonts=jsonfile(HERE,'fonts.json');font='\n'.join(fonts.get(f,'') for f in ['Cinzel','IBM Plex Sans'])
 wins=sum(b['optimizer']>b['value'] for v in r['leagues'].values() for b in v['backtest']);total=sum(len(v['backtest']) for v in r['leagues'].values());bd=r['leagues']['Bundesliga'];exception=bd['backtest'][0]
 shell='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#173e30"><title>Efficient Frontier FC | v2.0.0 · Five leagues</title><style>FONT</style><style>CSS</style><style>.xi-line{position:relative;isolation:isolate;display:flex;flex-direction:column;justify-content:space-around;gap:32px;margin:20px 0 30px;padding:44px 22px;min-height:650px;border:8px solid #244e35;outline:1px solid #aab695;background:repeating-linear-gradient(0deg,#245b3b 0,#245b3b 12.5%,#2b6441 12.5%,#2b6441 25%);border-radius:3px}.pitch-markings{position:absolute;inset:18px;width:calc(100% - 36px);height:calc(100% - 36px);z-index:-1;pointer-events:none}.xi-row{display:grid;grid-template-columns:repeat(var(--players),minmax(0,1fr));gap:16px;align-items:start}.xi-name{text-align:center;max-width:210px;width:100%;margin:auto;padding:10px 8px;border:1px solid #c6cba9;background:#fffdf3;box-shadow:0 3px 0 #163c2b;color:#173e30;overflow-wrap:anywhere;font-size:14px;line-height:1.3}.xi-name small{display:block;color:#52634c;font-size:10px;letter-spacing:.1em;margin-bottom:5px}.named-squad td:first-child{min-width:210px;white-space:normal;overflow-wrap:anywhere}.named-squad td:nth-child(2){min-width:140px;white-space:normal}@media(max-width:600px){.xi-line{padding:35px 7px;gap:38px;min-height:660px;border-width:4px}.xi-row{gap:5px}.xi-name{font-size:11px;padding:8px 3px}.xi-name small{font-size:9px}.pitch-markings{inset:8px;width:calc(100% - 16px);height:calc(100% - 16px)}}</style></head><body class="fc-v2"><a class="skip" href="#league-tabs">Skip to league selector</a><header class="mast"><div class="wordmark"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M5 5h30v30H5zM5 20h30M20 5v30" fill="none" stroke="currentColor"/><circle cx="20" cy="20" r="6" fill="none" stroke="currentColor"/></svg><span>Efficient Frontier FC</span></div><div class="edition">v2.0.0</div></header><div class="hero"><div><p class="kicker">FIVE-LEAGUE RESEARCH EDITION</p><h1>Five leagues.<br>One wage constraint.</h1><p class="intro">Where does constrained squad selection outperform a simple value rule? Inspect the evidence, league by league.</p><p class="byline">Faiyaz Enayet</p></div><div class="verdict"><span class="kicker">THE BACKTEST, IN CONTEXT</span><div class="big-result">WINS<span>/TOTAL</span></div><h2>league-budget cells <br>favor the optimizer</h2><p>Compare within each league. Realized totals scale with the evaluation match pool.</p><a href="#view" id="open-backtest">Inspect all budget comparisons <span aria-hidden="true">↘</span></a></div></div><nav id="league-tabs" role="tablist" aria-label="League views"></nav><main id="view" tabindex="-1" role="tabpanel" aria-labelledby="tab-overview"></main><footer><div><b>Efficient Frontier FC · v2.0.0</b><p>Five-league evaluation.</p></div><div><a href="https://github.com/FaiyazEnayet/efficient-frontier-fc/blob/main/runs/five_leagues/report.md">Research report ↗</a><a href="https://raw.githubusercontent.com/FaiyazEnayet/efficient-frontier-fc/main/runs/five_leagues/results.json">Machine-readable results ↗</a><a href="https://github.com/FaiyazEnayet/efficient-frontier-fc">Source repository ↗</a></div></footer><noscript><p class="noscript">Enable JavaScript to explore the recorded league results.</p></noscript><script>LIB</script><script>const EVIDENCE=PAYLOAD;
RUNTIME</script></body></html>'''
 css=r'''
 :root{--paper:#f4f1e6;--surface:#fffdf5;--ink:#173e30;--muted:#566359;--line:#c9cdbd;--gold:#ac8b3c;--rust:#9d4335;--green:#286749;color-scheme:light;font-family:'IBM Plex Sans',sans-serif;color:var(--ink);background:var(--paper)}*{box-sizing:border-box}body{margin:0}button,input,select{font:inherit}button,a,select{-webkit-tap-highlight-color:transparent}a{color:inherit}button{cursor:pointer}button:disabled{cursor:default}a:focus-visible,button:focus-visible,select:focus-visible,summary:focus-visible,[tabindex]:focus-visible{outline:3px solid #b75b28;outline-offset:4px}.skip{position:absolute;top:-80px;left:12px;background:white;padding:12px;z-index:20}.skip:focus{top:12px}.mast,.hero,.reality,#league-tabs,main,footer{max-width:1440px;margin:auto;padding-left:5vw;padding-right:5vw}.mast{display:flex;justify-content:space-between;align-items:center;gap:24px;padding-top:28px;padding-bottom:22px;border-bottom:1px solid var(--line)}.wordmark{display:flex;gap:13px;align-items:center;font-family:'Cinzel',Georgia,serif;font-size:20px;font-weight:700}.wordmark svg{width:34px;height:34px;flex-shrink:0}.edition{font-weight:600;font-size:14px;white-space:nowrap}.edition span{font-size:10px;letter-spacing:.12em;display:block;color:var(--muted);margin-top:3px}.hero{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(270px,.8fr);gap:6vw;padding-top:54px;padding-bottom:46px}.kicker{font-size:11px;letter-spacing:.16em;font-weight:600;margin:0 0 20px}h1{font-family:'Cinzel',Georgia,serif;font-size:clamp(36px,4.8vw,69px);font-weight:500;line-height:1.12;letter-spacing:-.035em;margin:0 0 22px}.intro{font-size:18px;line-height:1.6;max-width:620px;color:#405448}.byline{margin:26px 0 0;font-size:12px}.byline span{display:block;color:var(--muted);margin-top:6px}.verdict{background:var(--ink);color:#f6f1dc;padding:28px 32px;position:relative;border-top:4px solid var(--gold);display:flex;flex-direction:column;justify-content:center}.verdict .kicker{color:#d6c58f}.big-result{font-family:'Cinzel',Georgia,serif;font-size:76px;line-height:1}.big-result span{font-size:33px;color:#c0d0ba}.verdict h2{font-family:inherit;font-weight:500;font-size:20px;line-height:1.4;margin:12px 0}.verdict p{color:#d0dacb;font-size:13px;line-height:1.6}.verdict a{text-decoration:none;border-top:1px solid #56715e;padding-top:18px;margin-top:8px;font-size:12px}.verdict a span{float:right}.reality{display:grid;grid-template-columns:1fr 1.15fr 1fr;gap:24px;padding-top:22px;padding-bottom:22px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);background:#e9eadc}.reality div{font-size:12px;line-height:1.6}.reality b{display:block;font-size:13px;margin-bottom:4px}.reality div+div{border-left:1px solid #c0c6b4;padding-left:24px}#league-tabs{display:flex;gap:8px;flex-wrap:wrap;padding-top:28px;padding-bottom:28px}#league-tabs button{border:1px solid #b6c0ad;background:transparent;color:var(--ink);padding:12px 17px;border-radius:0;font-size:13px;min-height:44px}#league-tabs button[aria-selected=true]{background:var(--ink);border-color:var(--ink);color:white}#league-tabs button small{display:block;font-size:10px;margin-top:3px;opacity:.8}.section-head{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-bottom:24px}.section-head h2{font:500 31px/1.2 'Cinzel',Georgia,serif;margin:0 0 10px}.section-head p{max-width:700px;margin:0;color:var(--muted);font-size:14px;line-height:1.6}.stamp{border:1px solid var(--line);padding:10px 14px;white-space:nowrap;font-size:11px;line-height:1.6}.stamp b{display:block}.sheet{background:var(--surface);border:1px solid var(--line);padding:28px;margin-bottom:24px;min-width:0}.sheet h3{font:500 22px/1.3 'Cinzel',Georgia,serif;margin:0 0 10px}.caption{font-size:12px;line-height:1.6;color:var(--muted);margin:8px 0 18px}.section-label{font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 14px}.comparison{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}.comparison th{text-align:left;font-weight:500;font-size:11px;letter-spacing:.03em;border-bottom:2px solid var(--ink);padding:12px 14px;vertical-align:bottom}.comparison td{padding:16px 14px;border-bottom:1px solid #d8dccd;font-size:13px;vertical-align:top}.comparison tbody tr:last-child td{border-bottom:0}.comparison td small{display:block;color:var(--muted);font-size:11px;margin-top:5px;line-height:1.5}.comparison td strong{font-size:19px;font-weight:500}.comparison button{background:none;border:0;border-bottom:1px solid var(--gold);padding:0 0 3px;color:var(--ink);text-align:left;font-weight:600}.positive{color:#246447}.negative{color:#9d4335}.exception{background:#f7e7dc}.scroll{overflow-x:auto;max-width:100%;outline-offset:2px}.grid-two{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:24px}.grid-two .sheet{margin-bottom:0}.control-row{display:flex;gap:14px;flex-wrap:wrap;align-items:end;margin:20px 0}.control-row label{font-size:11px;display:flex;flex-direction:column;gap:7px;color:var(--muted)}select{min-height:42px;border:1px solid #b8c1b1;background:var(--surface);color:var(--ink);padding:8px 30px 8px 12px;max-width:100%}.chart{width:100%;height:340px;min-width:0}.chart.tall{height:390px}.league-meta{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin:20px 0 24px}.league-meta>div{padding:18px 14px;border-right:1px solid var(--line)}.league-meta>div:first-child{padding-left:0}.league-meta>div:last-child{border:0}.league-meta b{display:block;font-size:26px;font-weight:500;line-height:1.2}.league-meta span{font-size:11px;color:var(--muted);display:block;margin-top:6px}.view-nav{display:flex;gap:4px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-bottom:24px}.view-nav button{background:transparent;border:0;border-bottom:3px solid transparent;padding:13px 20px;color:var(--muted);min-height:44px}.view-nav button[aria-pressed=true]{border-bottom-color:var(--gold);color:var(--ink);font-weight:600}.warning{background:#f6e9d7;border-left:3px solid #a57534;padding:16px 20px;font-size:13px;line-height:1.6;margin:0 0 24px}.warning b{color:var(--rust)}.budget-buttons{display:flex;flex-wrap:wrap;gap:8px}.budget-buttons button{background:transparent;border:1px solid var(--line);color:var(--ink);padding:12px 18px;min-height:44px}.budget-buttons button[aria-pressed=true]{background:var(--ink);color:white;border-color:var(--ink)}.budget-ledger{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}.budget-ledger>div{border-left:2px solid var(--gold);padding-left:16px}.budget-ledger b{font-size:25px;font-weight:500;display:block}.budget-ledger span{font-size:11px;color:var(--muted)}.fine{font-size:12px;line-height:1.7;color:var(--muted)}.provenance{margin-top:26px;padding:22px 0;border-top:1px solid var(--line)}details{padding:17px 0}summary{cursor:pointer;font-weight:600;font-size:13px}details ul{padding-left:20px;font-size:13px;line-height:1.8}code{font-size:11px;overflow-wrap:anywhere}.checksum{font-size:10px;word-break:break-all;line-height:1.7}footer{display:flex;justify-content:space-between;gap:32px;border-top:1px solid var(--line);margin-top:38px;padding-top:28px;padding-bottom:34px;font-size:12px}footer p{color:var(--muted);margin:8px 0}footer a{display:block;margin-bottom:8px}.empty,.noscript{padding:24px;color:var(--rust)}.legend{display:flex;gap:22px;flex-wrap:wrap;font-size:12px;margin-top:12px}.legend span:before{content:'';width:9px;height:9px;display:inline-block;border-radius:50%;margin-right:7px;background:#246447}.legend span+span:before{background:#9d7965}.coverage-track{height:5px;background:#dce1d2;margin:8px 0;max-width:140px}.coverage-track i{display:block;height:100%;background:var(--green)}
 @media(max-width:900px){.hero{grid-template-columns:1.2fr 1fr;gap:30px}.hero h1{font-size:42px}.verdict{padding:22px}.big-result{font-size:64px}.grid-two{grid-template-columns:1fr}.reality{gap:16px}.reality div+div{padding-left:16px}.comparison th,.comparison td{padding:12px 10px}.league-meta{grid-template-columns:repeat(3,minmax(0,1fr))}.league-meta>div{border-bottom:1px solid var(--line)}.sheet{padding:22px}}
 @media(max-width:600px){.mast,.hero,.reality,#league-tabs,main,footer{padding-left:20px;padding-right:20px}.mast{align-items:flex-start;gap:12px}.wordmark{font-size:15px;gap:8px}.wordmark svg{width:27px}.edition{font-size:12px}.edition span{font-size:8px;letter-spacing:.04em}.hero{grid-template-columns:1fr;padding-top:32px;padding-bottom:28px;gap:25px}.hero h1{font-size:39px}.intro{font-size:15px}.byline{margin-top:18px}.verdict{padding:23px}.verdict h2 br{display:none}.verdict .big-result{font-size:66px}.reality{grid-template-columns:1fr;gap:14px;padding-top:18px;padding-bottom:18px}.reality div+div{border-left:0;border-top:1px solid var(--line);padding:13px 0 0}#league-tabs{gap:6px;padding-top:22px;padding-bottom:22px}#league-tabs button{font-size:11px;padding:10px 12px;flex:1 1 29%;white-space:normal}.section-head h2{font-size:24px}.section-head{display:block}.stamp{display:inline-block;margin-top:16px}.sheet{padding:18px;margin-bottom:18px}.sheet h3{font-size:20px}.comparison{min-width:600px}.comparison.small-table{min-width:450px}.comparison.score-table{min-width:780px}.league-meta{grid-template-columns:repeat(2,minmax(0,1fr))}.league-meta>div{padding:14px 10px}.league-meta b{font-size:22px}.league-meta>div:first-child{padding-left:10px}.view-nav button{padding:12px 10px;font-size:12px}.chart{height:330px}.chart.tall{height:380px}.budget-buttons button{flex:1;font-size:12px;padding:10px 8px}.budget-ledger{grid-template-columns:1fr;gap:18px}.budget-ledger b{font-size:23px}.control-row label{flex:1 1 40%;min-width:0}select{font-size:12px;width:100%}footer{display:block}footer>div+div{margin-top:22px}.scroll:after{content:'Scroll table horizontally';display:block;color:var(--muted);font-size:10px;padding-top:12px}}
 @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;animation:none!important;transition:none!important}}
 '''
 runtime=(HERE/'runtime.js').read_text(encoding='utf8')
 release_runtime='function renderFiveLeagueRelease(E){'+runtime.split('function renderFiveLeagueRelease(E){',1)[1]+'\nrenderFiveLeagueRelease(EVIDENCE);'
 replacements={'FONT':font,'CSS':css,'WINS':str(wins),'TOTAL':str(total),'BDM':str(bd['audit']['matches']),'BDT':str(bd['split']['test_n']),'OPT':f"{exception['optimizer']:.2f}",'VAL':f"{exception['value']:.2f}",'LIB':(HERE/'plotly.min.js').read_text(encoding='utf8'),'PAYLOAD':json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('</','<\\/'),'RUNTIME':release_runtime}
 # Replace placeholders in one pass so library/runtime contents are never searched as template tokens.
 shell=re.sub('|'.join(re.escape(x) for x in replacements),lambda m:replacements[m.group()],shell)
 return shell,d


def render_html(root,repo,raw):
 if repo=='efficient-frontier-fc':return render_five_leagues(root)
 d,soup=extract(raw,repo);d=enrich(d,soup,Path(root));head,js=page_parts(d)
 # Full source Plotly distribution: original reduced bundles silently downgrade unsupported traces.
 lib=(HERE/'plotly.min.js').read_text(encoding='utf8')
 katex=''
 if repo=='formulaforge' and (HERE/'katex.min.js').exists():katex='<style>'+(HERE/'katex.css').read_text(encoding='utf8')+'</style><script>'+(HERE/'katex.min.js').read_text(encoding='utf8')+'</script>'
 payload=json.dumps(d,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
 result=head+katex+'<script>'+lib+'</script><script>const EVIDENCE='+payload+';\n'+js+'</script></body></html>'
 return result,d
def render(root,repo,input_path,output_path):
 result,d=render_html(root,repo,Path(input_path).read_text(encoding='utf8'))
 Path(output_path).parent.mkdir(parents=True,exist_ok=True);Path(output_path).write_text(result,encoding='utf8');return d
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--root',required=True);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();render(a.root,a.repo,a.input,a.output)
if __name__=='__main__':main()
