"""Render source-backed README animation. Run scripts/make_portfolio_gif.py."""
from pathlib import Path
import json,base64,re,io,textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle,Circle,Arc,FancyArrowPatch
from matplotlib.ticker import PercentFormatter
from PIL import Image,ImageDraw,ImageFont
from fontTools.ttLib import TTFont

def five_league_showcase(root):
 root=Path(root)
 source=json.loads((root/'runs/five_leagues/results.json').read_text(encoding='utf8'))
 dashboard=(root/'docs/dashboard.html').read_text(encoding='utf8')
 if 'const EVIDENCE=' not in dashboard:raise ValueError('Regenerate the five-league dashboard before the GIF')
 evidence,_=json.JSONDecoder().raw_decode(dashboard.split('const EVIDENCE=',1)[1])
 data=evidence.get('five_leagues')
 if data!=source or evidence.get('release')!='v2.0.0':raise ValueError('Dashboard is stale; run scripts/make_dashboard.py before the GIF')
 if data.get('feature_version')!=4:raise ValueError('README visuals require FEATURE_VERSION 4 results')
 names=list(data['leagues']);colors=['#246447','#9d7965','#af862a','#526e83','#8f5543'];paper='#f4f1e6';ink='#173e30';muted='#566359';line='#c9cdbd';rust='#9d4335'
 fonts=json.loads((root/'portfolio_ui/fonts.json').read_text(encoding='utf8'));cache=root/'.cache/showcase-fonts';cache.mkdir(parents=True,exist_ok=True)
 def family(name):
  path=cache/(name.replace(' ','')+'.ttf')
  if not path.exists():
   raw=base64.b64decode(re.findall(r'base64,([^\)]+)',fonts[name])[-1]);f=TTFont(io.BytesIO(raw));f.flavor=None;f.save(path)
  font_manager.fontManager.addfont(str(path));return font_manager.FontProperties(fname=str(path)).get_name()
 body=family('IBM Plex Sans');display=family('Cinzel');plt.rcParams.update({'font.family':body,'text.color':ink,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':muted,'font.size':11})
 def header(fig,subtitle):
  fig.patch.set_facecolor(paper);fig.text(.055,.948,'EFFICIENT FRONTIER FC',fontfamily=display,fontsize=22,color=ink);fig.text(.945,.954,'v2.0.0',ha='right',fontsize=10,color=muted);fig.text(.055,.906,subtitle,fontsize=12,color=muted)
 def tidy(ax):
  ax.set_facecolor(paper);ax.spines[['top','right']].set_visible(False)
  for s in ax.spines.values():s.set_color(line)
  ax.tick_params(length=0,pad=8);ax.grid(axis='x',color=line,alpha=.6);ax.set_axisbelow(True)
 def matrix(fig,bounds):
  ax=fig.add_axes(bounds);ax.axis('off');ax.set(xlim=(0,4),ylim=(-.2,6))
  for j,budget in enumerate(data['budgets_eur']):ax.text(1.55+j,.0+5.5,f'€{budget:,.0f} / week',ha='center',fontweight='bold',fontsize=11)
  for i,n in enumerate(names):
   d=data['leagues'][n];y=4.65-i;ax.text(.0,y+.12,n.replace('English Premier League','Premier League'),fontsize=12,fontweight='bold');ax.text(.0,y-.2,f"{d['audit']['matches']} matches / {d['split']['test_n']} test",fontsize=9,color=muted)
   for j,b in enumerate(d['backtest']):
    x=1.55+j;win=b['optimizer']>b['value'];ax.add_patch(Rectangle((x-.42,y-.35),.84,.78,facecolor='#e5eddf' if win else '#f3dfd2',edgecolor=line,lw=.6));ax.text(x,y+.12,f"{b['optimizer']-b['value']:+.2f}",ha='center',fontsize=16,color=colors[0] if win else rust);ax.text(x,y-.17,f"{b['optimizer']:.2f} vs {b['value']:.2f}",ha='center',fontsize=9,color=muted)
  return ax
 def overview():
  fig=plt.figure(figsize=(11,7.4));header(fig,'Optimizer minus value / 14 of 15 league-budget cells favor the optimizer');matrix(fig,[.055,.245,.89,.61]);fig.text(.055,.18,'THE EXCEPTION',fontsize=10,fontweight='bold',color=rust);b=data['leagues']['Bundesliga']['backtest'][0];fig.text(.055,.137,f"Bundesliga · €300,000: {b['optimizer']:.2f} optimizer vs {b['value']:.2f} value",fontsize=12);fig.text(.055,.087,'Within-league comparisons only. Raw totals scale with scoring-half match counts.',fontsize=10,color=muted);fig.text(.055,.045,'FEATURE_VERSION 4 results / wage proxies / all 20 test-line confidence intervals overlap',fontsize=9,color=muted);return fig
 frames=[];durations=[]
 def capture(fig,duration):
  buf=io.BytesIO();fig.savefig(buf,format='png',dpi=110,facecolor=paper);plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB').quantize(colors=192));durations.append(duration)
 capture(overview(),1800)
 for i,n in enumerate(names):
  d=data['leagues'][n]
  for j,budget in enumerate(data['budgets_eur']):
   fig=plt.figure(figsize=(11,7.4));header(fig,n+' / '+str(d['audit']['matches'])+' matches / '+str(d['split']['test_n'])+' held-out test')
   ax=fig.add_axes([.10,.35,.375,.45]);tidy(ax);f=d['frontier'];ax.plot([x['budget']/1000 for x in f],[x['exp_vaep_per90'] for x in f],color=colors[0],lw=2,marker='o',ms=5);point=f[j];ax.scatter([budget/1000],[point['exp_vaep_per90']],s=160,color=colors[2],edgecolor=ink,zorder=4);ax.axvline(budget/1000,color=colors[2],lw=1,alpha=.6);ax.set(xlabel='Weekly wage cap · EUR thousands',ylabel='Expected squad VAEP/90',xticks=[300,500,700]);ax.set_title('The feasible frontier',loc='left',fontsize=14,pad=18,fontfamily=display)
   bx=fig.add_axes([.60,.35,.34,.45]);tidy(bx);b=d['backtest'][j];keys=['optimizer','value','random_mean','oracle'];vals=[b[k] for k in keys];bx.barh(range(4),vals,color=[colors[0],colors[1],'#8d9b8b',colors[2]],height=.55);bx.invert_yaxis();bx.set_yticks(range(4),['Optimizer','Value','Random mean','Oracle'],fontsize=10);bx.set_xlim(0,max(vals)*1.23);bx.set_xlabel('Realized second-half total VAEP');bx.set_title('The backtest',loc='left',fontsize=14,pad=18,fontfamily=display)
   for k,v in enumerate(vals):bx.text(v+max(vals)*.025,k,f'{v:.2f}',va='center',fontsize=10)
   fig.text(.075,.218,f"€{budget:,.0f} / week",fontsize=19,fontfamily=display);fig.text(.075,.171,f"Actual wage bill €{point['wage_bill']:,.0f} · {point['n_players']} players · {d['n_rated']} rated",fontsize=11,color=muted)
   delta=b['optimizer']-b['value'];fig.text(.60,.218,f"{delta:+.2f} vs value",fontsize=19,color=colors[0] if delta>0 else rust);fig.text(.60,.171,f"Wage match rate {d['wage_match_rate']:.1%}",fontsize=11,color=muted)
   fig.text(.075,.095,'Bundesliga: 68 matches vs roughly 380 elsewhere. Compare backtests within each league.',fontsize=10,color=muted);fig.text(.075,.05,'Recorded budget points only / model estimates and proxy wages / no player identities invented',fontsize=9,color=muted);capture(fig,800 if j<2 else 1200)
   fig=plt.figure(figsize=(11,7.4));header(fig,n+' / '+('2015/16 + 2023/24' if n=='Bundesliga' else '2015/16')+' / '+f'€{budget:,.0f} per week')
   selected=d['squads'][str(int(budget))];pitch=fig.add_axes([.055,.10,.55,.74]);pitch.set(xlim=(0,100),ylim=(0,100));pitch.axis('off')
   for strip in range(8):pitch.add_patch(Rectangle((0,strip*12.5),100,12.5,facecolor='#245b3b' if strip%2 else '#2b6441',edgecolor='none'))
   for x,y,w,h in [(2,2,96,96),(27,2,46,15),(27,83,46,15)]:pitch.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#c2d4b4',lw=1,alpha=.7))
   pitch.plot([2,98],[50,50],color='#c2d4b4',lw=1,alpha=.7);pitch.add_patch(Circle((50,50),11,fill=False,edgecolor='#c2d4b4',lw=1,alpha=.7))
   for pos,y in [('FWD',82),('MID',61),('DEF',36),('GK',13)]:
    players=[next(r for r in selected['squad'] if r['player']==name) for name in selected['best_xi']];players=[r for r in players if r['pos_group']==pos]
    for k,player in enumerate(players):
     x=(k+.5)*100/len(players);name='\n'.join(textwrap.wrap(player['player'],15));pitch.text(x,y,pos+'\n'+name,ha='center',va='center',fontsize=8.2,color=ink,linespacing=1.15,bbox=dict(facecolor='#fffdf3',edgecolor='#c6cba9',pad=4))
   ax=fig.add_axes([.65,.10,.295,.74]);ax.axis('off');rows=[[r['player'],r['pos_group'],f"€{r['wage_eur']:,.0f}"] for r in selected['squad']]
   tb=ax.table(cellText=rows,colLabels=['Player','Pos','Wage/wk'],colWidths=[.70,.10,.20],cellLoc='left',colLoc='left',bbox=[0,0,1,1]);tb.auto_set_font_size(False);tb.set_fontsize(6.6)
   for (row,col),cell in tb.get_celld().items():
    cell.set_edgecolor(line);cell.set_linewidth(.4);cell.set_facecolor('#e5eddf' if row==0 else paper);cell.set_text_props(color=ink,weight='bold' if row==0 else 'normal')
   fig.text(.055,.055,'Best XI · 4-3-3 / recorded names and positions / EA FC 24 wage proxies',fontsize=10,color=muted);capture(fig,1800)
 capture(overview(),2000);dest=root/'docs/demo.gif';frames[0].save(dest,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True,disposal=2)
 # Results figure uses the same frozen FV4 object, never the older demo tables.
 fig=plt.figure(figsize=(12,10));header(fig,'Five-league evidence / optimizer vs value, coverage and held-out uncertainty');matrix(fig,[.055,.445,.89,.43]);fig.text(.055,.407,'Within-league backtest deltas above. Bundesliga €300,000 trails value: 40.83 vs 41.53.',fontsize=11,color=rust)
 ax=fig.add_axes([.17,.13,.36,.21]);tidy(ax);entries=[data['leagues'][n]['scoreboard_test']['scores_incl'] for n in names]
 for j,kind in enumerate(['model','base_rate']):
  stats=[e[kind]['brier'] for e in entries];means=np.array([v['mean'] for v in stats]);errors=np.array([[v['mean']-v['ci95'][0] for v in stats],[v['ci95'][1]-v['mean'] for v in stats]]);ax.errorbar(means,np.arange(5)+(j-.5)*.18,xerr=errors,fmt='o' if j==0 else 'D',color=colors[j],ms=4,capsize=2,label='Model' if j==0 else 'Base rate',lw=1)
 ax.set_yticks(range(5),['Premier League','La Liga','Serie A','Bundesliga','Ligue 1']);ax.invert_yaxis();ax.set_xlabel('Scores including · Brier / lower is better');ax.set_title('Test means and stored 95% CIs',loc='left',fontsize=12,pad=16);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=2,frameon=False,fontsize=9)
 bx=fig.add_axes([.66,.13,.28,.21]);tidy(bx);rates=[data['leagues'][n]['wage_match_rate']*100 for n in names];bx.barh(range(5),rates,color=colors[0],height=.5);bx.set_yticks(range(5),['Premier League','La Liga','Serie A','Bundesliga','Ligue 1']);bx.invert_yaxis();bx.set(xlim=(0,80),xlabel='Wage-matched players / %');bx.set_title('Coverage limits the modeled pool',loc='left',fontsize=12,pad=16)
 for j,v in enumerate(rates):bx.text(v+1,j,f'{v:.1f}%',va='center',fontsize=9)
 fig.text(.055,.035,'All 20 test-line confidence intervals overlap the base rate. Exact-mean wins are not separated intervals.',fontsize=10,color=muted);fig.savefig(root/'docs/results.png',dpi=160,facecolor=paper);plt.close(fig)
 print('FV4 showcase:',len(frames),'frames;',dest.stat().st_size,'bytes; refreshed docs/results.png')


def main(root=None):
 root=Path(root or Path(__file__).resolve().parents[1]);support=root/'portfolio_ui'
 if (root/'runs/five_leagues/results.json').exists():return five_league_showcase(root)
 if not (support/'showcase-data.json').exists():support=root/'scripts'
 d=json.loads((support/'showcase-data.json').read_text(encoding='utf8'));c=d['design'];repo=d['repo'];bg,surface,ink,muted,accent,line=c['colors'];series=c['series']
 fontpath=root/'portfolio_ui/fonts.json'
 if not fontpath.exists():fontpath=root/'scripts/showcase-fonts.json'
 fonts=json.loads(fontpath.read_text(encoding='utf8'));cache=root/'.cache/showcase-fonts';cache.mkdir(parents=True,exist_ok=True)
 def font(family):
  p=cache/(family.replace(' ','')+'.ttf')
  if not p.exists():
   raw=base64.b64decode(re.findall(r'base64,([^\)]+)',fonts[family])[-1]);f=TTFont(io.BytesIO(raw));f.flavor=None;f.save(p)
  font_manager.fontManager.addfont(str(p));return font_manager.FontProperties(fname=str(p)).get_name(),str(p)
 body,bodypath=font(c['body']);display,displaypath=font(c['display'])
 plt.rcParams.update({'font.family':body,'font.size':10,'text.color':ink,'axes.labelcolor':muted,'xtick.color':muted,'ytick.color':muted,'axes.edgecolor':line,'axes.facecolor':surface,'figure.facecolor':bg,'grid.color':line,'grid.alpha':.45,'legend.facecolor':surface,'legend.edgecolor':line,'legend.labelcolor':ink})
 frames=[];durations=[]
 def capture(fig,duration=180):
  buf=io.BytesIO();fig.savefig(buf,format='png',dpi=100,facecolor=bg);plt.close(fig);buf.seek(0);frames.append(Image.open(buf).convert('RGB').quantize(colors=160));durations.append(duration)
 def title(fig,sub):
  fig.text(.065,.94,c['name'],fontfamily=display,fontsize=26,color=ink);fig.text(.065,.885,sub,fontsize=10,color=muted)
 def tidy(ax):ax.spines[['top','right']].set_visible(False);ax.grid(axis='y');ax.set_axisbelow(True)
 if repo=='deeparcade':
  # Exact recorded rollout of the exported network in the repository SnakeEnv.
  W,H=1040,730;headfont=ImageFont.truetype(displaypath,49);small=ImageFont.truetype(bodypath,16);medium=ImageFont.truetype(bodypath,22);large=ImageFont.truetype(bodypath,42)
  for i,state in enumerate(d['frames']):
   im=Image.new('RGB',(W,H),bg);draw=ImageDraw.Draw(im);draw.text((40,24),'DeepArcade',font=headfont,fill=ink);draw.text((42,84),'PRO / TRAINED DQN',font=small,fill=series[2]);draw.line((42,119,996,119),fill=line,width=1)
   ox,oy,size=42,143,43;draw.rounded_rectangle((ox-7,oy-7,ox+516+7,oy+516+7),radius=5,fill=surface,outline=line,width=2)
   for y in range(12):
    for x in range(12):draw.rectangle((ox+x*size+1,oy+y*size+1,ox+(x+1)*size-1,oy+(y+1)*size-1),fill='#15291f' if (x+y)%2==0 else '#12251b')
   for j,(r,col) in enumerate(state['body']):
    x,y=ox+col*size,oy+r*size;draw.rounded_rectangle((x+3,y+3,x+size-3,y+size-3),radius=6,fill=series[2] if j==len(state['body'])-1 else '#44755b')
   r,col=state['food'];x,y=ox+col*size+size//2,oy+r*size+size//2;draw.polygon([(x,y-11),(x+11,y),(x,y+11),(x-11,y)],fill=accent)
   draw.text((614,150),'SCORE',font=small,fill=muted);draw.text((610,180),str(state['score']).zfill(2),font=large,fill=accent)
   draw.text((614,261),'MOVE',font=small,fill=muted);draw.text((614,291),str(state['step']).zfill(3),font=large,fill=ink)
   draw.text((614,372),'POLICY DECISION',font=small,fill=muted);draw.text((614,404),['STRAIGHT','TURN LEFT','TURN RIGHT'][state['action']],font=medium,fill=series[2])
   draw.line((614,461,982,461),fill=line);draw.text((614,483),'12 x 12 board',font=small,fill=ink);draw.text((614,515),'Seed 2036 / greedy policy',font=small,fill=muted)
   for j,t in enumerate(['One recorded run.','Compare distributions','in the interactive demo.']):draw.text((614,567+j*25),t,font=small,fill=muted)
   draw.text((42,691),'EXISTING TRAINED WEIGHTS / ACTUAL ENVIRONMENT ROLLOUT',font=small,fill=muted)
   frames.append(im.quantize(colors=128));durations.append(90)
  durations[-1]=1700
 elif repo=='efficient-frontier-fc':
  rows=d['frontier'];budgets=np.array([r['budget']/1e6 for r in rows]);vals=np.array([r['exp_vaep_per90'] for r in rows])
  for i,row in enumerate(rows):
   fig=plt.figure(figsize=(11,7.4));title(fig,'A wage budget becomes a feasible 23-player squad and a best XI')
   ax=fig.add_axes([.08,.20,.37,.57]);tidy(ax);ax.plot(budgets,vals,color=series[0],lw=2);ax.scatter([budgets[i]],[vals[i]],s=90,color=accent,zorder=4);ax.axvline(budgets[i],color=line,ls='--');ax.set(xlabel='Weekly wage budget / EUR millions',ylabel='Expected squad VAEP / 90');ax.set_title('Efficient frontier',loc='left',fontsize=13,pad=13)
   pitch=fig.add_axes([.53,.19,.41,.60]);pitch.set(xlim=(0,100),ylim=(0,110));pitch.axis('off');pitch.add_patch(Rectangle((0,0),100,110,facecolor='#183f30',edgecolor=series[0],lw=1.5));pitch.add_patch(Rectangle((5,5),90,100,fill=False,edgecolor='#779c7c'));pitch.plot([5,95],[55,55],color='#779c7c');pitch.add_patch(Circle((50,55),12,fill=False,edgecolor='#779c7c'));pitch.add_patch(Rectangle((28,5),44,17,fill=False,edgecolor='#779c7c'));pitch.add_patch(Rectangle((28,88),44,17,fill=False,edgecolor='#779c7c'))
   xi=d['squads'][i]
   for pos,y in [('GK',14),('DEF',38),('MID',66),('FWD',93)]:
    players=sorted([p for p in xi if p['pos_group']==pos],key=lambda p:p['vaep_per90'],reverse=True)
    for x,p in zip(np.linspace(13,87,len(players)) if len(players)>1 else [50],players):
     pitch.scatter(x,y,s=230,color='#d2bb70',edgecolor='#eee7cb',zorder=3);name=p['player'].split()[-1];pitch.text(x,y-8,'\n'.join(textwrap.wrap(name,13)),ha='center',va='top',fontsize=9,color='#fff8df')
   pitch.set_title('Best XI / 4-3-3',loc='left',fontsize=13,pad=13)
   fig.text(.08,.09,f"Budget EUR {row['budget']/1e6:.2f}M/wk    Actual wage bill EUR {row['wage_bill']/1e6:.2f}M/wk",fontsize=12,color=ink)
   fig.text(.08,.045,'Source demo selections / wage proxies / historical rating estimates',fontsize=9,color=muted);capture(fig,350)
 elif repo=='formulaforge':
  nodes=d['sheet']['nodes'];targets=['work-def','ke','work-energy','energy-cons','momentum','torque','ang-mom'];keep=set()
  def ancestors(k):
   if k in keep:return
   keep.add(k)
   for p in nodes[k]['prereqs']:ancestors(p)
  for k in targets:ancestors(k)
  levels={}
  for k in sorted(keep):levels.setdefault(nodes[k]['level'],[]).append(k)
  coords={k:(j,max(levels)-lev) for lev,ks in levels.items() for j,k in enumerate(ks)}
  for target in targets:
   active=set()
   def visit(k):
    active.add(k)
    for p in nodes[k]['prereqs']:
     if p not in active:visit(p)
   visit(target)
   for phase in range(4):
    fig=plt.figure(figsize=(11,7.4));title(fig,'Follow the prerequisite structure / Classical Physics I');ax=fig.add_axes([.06,.21,.61,.59]);ax.axis('off');ax.set(xlim=(-.8,max(x for x,y in coords.values())+.8),ylim=(-.65,max(y for x,y in coords.values())+.65))
    for k in keep:
     x,y=coords[k]
     for p in nodes[k]['prereqs']:
      if p not in keep:continue
      ax.add_patch(FancyArrowPatch(coords[p],(x,y),arrowstyle='-|>',mutation_scale=10,shrinkA=16,shrinkB=16,color=accent if k in active and p in active else line,lw=1.8 if k in active else .8))
    for k in keep:
     x,y=coords[k];selected=k==target;ax.scatter(x,y,s=500 if selected else 230,color=accent if selected else series[0] if k in active else line,zorder=3);ax.text(x,y-.22,'\n'.join(textwrap.wrap(nodes[k]['name'],16)),ha='center',va='top',fontsize=9,color=ink if k in active else muted)
    fig.text(.73,.76,'SELECTED FORMULA',fontsize=10,color=muted);fig.text(.73,.65,'\n'.join(textwrap.wrap(nodes[target]['name'],19)),fontfamily=display,fontsize=22,color=ink);fig.text(.73,.44,'\n'.join(textwrap.wrap(nodes[target]['desc'],30)),fontsize=11,color=muted,linespacing=1.5)
    fig.text(.07,.10,'Highlight: selected formula and its structural ancestors',fontsize=11,color=ink);fig.text(.07,.052,'Relationships from the authored sheet / structural reach does not establish derivability',fontsize=9,color=muted);capture(fig,220 if phase<3 else 700)
 elif repo=='studyforge':
  F=d['figures'];curve=F[3]['data'][0];reviews=F[3]['data'][1];table=F[2]['data'][0]['cells']['values'];days=sorted(set(int(v.split()[-1]) for v in table[0]));exams=list(dict.fromkeys(v for v in table[1] if v!='rest'));matrix=np.zeros((len(exams),max(days)+1))
  for day,exam,topic,mins in zip(*table):
   if exam in exams:matrix[exams.index(exam),int(day.split()[-1])]+=float(str(mins).split()[0])
  for frame in range(35):
   t=max(curve['x'])*(frame+1)/35;fig=plt.figure(figsize=(11,7.4));title(fig,'Limited study time / scheduled sessions and estimated recall')
   ax=fig.add_axes([.10,.50,.83,.28]);tidy(ax);x=np.array(curve['x']);y=np.array(curve['y']);mask=x<=t;ax.plot(x[mask],y[mask],color=series[0],lw=2);rx=np.array(reviews['x']);ry=np.array(reviews['y']);m=rx<=t;ax.scatter(rx[m],ry[m],color=series[1],s=35,zorder=3);ax.set(xlim=(0,max(x)),ylim=(0,105),ylabel='Estimated recall / %');ax.set_title(curve['name']+' / model trajectory',loc='left',fontsize=12)
   cal=fig.add_axes([.10,.16,.83,.22]);bottom=np.zeros(matrix.shape[1]);xs=np.arange(matrix.shape[1]);
   for j,exam in enumerate(exams):cal.bar(xs,matrix[j],bottom=bottom,color=series[j],label=exam,width=.73);bottom+=matrix[j]
   cal.axvline(min(t,max(days)),color=accent,lw=2);cal.set(xlim=(-.7,max(days)+.7),xlabel='Plan day',ylabel='Study minutes');cal.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=3,fontsize=9,frameon=False);tidy(cal);fig.text(.10,.025,'Existing 72-session plan / future recall is estimated, not a measured exam outcome',fontsize=9,color=muted);capture(fig,200)
 elif repo=='grind-bandit':
  F=d['figures'];traces=F[1]['data'];chosen=sorted(traces,key=lambda t:t['y'][0],reverse=True)[:5]
  for frame in range(35):
   k=max(1,round(len(chosen[0]['x'])*(frame+1)/35));fig=plt.figure(figsize=(11,7.4));title(fig,'Mastery estimates fade / a practice outcome updates the model')
   ax=fig.add_axes([.10,.25,.49,.53]);tidy(ax)
   for j,tr in enumerate(chosen):ax.plot(tr['x'][:k],tr['y'][:k],lw=2,color=series[j%len(series)],label=tr['name'])
   ax.set(xlim=(0,180),ylim=(0,1),xlabel='Days without further practice',ylabel='Model-estimated mastery');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend(loc='upper right',fontsize=9)
   bx=fig.add_axes([.72,.25,.23,.53]);tidy(bx);outcomes=F[2]['data'][:3];ys=[float(t['y'][0]) for t in outcomes];bx.bar(range(3),ys,color=[line,series[2],series[1]],width=.58);bx.set(ylim=(0,1));bx.set_xticks(range(3),['Before','Solve','Fail'],fontsize=9);bx.yaxis.set_major_formatter(PercentFormatter(1));bx.set_title('Graph / BFS update',fontsize=12,pad=15)
   for j,y in enumerate(ys):bx.text(j,y+.035,f'{y:.0%}',ha='center',fontsize=12,color=ink)
   fig.text(.10,.12,'Compare topic trajectories before choosing the next practice problem.',fontsize=11,color=ink);fig.text(.10,.06,'Source demonstration history and learning model / estimates are not observed proficiency',fontsize=9,color=muted);capture(fig,180)
 durations[-1]=1600;dest=root/'docs/demo.gif';dest.parent.mkdir(exist_ok=True)
 frames[0].save(dest,save_all=True,append_images=frames[1:],duration=durations,loop=0,optimize=True,disposal=2)
 print(repo,len(frames),'frames',dest.stat().st_size,'bytes')

if __name__=='__main__':main()
