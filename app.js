/* =====================================================================
   회귀 분석 모델 · 이론 시뮬레이션 (교과서 110–113쪽)
   ===================================================================== */
const APP={key:'regression-theory-v1',
  init:()=>({roles:{},cause:{},hidden:false,
    a:1,b:2,showMy:false,showRes:true,showFit:false,
    pts:[],
    r:.8,outlier:false,ox:2.6,oy:-2.4,showLine:true,
    ea:1,eb:0,ys:[3,5,3,7],sq:false,
    noise:1.4,r2view:'sse',seed:1,hist:[]}),
  resetStep(i){const d=APP.init(),a=S.app;
    ({1:['roles','cause'],2:['a','b','showMy','showRes','showFit','pts'],3:['r','outlier','ox','oy','showLine','hidden'],4:['ea','eb','ys','sq'],5:['noise','r2view','seed','hist']}[i]||[]).forEach(k=>a[k]=d[k])}
};

/* ---------- 계산 도우미 ---------- */
function linfit(xs,ys){const mx=mean(xs),my=mean(ys);let sxy=0,sxx=0;xs.forEach((x,k)=>{sxy+=(x-mx)*(ys[k]-my);sxx+=(x-mx)**2});const a=sxx?sxy/sxx:0;return{a,b:my-a*mx,mx,my,sxy,sxx}}
function pearson(xs,ys){const mx=mean(xs),my=mean(ys);let sxy=0,sxx=0,syy=0;xs.forEach((x,k)=>{const dx=x-mx,dy=ys[k]-my;sxy+=dx*dy;sxx+=dx*dx;syy+=dy*dy});return{r:sxx&&syy?sxy/Math.sqrt(sxx*syy):0,sxy,sxx,syy,mx,my}}
const mseOf=(xs,ys,a,b)=>mean(xs.map((x,k)=>(ys[k]-(a*x+b))**2));
const eqStr=(a,b,d=2)=>`y = ${fx(a,d)}x ${b<0?'−':'+'} ${fx(Math.abs(b),d)}`;
const r2s=v=>Math.round(v*100)/100;

/* ---------- 데이터 ---------- */
// 111쪽 스스로 실습: 광고비(x, 백만 원)와 매출액(y, 천만 원)
const AD_X=[1,2,3,4,5,6,7,8,9],AD_Y=[5,3,2,3,4,6,10,12,18];
const AD_FIT=linfit(AD_X,AD_Y),AD_BEST=mseOf(AD_X,AD_Y,AD_FIT.a,AD_FIT.b);
// 4단계 오차 예시: x = 2, 4, 6, 8
const E_X=[2,4,6,8];
// 상관관계 시뮬레이션용 표준정규 난수 50쌍 (시드 고정)
const CZ=(()=>{const r=rng(7),n=50,z1=[],z2=[];for(let i=0;i<n;i++){z1.push(gauss(r));z2.push(gauss(r))}
  // 두 열을 평균 0, 표준편차 1, 서로 상관 0으로 맞춥니다 → 슬라이더 값과 실제 r이 같아집니다
  const st=a=>{const m=mean(a),s=Math.sqrt(mean(a.map(v=>(v-m)**2)));return a.map(v=>(v-m)/s)};const a=st(z1);let b=st(z2);const c=mean(a.map((v,k)=>v*b[k]));b=st(b.map((v,k)=>v-c*a[k]));return{a,b}})();
function corrPts(){const r=A().r,s=Math.sqrt(Math.max(0,1-r*r));const xs=CZ.a.slice(),ys=CZ.a.map((v,k)=>r*v+s*CZ.b[k]);if(A().outlier){xs.push(A().ox);ys.push(A().oy)}return{xs,ys}}
// 숨은 원인: 기온 → 아이스크림 판매량, 기온 → 물놀이 사고
const HID=(()=>{const r=rng(21),d=[];for(let i=0;i<60;i++){const t=10+r()*25;d.push({t,ice:Math.round(40+9*t+gauss(r)*28),acc:Math.max(0,Math.round(.32*t+gauss(r)*1.4))})}return d})();
const band=t=>t<18?1:t<27?2:3;
// 결정 계수: x 40개, 오차 방향은 고정하고 크기만 슬라이더로 바꿉니다
const R2D=(()=>{const r=rng(11),xs=[],e=[];for(let i=0;i<40;i++){xs.push(Math.round(r()*1000)/100);e.push(gauss(r))}return{xs,e}})();
const r2Y=()=>{const sd=A().noise**2;return R2D.xs.map((x,k)=>2*x+1+sd*R2D.e[k])};
function r2Stat(){const ys=r2Y(),f=linfit(R2D.xs,ys),my=mean(ys);const sst=sum(ys.map(y=>(y-my)**2)),sse=sum(R2D.xs.map((x,k)=>(ys[k]-(f.a*x+f.b))**2));return{ys,f,my,sst,sse,r2:sst?1-sse/sst:1}}
// 기계학습: 데이터 30개를 학습 70%(21개) · 평가 30%(9개)로 나눕니다
const ML=(()=>{const r=rng(5),xs=[],ys=[];for(let i=0;i<30;i++){const x=Math.round(r()*1000)/100;xs.push(x);ys.push(1.5*x+2+gauss(r)*2.2)}return{xs,ys}})();
function mlSplit(seed){const r=rng(seed*97+3),idx=[...ML.xs.keys()];for(let i=idx.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]]}
  const te=new Set(idx.slice(0,9)),tr=idx.filter(i=>!te.has(i));const f=linfit(tr.map(i=>ML.xs[i]),tr.map(i=>ML.ys[i]));
  const m=ids=>mean(ids.map(i=>(ML.ys[i]-(f.a*ML.xs[i]+f.b))**2));return{te,tr,f,trMSE:m(tr),teMSE:m([...te])}}

/* ---------- 1단계 자료 ---------- */
const PAIRS=[['광고비','매출액',0],['시험 점수','공부 시간',1],['기온','아이스크림 판매량',0],['수확량','비료 사용량',1]];
const CAUSE=[{t:'키가 큰 사람은 몸무게도 많이 나가는 편이다.',a:'cor',e:'키가 크다고 해서 반드시 몸무게가 많이 나가는 것은 아니에요. (교과서 110쪽)'},
  {t:'아이스크림이 많이 팔리는 날에는 물놀이 사고도 많다.',a:'cor',e:'둘 다 기온이 높을 때 늘어날 뿐, 아이스크림이 사고의 원인은 아니에요.'},
  {t:'운동 시간을 늘리면 소모 칼로리가 늘어난다.',a:'cause',e:'운동이 칼로리를 소모하게 만드는 원인이에요.'},
  {t:'스위치를 누르면 전등이 켜진다.',a:'cause',e:'스위치를 누르는 것이 전등이 켜지는 원인이에요.'}];

/* ---------- 도전 과제 (이 앱은 사용하지 않음) ---------- */
const MISSIONS={};

/* ---------- 연습문제 ---------- */
const AD10=AD_FIT.a*10+AD_FIT.b;
const QUIZ={
  var:{title:'독립 변수 · 종속 변수와 인과 관계',intro:'1단계 시뮬레이션을 떠올리며 알맞은 답을 고르세요.',items:[
    {q:'광고비에 따라 매출액이 달라질 때, 종속 변수는 무엇인가요?',o:['광고비','매출액','광고비와 매출액 모두','둘 다 아니다'],a:2,e:'매출액은 광고비의 영향을 받아 달라지는 값이므로 종속 변수입니다.\n광고비처럼 다른 변수에 영향을 주는 변수가 독립 변수입니다.'},
    {q:'아이스크림 판매량과 물놀이 사고 수의 관계로 알맞은 것은 무엇인가요?',o:['아이스크림이 물놀이 사고의 원인이다','물놀이 사고가 아이스크림 판매의 원인이다','상관관계는 있지만 인과 관계라고 할 수 없다','두 변수는 아무 관계가 없다'],a:3,e:'두 변수는 함께 늘어나는 상관관계가 있지만, 실제 원인은 둘 다에 영향을 주는 기온입니다.\n같은 기온끼리 묶어 보면 둘 사이의 관계가 거의 사라집니다.'},
    {q:'상관관계와 인과 관계에 대한 설명으로 옳은 것은 무엇인가요?',o:['상관관계가 있으면 항상 인과 관계가 있다','인과 관계는 원인과 결과의 관계까지 밝힌다','인과 관계가 있으면 상관관계는 없다','상관관계는 원인과 결과를 뜻한다'],a:2,e:'인과 관계는 상관관계에서 원인과 결과의 관계까지 밝힙니다.\n상관관계만으로는 인과 관계를 정확히 밝히지 못합니다.'}]},
  lin:{title:'선형 회귀 분석 이해하기',intro:'2단계 시뮬레이션의 결과를 보고 알맞은 답을 고르세요.',items:[
    {q:'광고비, 상품 구매 후기 수, 가격 세 가지로 매출액을 설명하는 회귀 분석은 무엇인가요?',o:['단순 선형 회귀 분석','다중 선형 회귀 분석','군집 분석','연관 분석'],a:2,e:'독립 변수가 한 개이면 단순 선형 회귀 분석, 두 개 이상이면 다중 선형 회귀 분석입니다.\n여기서는 독립 변수가 세 개입니다.'},
    {q:`광고비·매출액 데이터의 회귀 직선 ${eqStr(AD_FIT.a,AD_FIT.b)}로 광고비가 10(1,000만 원)일 때 매출액(천만 원)을 예측하면 약 얼마인가요?`,o:[`약 ${fx(AD_FIT.a)}`,`약 ${fx(AD_FIT.my+1.6)}`,`약 ${fx(AD10,1)}`,'약 18.0'],a:3,e:`x에 10을 넣으면 ${fx(AD_FIT.a)} × 10 ${AD_FIT.b<0?'−':'+'} ${fx(Math.abs(AD_FIT.b))} = ${fx(AD10)}입니다.\n회귀 분석 모델로 아직 관측하지 않은 값을 예측할 수 있습니다.`},
    {q:'최소제곱법으로 구한 회귀 직선(빨간 선)은 어떤 직선인가요?',o:['모든 점을 반드시 지나는 직선','첫 번째 점과 마지막 점을 잇는 직선','기울기가 항상 1인 직선','오차를 제곱한 값의 평균이 가장 작아지는 직선'],a:4,e:'회귀 직선은 데이터를 가장 잘 대표하는 하나의 선으로, 오차 제곱의 평균(평균제곱오차)이 가장 작아지도록 정합니다.\n그래서 내 직선을 아무리 움직여도 빨간 선의 평균제곱오차보다 작아지지 않습니다.'}]},
  cor:{title:'상관관계 분석 이해하기',intro:'3단계 시뮬레이션의 결과를 보고 알맞은 답을 고르세요.',items:[
    {q:'변수 x가 증가함에 따라 변수 y가 감소하는 관계는 무엇인가요?',o:['양의 상관관계','음의 상관관계','상관관계 없음','인과 관계'],a:2,e:'x가 증가할 때 y가 감소하면 음의 상관관계, y도 증가하면 양의 상관관계입니다.\n음의 상관관계에서는 상관관계 계수가 음수입니다.'},
    {q:'상관관계 계수가 0.15일 때 알맞은 해석은 무엇인가요?',o:['강한 양의 상관관계','약한 상관관계','강한 음의 상관관계','완벽한 상관관계'],a:2,e:'상관관계 계수가 ±0.3 미만이면 약한 상관관계를 뜻합니다.\n시뮬레이션에서 r을 0 근처로 두면 점들이 둥글게 흩어집니다.'},
    {q:'피어슨 상관관계 계수가 가질 수 있는 값의 범위는 어떻게 되나요?',o:['0부터 1까지','0부터 100까지','−1부터 1까지','−100부터 100까지'],a:3,e:'상관관계 계수는 −1부터 1 사이의 값입니다.\n1이나 −1에 가까울수록 점들이 한 직선 위에 가깝게 모여 있습니다.'},
    {q:'숨은 원인 찾기에서 기온이 비슷한 날끼리만 보면 아이스크림 판매량과 물놀이 사고의 상관관계 계수가 약해졌습니다. 그 이유로 알맞은 것은 무엇인가요?',o:['아이스크림이 물놀이 사고를 줄이기 때문이다','두 변수가 모두 기온의 영향을 받아 함께 변했기 때문이다','데이터의 개수가 늘어났기 때문이다','상관관계 계수는 날마다 무작위로 바뀌기 때문이다'],a:2,e:'두 변수는 서로 원인이 아니라, 둘 다 기온이라는 숨은 원인 때문에 함께 늘어났습니다.\n그래서 기온을 같게 맞추면 관계가 약해집니다.\n상관관계 계수가 커도 인과 관계라고 단정할 수 없습니다.'}]},
  err:{title:'오차와 평가 지표 이해하기',intro:'4단계 시뮬레이션의 결과를 보고 알맞은 답을 고르세요.',items:[
    {q:'실젯값이 5이고 예측값이 4일 때, 오차(실젯값 − 예측값)는 얼마인가요?',o:['1','−1','9','20'],a:1,e:'오차는 실젯값에서 예측값을 뺀 값이므로 5 − 4 = 1입니다.\n4단계 표도 실젯값 − 예측값으로 오차를 나타냅니다.'},
    {q:'오차가 1, 1, −3, −1일 때 평균절대오차(MAE)는 얼마인가요?',o:['0.5','3','2.75','1.5'],a:4,e:'절댓값을 취하면 1, 1, 3, 1이고, 합 6을 개수 4로 나누면 1.5입니다.\n같은 오차로 평균제곱오차를 구하면 (1 + 1 + 9 + 1) ÷ 4 = 3입니다.'},
    {q:'평균제곱오차가 평균절대오차보다 큰 오차에 더 민감한 까닭은 무엇인가요?',o:['오차를 제곱하기 때문','오차에 절댓값을 취하기 때문','개수로 나누지 않기 때문','음수 오차를 빼기 때문'],a:1,e:'오차를 제곱하면 1은 1, 3은 9가 되어 큰 오차가 훨씬 크게 반영됩니다.\n‘제곱을 넓이로 보기’를 켜면 큰 오차의 정사각형이 눈에 띄게 커집니다.'}]},
  r2:{title:'결정 계수와 두 가지 회귀 분석',intro:'5단계 시뮬레이션의 결과를 보고 알맞은 답을 고르세요.',items:[
    {q:'결정 계수(R²)에 대한 설명으로 옳은 것은 무엇인가요?',o:['−1부터 1 사이이며 0이 가장 좋다','0부터 1 사이이며 1에 가까울수록 설명력이 높다','1보다 커야 좋은 모델이다','작을수록 좋은 모델이다'],a:2,e:'결정 계수는 0에서 1 사이의 값을 가지며, 1에 가까울수록 회귀 분석 모델이 데이터를 잘 설명합니다.\n점들이 회귀 직선에 모두 놓이면 1이 됩니다.'},
    {q:'기계학습 기반 회귀 분석에서 가장 많이 사용하는 평가 지표는 무엇인가요?',o:['결정 계수','상관관계 계수','평균제곱오차','향상도'],a:3,e:'기계학습 기반 회귀 분석은 오차가 작을수록 좋은 예측 모델이므로 평균제곱오차를 가장 많이 사용합니다.\n평가 데이터로 계산한 평균제곱오차가 작을수록 좋은 모델입니다.'},
    {q:'기계학습 기반 회귀 분석에서 평가 데이터를 따로 두는 까닭은 무엇인가요?',o:['계산을 빠르게 하려고','결정 계수를 1로 만들려고','학습 데이터를 늘리려고','학습에 쓰지 않은 데이터로 예측 성능을 확인하려고'],a:4,e:'학습 데이터로 모델을 학습시키고, 학습에 쓰지 않은 평가 데이터로 예측 성능을 평가해야 처음 보는 데이터도 잘 예측하는지 알 수 있습니다.\n나누는 방법에 따라 평가 오차가 조금씩 달라지는 것도 확인해 보세요.'}]}
};

/* ---------- 화면 조각 ---------- */
const VIEWS={};
// 1단계
VIEWS.pairs=()=>`<div class="pairs">${PAIRS.map((p,k)=>{const c=A().roles[k];return`<div class="pair-row"><button type="button" class="opt ${c===0?(p[2]===0?'right':'wrong'):''}" data-act="role" data-k="${k}" data-v="0">${p[0]}</button><span class="arr">독립 변수는?</span><button type="button" class="opt ${c===1?(p[2]===1?'right':'wrong'):''}" data-act="role" data-k="${k}" data-v="1">${p[1]}</button></div>`+
  (c===p[2]?`<div class="vdiag" style="margin:-2px 0 4px"><span class="vbox x">독립 변수 · ${p[p[2]]}</span><span>→</span><span class="vbox y">종속 변수 · ${p[1-p[2]]}</span></div>`:'')}).join('')}</div>`;
VIEWS.cause=()=>`<div class="pairs">${CAUSE.map((c,k)=>{const v=A().cause[k];return`<div class="quiz-item" style="padding-top:8px"><p class="q">${c.t}</p><div class="opts">${[['cor','상관관계만 있다'],['cause','인과 관계도 있다']].map(([o,t])=>`<button type="button" class="opt ${v===o?(c.a===o?'right':'wrong'):''}" data-act="cause" data-k="${k}" data-v="${o}">${t}</button>`).join('')}</div>${v?`<p class="fb ${v===c.a?'good':'bad'}">${v===c.a?c.e:'다시 생각해 보세요. 한쪽이 다른 쪽의 원인인가요?'}</p>`:''}</div>`}).join('')}</div>`;
VIEWS.hidden=()=>{
  const P=plot({id:'p-hid',w:520,h:330,x:[100,420],y:[-1,15],xl:'아이스크림 판매량(개)',yl:'물놀이 사고(건)',label:'아이스크림 판매량과 물놀이 사고 산점도'});
  let s=P.head;const on=A().hidden,C=['','var(--s1)','var(--s4)','var(--s2)'];
  const hf=linfit(HID.map(d=>d.ice),HID.map(d=>d.acc));
  s+=`<clipPath id="cp-hid"><rect x="${P.m.l}" y="${P.m.t}" width="${P.iw}" height="${P.ih}"/></clipPath><line class="line-faint" clip-path="url(#cp-hid)" x1="${P.X(100)}" y1="${P.Y(hf.a*100+hf.b)}" x2="${P.X(420)}" y2="${P.Y(hf.a*420+hf.b)}" pointer-events="none"/>`;
  HID.forEach(d=>{s+=`<circle cx="${P.X(d.ice)}" cy="${P.Y(d.acc)}" r="5" class="ring2" style="fill:${on?C[band(d.t)]:'var(--unk)'}" data-tip="${esc(`기온 ${fx(d.t,1)}°C\n아이스크림 ${d.ice}개 · 사고 ${d.acc}건`)}"/>`});
  if(on)[['18°C 미만',1],['18~27°C',2],['27°C 이상',3]].forEach(([t,b],i)=>{s+=`<circle cx="${P.m.l+14}" cy="${P.m.t+12+i*16}" r="5" style="fill:${C[b]}"/><text class="leg" x="${P.m.l+24}" y="${P.m.t+16+i*16}">${t}</text>`});
  s+=P.end;
  const all=pearson(HID.map(d=>d.ice),HID.map(d=>d.acc)).r;
  const per=[1,2,3].map(b=>{const g=HID.filter(d=>band(d.t)===b);return pearson(g.map(d=>d.ice),g.map(d=>d.acc)).r});
  return `<div class="sim2"><div>${s}</div><div><div class="stat">${fact('전체 상관관계 계수',`r = ${fx(all)}`,'아이스크림이 많이 팔리면 사고도 많다?','hl')}</div><p class="hint">옅은 빨간 선은 60일 전체 데이터로 구한 회귀 직선입니다.</p>${on?`<div class="stat">${['18°C 미만','18~27°C','27°C 이상'].map((t,i)=>fact(t,`r = ${fx(per[i])}`,'같은 기온끼리')).join('')}</div>`+insight('숨은 원인은 기온','기온이 비슷한 날끼리만 비교하면 상관관계가 전체보다 약해져요. 아이스크림과 사고는 둘 다 기온의 영향을 받았을 뿐이에요.'):`<p class="note">점 위에 마우스를 올려 보세요. 두 변수는 강한 양의 상관관계를 보입니다. 그렇다면 아이스크림을 덜 팔면 사고가 줄어들까요?</p>`}</div></div>`;
};
// 2단계: 오차의 합 대신 오차제곱합으로 비교하는 까닭 (최소제곱법)
const LSQ=`<p>직선이 점들에서 얼마나 벗어났는지 <b>오차를 그냥 더해서</b> 비교하면 안 됩니다. 오차에는 <b>+</b>와 <b>−</b>가 섞여 있어서 더하면 서로 지워지기 때문에, 점들에서 많이 벗어난 직선도 오차의 합이 0에 가까워질 수 있습니다.</p>
  <p>그래서 오차를 <b>제곱해서 모두 더한 값(오차제곱합)</b>으로 판단합니다. 제곱하면 모두 0 이상이 되어 서로 지워지지 않습니다.</p>
  <div class="insight"><span>최소제곱법 :</span><p>오차제곱합이 <b>가장 작아지는</b> 직선을 찾는 방법</p></div>`;
// 2단계: 광고비·매출액 데이터와 내 직선
VIEWS.ad=()=>{
  const {a,b,showMy,showRes,showFit}=A(),P=plot({id:'p-ad',w:520,h:360,x:[0,10],y:[-2,20],xt:[0,1,2,3,4,5,6,7,8,9,10],xl:'광고비 (x, 백만 원)',yl:'매출액 (y, 천만 원)',label:'광고비와 매출액 산점도와 내 직선',drag:1});
  let s=P.head;const L=(x0,x1,aa,bb,cls)=>`<line class="${cls}" x1="${P.X(x0)}" y1="${P.Y(aa*x0+bb)}" x2="${P.X(x1)}" y2="${P.Y(aa*x1+bb)}"/>`;
  s+=`<clipPath id="cp-ad"><rect x="${P.m.l}" y="${P.m.t}" width="${P.iw}" height="${P.ih}"/></clipPath><g clip-path="url(#cp-ad)">`;
  if(showFit)s+=L(0,10,AD_FIT.a,AD_FIT.b,'line-ghost');
  if(showMy&&showRes)AD_X.forEach((x,k)=>{s+=`<line class="res" x1="${P.X(x)}" x2="${P.X(x)}" y1="${P.Y(AD_Y[k])}" y2="${P.Y(a*x+b)}"/>`});
  s+=(showMy?L(0,10,a,b,'line-my'):'')+'</g>';
  AD_X.forEach((x,k)=>{s+=`<circle class="dot ring2" cx="${P.X(x)}" cy="${P.Y(AD_Y[k])}" r="5.5" data-tip="${esc(`광고비 ${x}백만 원 · 매출액 ${AD_Y[k]}천만 원`+(showMy?`\n내 직선의 예측 ${fx(a*x+b)} · 오차 ${fx(AD_Y[k]-(a*x+b))}`:''))}"/>`});
  if(showMy)[1,9].forEach((x,i)=>{const y=clamp(a*x+b,-2,20);s+=`<circle class="handle" data-drag="h" data-i="${i}" cx="${P.X(x)}" cy="${P.Y(y)}" r="8"/>`});
  s+=P.end;
  if(!showMy)return `<div class="sim2"><div>${s}<p class="hint">먼저 점들의 모양을 살펴보고, 어떤 직선이 데이터를 잘 대표할지 생각해 보세요.</p></div><div>
    <p class="note"><b>내 직선 보기</b>를 켜거나 <b>기울기 a</b>·<b>절편 b</b> 슬라이더를 움직이면 직선이 나타납니다.</p>
    ${gridTable(['광고비 x (백만 원)','매출액 y (천만 원)'],AD_X.map((x,k)=>[`<b>${x}</b>`,AD_Y[k]]))}</div></div>`;
  const my=mseOf(AD_X,AD_Y,a,b);
  return `<div class="sim2"><div>${s}<p class="hint">흰 손잡이 두 개를 위아래로 끌면 직선이 움직입니다. 빨간 세로선이 각 점의 오차입니다.</p></div><div>
    <div class="big-eq">내 직선 &nbsp;<em>${eqStr(a,b)}</em></div>
    <div class="stat">${fact('내 직선의 평균제곱오차',fx(my),my<=AD_BEST*1.1?'최적에 거의 도달!':'더 줄일 수 있어요',my<=AD_BEST*1.1?'':'hl')}${fact('최적 직선의 평균제곱오차',fx(AD_BEST),eqStr(AD_FIT.a,AD_FIT.b))}</div>
    ${(()=>{const es=AD_X.map((x,k)=>AD_Y[k]-(a*x+b));
      return gridTable(['광고비 x (백만 원)','매출액 y (천만 원)','내 예측','오차','오차 제곱'],[...AD_X.map((x,k)=>{const e=es[k];return[`<b>${x}</b>`,AD_Y[k],fx(a*x+b),{v:fx(e),cls:Math.abs(e)>3?'c-out':''},{v:fx(e*e),cls:Math.abs(e)>3?'c-out':''}]}),
        {cls:'sum-row',cells:['<b>합계</b>','','',`<b>${fx(sum(es))}</b><small>오차의 합</small>`,`<b>${fx(sum(es.map(e=>e*e)))}</b><small>오차제곱합</small>`]}])})()}
    <p class="note">광고비가 10(1,000만 원)이면 내 직선은 매출액을 <b>${fx(a*10+b,1)}</b>(천만 원)으로 예측합니다.</p></div></div>`;
};
VIEWS.click=()=>{
  const pts=A().pts,P=plot({id:'p-click',w:520,h:340,x:[0,10],y:[0,20],xt:11,yt:5,xl:'광고비 (x, 백만 원)',yl:'매출액 (y, 천만 원)',click:'add',label:'광고비와 매출액 점을 찍어 회귀 직선 만들기'});
  let s=P.head,info='';
  if(pts.length>=2){const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),f=linfit(xs,ys),r=pearson(xs,ys).r;
    s+=`<clipPath id="cp-cl"><rect x="${P.m.l}" y="${P.m.t}" width="${P.iw}" height="${P.ih}"/></clipPath><line class="line-fit" clip-path="url(#cp-cl)" x1="${P.X(0)}" y1="${P.Y(f.b)}" x2="${P.X(10)}" y2="${P.Y(f.a*10+f.b)}" pointer-events="none"/>`;
    info=`<div class="big-eq">회귀 직선 &nbsp;<em>${eqStr(f.a,f.b)}</em></div><div class="stat">${fact('점의 개수',pts.length)}${fact('기울기',fx(f.a),f.a>0?'양의 관계':f.a<0?'음의 관계':'관계 없음')}${fact('상관관계 계수',`r = ${fx(r)}`)}</div>`}
  else info=`<p class="note">그래프의 빈 곳을 눌러 점을 찍어 보세요. 점이 2개 이상이면 회귀 직선이 그려집니다.</p>`;
  pts.forEach((p,k)=>{s+=`<circle class="dot ring2" data-pick="rm" data-i="${k}" cx="${P.X(p[0])}" cy="${P.Y(p[1])}" r="6" data-tip="광고비 ${p[0]}백만 원 · 매출액 ${p[1]}천만 원 · 누르면 지워져요"/>`});
  s+=P.end;
  return `<div class="sim2"><div>${s}</div><div>${info}<p class="hint">찍은 점을 다시 누르면 지워집니다. 점 하나를 멀리 찍으면 직선이 얼마나 움직이는지도 살펴보세요.</p></div></div>`;
};
// 3단계: 상관관계
VIEWS.corr=()=>{
  const c=corrPts(),pr=pearson(c.xs,c.ys),r=pr.r,f=linfit(c.xs,c.ys),n=CZ.a.length;
  const P=plot({id:'p-cor',w:460,h:400,x:[-4.2,4.2],y:[-4.2,4.2],xt:[-4,-3,-2,-1,0,1,2,3,4],yt:[-4,-3,-2,-1,0,1,2,3,4],xl:'x',yl:'y',label:'상관관계 산점도',max:520});
  let s=P.head;
  if(A().showLine)s+=`<clipPath id="cp-co"><rect x="${P.m.l}" y="${P.m.t}" width="${P.iw}" height="${P.ih}"/></clipPath><line class="line-fit" clip-path="url(#cp-co)" x1="${P.X(-4.2)}" y1="${P.Y(f.a*-4.2+f.b)}" x2="${P.X(4.2)}" y2="${P.Y(f.a*4.2+f.b)}"/>`;
  for(let k=0;k<n;k++)s+=`<circle class="dot" cx="${P.X(c.xs[k])}" cy="${P.Y(c.ys[k])}" r="4.5" opacity=".75"/>`;
  if(A().outlier)s+=`<circle data-drag="out" cx="${P.X(A().ox)}" cy="${P.Y(A().oy)}" r="8" class="ring2" style="fill:var(--cen)" data-tip="끌어서 옮겨 보세요"/>`;
  s+=P.end;
  const lab=Math.abs(r)<.3?['약한 상관관계','']:r>0?['양의 상관관계','pos']:['음의 상관관계','neg'];
  const str=Math.abs(r)>=.7?'강함':Math.abs(r)>=.3?'보통':'약함';
  const r0=A().outlier?pearson(CZ.a,CZ.a.map((v,k)=>A().r*v+Math.sqrt(Math.max(0,1-A().r**2))*CZ.b[k])).r:null;
  return `<div class="sim2"><div>${s}</div><div>
    <div class="stat">${fact('상관관계 계수',`r = ${fx(r)}`,`점 ${c.xs.length}개로 계산`,'hl')}${fact('방향 · 강도',`<span class="badge ${lab[1]}">${lab[0]}</span>`,`강도: ${str}`)}</div>
    ${A().outlier?`<p class="note">빨간 점이 없을 때 r = <b>${fx(r0)}</b> → 빨간 점 하나로 <b>${fx(r-r0)}</b>만큼 달라졌습니다.</p>`:''}
    ${gridTable(['상관관계 계수','의미'],[[{v:'0.7 이상',cls:'l'},'강한 양의 상관관계'],[{v:'0.3 ~ 0.7',cls:'l'},'양의 상관관계'],[{v:'−0.3 ~ 0.3',cls:'l'},'약한 상관관계 (±0.3 미만)'],[{v:'−0.7 ~ −0.3',cls:'l'},'음의 상관관계'],[{v:'−0.7 이하',cls:'l'},'강한 음의 상관관계']].map((row,i)=>({cells:row,cls:[r>=.7,r>=.3&&r<.7,Math.abs(r)<.3,r<=-.3&&r>-.7,r<=-.7][i]?'sel':''})),{left:1})}
  </div></div>`;
};
// 4단계: 오차
function errStat(){const {ea,eb,ys}=A();const pr=E_X.map(x=>ea*x+eb),er=ys.map((y,k)=>y-pr[k]);return{pr,er,mae:mean(er.map(Math.abs)),mse:mean(er.map(e=>e*e))}}
VIEWS.err=()=>{
  const {ea,eb,ys,sq}=A(),e=errStat(),P=plot({id:'p-err',w:440,h:440,m:{l:40,b:40},x:[0,10],y:[0,10],xt:11,yt:[0,2,4,6,8,10],xl:'x',yl:'y',label:'실젯값과 예측값, 오차',max:460});
  let s=P.head+`<clipPath id="cp-er"><rect x="${P.m.l}" y="${P.m.t}" width="${P.iw}" height="${P.ih}"/></clipPath><g clip-path="url(#cp-er)">`;
  s+=`<line class="line-fit" style="stroke:var(--s2)" x1="${P.X(0)}" y1="${P.Y(eb)}" x2="${P.X(10)}" y2="${P.Y(10*ea+eb)}"/>`;
  E_X.forEach((x,k)=>{const y=ys[k],p=e.pr[k],d=e.er[k];
    if(sq&&d){const side=Math.abs(d),x0=P.X(x),w=P.X(x+side)-P.X(x);s+=`<rect class="res-sq" x="${x0}" y="${P.Y(Math.max(y,p))}" width="${w}" height="${P.Y(Math.min(y,p))-P.Y(Math.max(y,p))}"/><text class="lbl" x="${x0+w/2}" y="${P.Y((y+p)/2)+4}" text-anchor="middle">${fx(d*d,d*d%1?2:0)}</text>`}
    s+=`<line class="res" x1="${P.X(x)}" x2="${P.X(x)}" y1="${P.Y(y)}" y2="${P.Y(p)}"/>`});
  s+='</g>';
  E_X.forEach((x,k)=>{const y=ys[k],p=e.pr[k];
    s+=`<circle cx="${P.X(x)}" cy="${P.Y(p)}" r="5" style="fill:var(--s3)" class="ring2"/><circle data-drag="ey" data-i="${k}" cx="${P.X(x)}" cy="${P.Y(y)}" r="7.5" class="ring2" style="fill:var(--s5)" data-tip="${esc(`실젯값 (${x}, ${y})\n위아래로 끌어 보세요`)}"/>`;
    s+=`<text class="lbl" x="${P.X(x)-10}" y="${P.Y(y)+(y>=p?-10:16)}" text-anchor="middle">(${x}, ${y})</text>`});
  s+=`<circle cx="${P.m.l+12}" cy="${P.m.t+10}" r="5" style="fill:var(--s5)"/><text class="leg" x="${P.m.l+22}" y="${P.m.t+14}">실제 데이터 (끌기)</text><circle cx="${P.m.l+12}" cy="${P.m.t+28}" r="5" style="fill:var(--s3)"/><text class="leg" x="${P.m.l+22}" y="${P.m.t+32}">입력에 대한 예측</text>`+P.end;
  return `<div class="sim2"><div>${s}</div><div>
    <div class="big-eq">회귀 분석 모델 &nbsp;<em>${eqStr(ea,eb)}</em></div>
    ${gridTable(['x','실젯값','예측값','오차','|오차|','오차²'],E_X.map((x,k)=>[`<b>${x}</b>`,ys[k],fx(e.pr[k]),{v:fx(e.er[k]),cls:Math.abs(e.er[k])>=3?'c-out':''},fx(Math.abs(e.er[k])),fx(e.er[k]**2)]))}
    <div class="stat">${fact('평균절대오차 MAE',fx(e.mae,3).replace(/\.?0+$/,''),'|오차|의 평균')}${fact('평균제곱오차 MSE',fx(e.mse,3).replace(/\.?0+$/,''),'오차²의 평균','hl')}</div>
  </div></div>`;
};
// 5단계: 결정 계수
VIEWS.r2=()=>{
  const st=r2Stat(),{ys,f,my}=st,v=A().r2view;
  const lo=Math.min(-2,...ys)-1,hi=Math.max(24,...ys)+1;
  const P=plot({id:'p-r2',w:520,h:360,x:[0,10.5],y:[lo,hi],xl:'x',yl:'y',label:'결정 계수 시뮬레이션'});
  let s=P.head+`<clipPath id="cp-r2"><rect x="${P.m.l}" y="${P.m.t}" width="${P.iw}" height="${P.ih}"/></clipPath><g clip-path="url(#cp-r2)">`;
  R2D.xs.forEach((x,k)=>{const to=v==='sst'?my:f.a*x+f.b;s+=`<line class="res" style="${v==='sst'?'stroke:var(--s4)':''}" x1="${P.X(x)}" x2="${P.X(x)}" y1="${P.Y(ys[k])}" y2="${P.Y(to)}"/>`});
  s+=`<line class="line-mean" x1="${P.X(0)}" x2="${P.X(10.5)}" y1="${P.Y(my)}" y2="${P.Y(my)}"/><line class="line-fit" x1="${P.X(0)}" y1="${P.Y(f.b)}" x2="${P.X(10.5)}" y2="${P.Y(f.a*10.5+f.b)}"/></g>`;
  R2D.xs.forEach((x,k)=>{s+=`<circle class="dot ring2" cx="${P.X(x)}" cy="${P.Y(ys[k])}" r="4.5"/>`});
  s+=`<line class="line-fit" x1="${P.m.l+10}" x2="${P.m.l+34}" y1="${P.m.t+10}" y2="${P.m.t+10}"/><text class="leg" x="${P.m.l+40}" y="${P.m.t+14}">회귀 직선</text><line class="line-mean" x1="${P.m.l+10}" x2="${P.m.l+34}" y1="${P.m.t+28}" y2="${P.m.t+28}"/><text class="leg" x="${P.m.l+40}" y="${P.m.t+32}">y의 평균</text>`+P.end;
  const W=100,bar=(val,max,col)=>`<div style="height:14px;border-radius:4px;background:${col};width:${Math.max(.5,val/max*W)}%"></div>`;
  return `<div class="sim2"><div>${s}</div><div>
    <div class="stat">${fact('결정 계수',`R² = ${fx(st.r2,3)}`,st.r2>=.7?'설명력이 높아요':st.r2>=.3?'어느 정도 설명해요':'거의 설명하지 못해요','hl')}</div>
    <div style="display:grid;gap:6px;font-size:15px"><span>평균선과의 차이² 합 <b>SST = ${fx(st.sst,1)}</b></span>${bar(st.sst,st.sst,'var(--s4)')}<span>회귀선과의 차이² 합 <b>SSE = ${fx(st.sse,1)}</b></span>${bar(st.sse,st.sst,'var(--out-ink)')}</div>
    <p class="note">R² = 1 − SSE ÷ SST = 1 − ${fx(st.sse,1)} ÷ ${fx(st.sst,1)} = <b>${fx(st.r2,3)}</b><br>회귀 직선이 평균선보다 오차를 얼마나 줄였는지 나타냅니다.</p>
  </div></div>`;
};
VIEWS.ml=()=>{
  const sp=mlSplit(A().seed),P=plot({id:'p-ml',w:520,h:330,x:[0,10.5],y:[-2,22],xl:'x',yl:'y',label:'학습 데이터와 평가 데이터'});
  let s=P.head+`<line class="line-fit" x1="${P.X(0)}" y1="${P.Y(sp.f.b)}" x2="${P.X(10.5)}" y2="${P.Y(sp.f.a*10.5+sp.f.b)}"/>`;
  ML.xs.forEach((x,i)=>{const te=sp.te.has(i),y=ML.ys[i];if(te)s+=`<line class="res" x1="${P.X(x)}" x2="${P.X(x)}" y1="${P.Y(y)}" y2="${P.Y(sp.f.a*x+sp.f.b)}"/>`;s+=te?`<rect x="${P.X(x)-5}" y="${P.Y(y)-5}" width="10" height="10" rx="2" class="ring2" style="fill:var(--s2)"/>`:`<circle class="dot ring2" cx="${P.X(x)}" cy="${P.Y(y)}" r="5"/>`});
  s+=`<circle cx="${P.m.l+12}" cy="${P.m.t+10}" r="5" class="dot"/><text class="leg" x="${P.m.l+22}" y="${P.m.t+14}">학습 데이터 21개 (70%)</text><rect x="${P.m.l+7}" y="${P.m.t+23}" width="10" height="10" rx="2" style="fill:var(--s2)"/><text class="leg" x="${P.m.l+22}" y="${P.m.t+32}">평가 데이터 9개 (30%)</text>`+P.end;
  const h=A().hist;
  return `<div class="sim2"><div>${s}</div><div>
    <div class="big-eq">학습한 모델 &nbsp;<em>${eqStr(sp.f.a,sp.f.b)}</em></div>
    <div class="stat">${fact('학습 데이터 MSE',fx(sp.trMSE))}${fact('평가 데이터 MSE',fx(sp.teMSE),'모델의 성능 평가','hl')}</div>
    ${h.length?gridTable(['나눈 횟수','학습 MSE','평가 MSE'],h.map((r,i)=>[`<b>${i+1}번째</b>`,fx(r[0]),fx(r[1])])):''}
    <p class="hint">학습 데이터(파란 점)로 직선을 구하고, 학습에 쓰지 않은 평가 데이터(주황 네모)의 오차로 성능을 평가합니다.</p>
  </div></div>`;
};

/* ---------- 입력 · 동작 ---------- */
const IN={
  a:{get:()=>A().a,set:v=>{A().a=v;A().showMy=true},fmt:v=>fx(v)},b:{get:()=>A().b,set:v=>{A().b=v;A().showMy=true},fmt:v=>fx(v,1)},
  showMy:{get:()=>A().showMy,set:v=>A().showMy=v},
  showRes:{get:()=>A().showRes,set:v=>A().showRes=v},showFit:{get:()=>A().showFit,set:v=>A().showFit=v},
  r:{get:()=>A().r,set:v=>A().r=v,fmt:v=>fx(v)},outlier:{get:()=>A().outlier,set:v=>A().outlier=v},showLine:{get:()=>A().showLine,set:v=>A().showLine=v},
  ea:{get:()=>A().ea,set:v=>A().ea=v,fmt:v=>fx(v)},eb:{get:()=>A().eb,set:v=>A().eb=v,fmt:v=>fx(v,1)},sq:{get:()=>A().sq,set:v=>A().sq=v},
  noise:{get:()=>A().noise,set:v=>A().noise=v,fmt:v=>`σ ${fx(v*v,1)}`},
  hidden:{get:()=>A().hidden,set:v=>A().hidden=v}
};
const ACT={
  role:d=>{A().roles[d.k]=+d.v},
  cause:d=>{A().cause[d.k]=d.v},
  best:()=>{A().a=AD_FIT.a;A().b=AD_FIT.b;A().showMy=true},
  clear:()=>{A().pts=[]},
  loadad:()=>{A().pts=AD_X.map((x,k)=>[x,AD_Y[k]])},
  rand:()=>{const r=rng(Date.now()%1e6),a=r()*1.6-.8;A().pts=Array.from({length:20},()=>{const x=Math.round((.5+r()*9)*10)/10;return[x,Math.round(clamp(10+2*a*(x-5)+gauss(r)*2.4,.4,19.6)*10)/10]})},
  rpre:d=>{A().r=+d.v},
  ebest:()=>{const f=linfit(E_X,A().ys);A().ea=f.a;A().eb=f.b},
  r2pre:d=>{A().noise=+d.v},
  r2view:d=>{A().r2view=d.v},
  resplit:()=>{const sp=mlSplit(A().seed);A().hist.push([sp.trMSE,sp.teMSE]);if(A().hist.length>8)A().hist.shift();A().seed++}
};
const DRAG={
  h:(p,i)=>{const {a,b}=A();let y1=a+b,y9=9*a+b;if(i===0)y1=clamp(p.y,-2,20);else y9=clamp(p.y,-2,20);const na=(y9-y1)/8;A().a=Math.round(na*100)/100;A().b=Math.round((y1-na)*10)/10},
  out:p=>{A().ox=clamp(p.x,-4,4);A().oy=clamp(p.y,-4,4)},
  ey:(p,i)=>{A().ys[i]=clamp(Math.round(p.y),0,10)}
};
const CLICK={add:p=>{if(A().pts.length>=60)return false;A().pts.push([Math.round(clamp(p.x,0,10)*10)/10,Math.round(clamp(p.y,0,20)*10)/10])}};
const PICK={rm:i=>{A().pts.splice(i,1)}};

/* ---------- 먼저 알아보기: 회귀 분석 사용 예시 그림 ---------- */
const EX_FIG=`<figure class="ex-fig" aria-label="스마트폰 회사가 광고비와 구매 후기로 매출액을 예측하는 회귀 분석 예시">
  <p class="ex-who"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="2" width="12" height="20" rx="2.5"/><line x1="10" y1="18.5" x2="14" y2="18.5"/></svg>스마트폰 회사의 궁금증</p>
  <div class="ex-row">
    <div class="ex-ins">
      <div class="ex-card x"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M6 16h6l14-8v24l-14-8H6z" class="f"/><path d="M12 24l3 9h4l-3-9" class="f"/><path d="M30 14q4 6 0 12" class="s"/><path d="M33 10q7 10 0 20" class="s"/></svg><b>광고비</b><small>얼마나 썼을까?</small></div>
      <div class="ex-card x"><svg viewBox="0 0 40 40" aria-hidden="true"><path d="M5 7h30v20H17l-8 7v-7H5z" class="f"/><path d="M13 14l1.5 3 3.3.5-2.4 2.3.6 3.3-3-1.6-3 1.6.6-3.3-2.4-2.3 3.3-.5z" class="star"/><path d="M27 14l1.5 3 3.3.5-2.4 2.3.6 3.3-3-1.6-3 1.6.6-3.3-2.4-2.3 3.3-.5z" class="star"/></svg><b>상품 구매 후기</b><small>평이 좋을까?</small></div>
    </div>
    <span class="ex-arr" aria-hidden="true">→</span>
    <div class="ex-card mid"><svg viewBox="0 0 90 60" aria-hidden="true"><line x1="8" y1="54" x2="86" y2="54" class="ax"/><line x1="8" y1="4" x2="8" y2="54" class="ax"/><circle cx="18" cy="45" r="2.6"/><circle cx="28" cy="40" r="2.6"/><circle cx="36" cy="43" r="2.6"/><circle cx="46" cy="32" r="2.6"/><circle cx="56" cy="30" r="2.6"/><circle cx="64" cy="21" r="2.6"/><circle cx="76" cy="17" r="2.6"/><line x1="12" y1="48" x2="84" y2="13" class="ln"/></svg><b>회귀 분석</b><small>관계를 찾아 예측하기</small></div>
    <span class="ex-arr" aria-hidden="true">→</span>
    <div class="ex-card y"><svg viewBox="0 0 40 40" aria-hidden="true"><rect x="5" y="26" width="7" height="9" rx="1" class="f"/><rect x="16" y="19" width="7" height="16" rx="1" class="f"/><rect x="27" y="11" width="7" height="24" rx="1" class="f"/><path d="M6 19L17 11l6 4 11-10" class="s"/><path d="M28 5h6v6" class="s"/></svg><b>매출액</b><small>얼마나 늘어날까?</small></div>
  </div>
  <figcaption>광고비와 구매 후기가 매출액에 <b>얼마나</b> 영향을 주는지 알아내면, 앞으로의 매출액도 예측할 수 있어요</figcaption>
</figure>`;

/* ---------- 5단계: 두 가지 회귀 분석 그림 ---------- */
// 같은 광고비·매출액 데이터로 두 방법을 나란히 보여 줍니다
const CMP_TEST=[1,4,7]; // 평가 데이터로 떼어 둘 점 (광고비 2, 5, 8)
function cmpFig(){
  const sx=x=>34+x*21,sy=y=>148-y*6.3,W=260,H=190;
  const axes=`<line class="cf-ax" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(10.4)}" y2="${sy(0)}"/><line class="cf-ax" x1="${sx(0)}" y1="${sy(0)}" x2="${sx(0)}" y2="${sy(21)}"/>`+
    [0,5,10].map(x=>`<text class="cf-t" x="${sx(x)}" y="${sy(0)+13}" text-anchor="middle">${x}</text>`).join('')+
    [0,10,20].map(y=>`<text class="cf-t" x="${sx(0)-6}" y="${sy(y)+4}" text-anchor="end">${y}</text>`).join('')+
    `<text class="cf-t" x="${sx(10.4)}" y="${H-6}" text-anchor="end">광고비(백만 원)</text><text class="cf-t" x="${sx(0)}" y="13" text-anchor="start">매출액(천만 원)</text>`;
  const ln=(f,cls)=>`<line class="${cls}" x1="${sx(.3)}" y1="${sy(f.a*.3+f.b)}" x2="${sx(9.8)}" y2="${sy(f.a*9.8+f.b)}"/>`;
  // 통계적: 전체 데이터
  const sse=AD_X.reduce((s,x,k)=>s+(AD_Y[k]-(AD_FIT.a*x+AD_FIT.b))**2,0),sst=AD_Y.reduce((s,y)=>s+(y-AD_FIT.my)**2,0),r2=1-sse/sst;
  const st=`<svg viewBox="0 0 ${W} ${H}" class="cf-svg" role="img" aria-label="전체 데이터 9개로 구한 회귀 직선">${axes}${ln(AD_FIT,'cf-fit')}${AD_X.map((x,k)=>`<circle class="cf-all" cx="${sx(x)}" cy="${sy(AD_Y[k])}" r="4.5"/>`).join('')}<rect class="cf-tag" x="${sx(.6)}" y="${sy(19.6)}" width="78" height="22" rx="6"/><text class="cf-tagt" x="${sx(.6)+39}" y="${sy(19.6)+15}" text-anchor="middle">R² = ${fx(r2)}</text></svg>`;
  // 기계학습: 학습 / 평가로 나누기
  const trX=[],trY=[],teX=[],teY=[];AD_X.forEach((x,k)=>{(CMP_TEST.includes(k)?teX:trX).push(x);(CMP_TEST.includes(k)?teY:trY).push(AD_Y[k])});
  const tf=linfit(trX,trY),mse=mseOf(teX,teY,tf.a,tf.b);
  const ml=`<svg viewBox="0 0 ${W} ${H}" class="cf-svg" role="img" aria-label="학습 데이터로 만든 직선을 평가 데이터로 시험하기">${axes}${ln(tf,'cf-fit')}${teX.map((x,k)=>`<line class="cf-err" x1="${sx(x)}" x2="${sx(x)}" y1="${sy(teY[k])}" y2="${sy(tf.a*x+tf.b)}"/>`).join('')}${trX.map((x,k)=>`<circle class="cf-tr" cx="${sx(x)}" cy="${sy(trY[k])}" r="4.5"/>`).join('')}${teX.map((x,k)=>`<circle class="cf-te" cx="${sx(x)}" cy="${sy(teY[k])}" r="5"/>`).join('')}<rect class="cf-tag" x="${sx(.6)}" y="${sy(19.6)}" width="112" height="22" rx="6"/><text class="cf-tagt" x="${sx(.6)+56}" y="${sy(19.6)+15}" text-anchor="middle">평가 MSE = ${fx(mse)}</text></svg>`;
  const flow=xs=>`<ol class="cf-flow">${xs.map(t=>`<li>${t}</li>`).join('')}</ol>`;
  return `<div class="cf">
  <figure class="cf-col st"><figcaption><b>통계적 회귀 분석</b><span>관계를 설명</span></figcaption>
    ${flow(['전체 데이터 9개 모두 사용','회귀 직선 구하기','R²로 설명력 확인'])}${st}
    <p class="cf-leg"><span><i class="d all"></i>전체 데이터</span><span><i class="l"></i>회귀 직선</span></p>
    <p class="cf-say">“광고비가 100만 원 늘면 매출액이 약 <b>${(Math.round(AD_FIT.a*100)*10).toLocaleString()}만 원</b> 늘어나는 관계가 있고, 이 직선이 매출액 변화의 약 <b>${Math.round(r2*100)}%</b>를 설명한다”</p></figure>
  <figure class="cf-col ml"><figcaption><b>기계학습 기반 회귀 분석</b><span>새 값을 예측</span></figcaption>
    ${flow(['학습 6개 · 평가 3개로 나누기','학습 데이터로만 직선 만들기','평가 데이터로 MSE 확인'])}${ml}
    <p class="cf-leg"><span><i class="d tr"></i>학습 데이터</span><span><i class="d te"></i>평가 데이터</span><span><i class="e"></i>예측 오차</span></p>
    <p class="cf-say">“직선을 만들 때 보지 않은 평가 데이터에서 오차가 얼마나 작은지로 <b>예측 성능</b>을 판단한다”</p></figure>
</div>`;
}

/* ---------- 단계 ---------- */
const STEPS=[
{title:'회귀 분석 모델 미리 보기',short:'먼저 알아보기',page:'110–113쪽',
  goal:'회귀 분석이 어떤 문제를 해결하는 방법인지 알아보고, 이 앱에서 어떤 시뮬레이션을 하게 되는지 살펴보세요.',
  concept:'<p><b>회귀 분석</b>은 변수와 변수 사이의 관계를 파악하고, 목적에 맞는 결과를 예측하기 위해 사용하는 분석 방법입니다. 예를 들어 스마트폰 회사는 광고비나 상품 구매 후기가 매출액 증가에 얼마나 영향을 미치는지 알고 싶을 때 회귀 분석을 사용합니다.</p>'+EX_FIG,
  render:()=>card('이 앱에서 해 볼 시뮬레이션',`<ol class="proc">${[['독립 · 종속 변수','변수 고르기 · 상관/인과 구분',1],['선형 회귀 분석','직선 움직이기 · 점 찍기',2],['상관관계 분석','r 바꾸기 · 이상치 · 숨은 원인',3],['오차','MAE · MSE 비교',4],['결정 계수','R² · 학습/평가',5]].map(([t,s,to])=>`<li><button type="button" data-act="go" data-step="${to}"><b>${t}</b><small>${s}</small></button></li>`).join('')}</ol><p class="note">각 단계 끝에는 <b>연습문제</b>가 있습니다. 연습문제를 모두 맞히면 단계에 ✓ 표시가 됩니다.</p>`,{sub:'단계를 누르면 해당 시뮬레이션으로 이동합니다.'})},
{title:'독립 변수와 종속 변수',short:'독립·종속 변수',page:'110쪽',quiz:'var',
  goal:'어떤 변수가 영향을 주고 어떤 변수가 영향을 받는지 구분해 보고, 상관관계와 인과 관계의 차이를 알아봅니다.',
  concept:'<p>다른 변수에 영향을 주는 변수를 <b>독립 변수</b>, 다른 변수의 영향을 받으며 예측하고자 하는 변수를 <b>종속 변수</b>라고 합니다. 예를 들어 광고비와 상품 구매 후기는 독립 변수, 매출액은 종속 변수입니다.</p><div class="multi concept-fig"><div class="xs"><span class="vbox x">광고비</span><span class="vbox x">상품 구매 후기</span><small>독립 변수</small></div><span class="arr">→</span><div class="xs"><span class="vbox y">매출액</span><small>종속 변수</small></div></div>',
  render:()=>card('어느 쪽이 독립 변수일까?',slot('pairs'),{sub:'두 변수 중 다른 쪽에 영향을 주는 변수(독립 변수)를 눌러 보세요.'})+
    '<div class="concept wide-concept"><p><b>상관관계</b>는 한 변수의 변화에 따라 다른 변수도 변화하는 관계이고, <b>인과 관계</b>는 어떤 변수가 다른 변수의 원인이 되는 관계입니다. 상관관계만으로는 인과 관계를 정확히 밝히지 못합니다.</p><p class="concept-more">3단계 <b>더 알아보기</b>에서 상관관계 계수로 직접 확인해 봅니다.</p></div>'+
    card('상관관계일까, 인과 관계일까?',slot('cause'),{sub:'두 변수가 함께 변하기만 하는지(상관관계), 한쪽이 원인인지(인과 관계) 골라 보세요.'}),
  calc(){const r=A().roles,lines=PAIRS.map((p,k)=>r[k]===undefined?`${p[0]} · ${p[1]} → (아직 고르지 않음)`:`${p[r[k]]} → ${p[1-r[k]]}   ${r[k]===p[2]?'✓':'✗ 다시 생각해 보기'}`);
    const c=A().cause,cl=CAUSE.map((q,k)=>c[k]===undefined?`${k+1}. (아직 고르지 않음)`:`${k+1}. ${c[k]==='cor'?'상관관계만':'인과 관계'}   ${c[k]===q.a?'✓':'✗ 다시 생각해 보기'}`);
    return `# 독립 변수 → 종속 변수\n${lines.join('\n')}\n\n# 상관관계일까, 인과 관계일까?\n${cl.join('\n')}\n\n# 숫자로 확인하기: 3단계 더 알아보기 (숨은 원인 찾기)`}},
{title:'선형 회귀 분석',short:'선형 회귀',page:'111쪽',quiz:'lin',
  goal:'한 회사의 광고비와 매출액 데이터로 산점도를 그리고, 데이터를 가장 잘 대표하는 직선을 직접 찾아봅니다.',
  concept:'<p><b>선형 회귀 분석</b>은 변수 사이의 관계를 직선 형태로 표현하여 데이터를 대표하는 하나의 선을 찾는 방법입니다. 두 변수 x와 y의 회귀 분석 모델은 <span class="formula">y = ax + b</span> 형태이며, x는 독립 변수, y는 종속 변수입니다.</p><p>독립 변수가 한 개이면 <b>단순 선형 회귀 분석</b>, 두 개 이상이면 <b>다중 선형 회귀 분석</b>입니다.</p>',
  render:()=>card('광고비와 매출액',`<div class="ctls">${range('a','기울기 a',-2,4,.01)}${range('b','절편 b',-10,12,.1)}</div><div class="ctls"><label class="chk"><input type="checkbox" data-in="showMy">내 직선 보기</label><label class="chk"><input type="checkbox" data-in="showRes">오차(빨간 선) 보기</label><label class="chk"><input type="checkbox" data-in="showFit">최적 직선 겹쳐 보기</label>${btn('best','최적 직선으로 맞추기')}</div>${slot('ad')}`,{sub:'교과서 111쪽 스스로 실습 · 광고비 단위: 백만 원, 매출액 단위: 천만 원 · 직선을 움직여 데이터를 가장 잘 설명하는 직선을 찾아보세요.'})+
    card('오차의 합이 아니라 오차제곱합으로 비교해요',LSQ)+
    card('점을 찍어 회귀 직선 만들기',`<div class="btns">${btn('clear','모두 지우기')}${btn('loadad','광고비·매출액 데이터 불러오기')}${btn('rand','무작위 점 20개')}</div>${slot('click')}`,{sub:'점을 찍을 때마다 최소제곱법으로 회귀 직선을 다시 계산합니다.'})+
    card('단순 선형 회귀 분석과 다중 선형 회귀 분석',`<div class="sim2"><div><p class="eyebrow" style="color:var(--muted)">단순 선형 회귀 분석</p><div class="vdiag"><span class="vbox x">독립 변수</span><span>→</span><span class="vbox y">종속 변수</span></div><p class="note">예: 광고비 → 매출액</p></div><div><p class="eyebrow" style="color:var(--muted)">다중 선형 회귀 분석</p><div class="multi"><div class="xs"><span class="vbox x">독립 변수 A</span><span class="vbox x">독립 변수 B</span><span class="vbox x">독립 변수 C</span></div><span>→</span><span class="vbox y">종속 변수</span></div><p class="note">예: 광고비 · 후기 수 · 가격 → 매출액</p></div></div>`,{sub:'선형 회귀 분석의 종류'}),
  calc(){const f=AD_FIT,{a,b}=A();
    return `# 최소제곱법으로 구한 회귀 직선 (광고비·매출액 데이터 9개)
# 단위: 광고비 x = 백만 원, 매출액 y = 천만 원
x 평균 = (1+2+…+9) ÷ 9 = ${fx(f.mx,3)}
y 평균 = ${sum(AD_Y)} ÷ 9 = ${fx(f.my,3)}
Σ(x−x̄)(y−ȳ) = ${fx(f.sxy,3)}
Σ(x−x̄)²     = ${fx(f.sxx,3)}
기울기 a = ${fx(f.sxy,3)} ÷ ${fx(f.sxx,3)} = ${fx(f.a,4)}
절편   b = ȳ − a·x̄ = ${fx(f.my,3)} − ${fx(f.a,4)} × ${fx(f.mx,3)} = ${fx(f.b,4)}
→ ${eqStr(f.a,f.b,4)}
최소 평균제곱오차 = ${fx(AD_BEST,4)}${A().showMy?`

# 내 직선 ${eqStr(a,b)}
 x   y   예측     오차    오차²
${AD_X.map((x,k)=>{const p=a*x+b,e=AD_Y[k]-p;return`${String(x).padStart(2)}  ${String(AD_Y[k]).padStart(2)}  ${fx(p).padStart(6)}  ${fx(e).padStart(6)}  ${fx(e*e).padStart(7)}`}).join('\n')}
평균제곱오차 = ${fx(mseOf(AD_X,AD_Y,a,b),4)}`:`

# 내 직선: 아직 그리지 않았어요
# 내 직선 보기를 켜거나 슬라이더를 움직여 보세요`}`+(A().pts.length>=2?(()=>{const xs=A().pts.map(p=>p[0]),ys=A().pts.map(p=>p[1]),g=linfit(xs,ys);return`\n\n# 내가 찍은 점 ${xs.length}개의 회귀 직선\n기울기 a = ${fx(g.sxy,3)} ÷ ${fx(g.sxx,3)} = ${fx(g.a,4)}\n절편   b = ${fx(g.my,3)} − ${fx(g.a,4)} × ${fx(g.mx,3)} = ${fx(g.b,4)}`})():'')}},
{title:'상관관계 분석',short:'상관관계',page:'112쪽',quiz:'cor',
  goal:'상관관계 계수를 바꾸어 가며 산점도의 모양이 어떻게 달라지는지 관찰하고, 점 하나가 상관관계 계수에 주는 영향과 상관관계가 강해도 인과 관계라고 할 수 없는 이유를 확인합니다.',
  concept:'<p><b>상관관계 계수</b>는 상관관계의 강도를 나타내는 수치입니다. x가 증가할 때 y도 증가하면 <b>양의 상관관계</b>, y가 감소하면 <b>음의 상관관계</b>입니다. 연속형 변수의 상관관계 계수를 <b>피어슨 상관관계 계수</b>라고 하며, ±0.3 미만이면 약한 상관관계를 뜻합니다.</p>',
  render:()=>card('상관관계 계수 바꿔 보기',`<div class="ctls">${range('r','목표 상관관계 계수',-1,1,.05)}${seg('rpre',[['0.8','양의 상관관계'],['0','상관관계 없음'],['-0.8','음의 상관관계']],'')}</div><div class="ctls"><label class="chk"><input type="checkbox" data-in="showLine">회귀 직선 보기</label><label class="chk"><input type="checkbox" data-in="outlier">빨간 점(이상치) 하나 추가하기</label></div>${slot('corr')}`,{sub:'점 50개의 상관관계 계수를 슬라이더로 바꿉니다.'})+
    card('강도는 기울기가 아니라 ‘모여 있는 정도’',guide('상관관계의 강도를 읽는 법',`<p>상관관계 계수가 ±1에 가까울수록 점들이 하나의 직선 주위에 가깝게 모여 있습니다. 0에 가까우면 점들이 둥글게 흩어져 직선으로 나타내기 어렵습니다.</p><p>같은 데이터라도 y의 단위를 바꾸면(예: 원 → 천 원) 직선의 기울기는 달라지지만 상관관계 계수는 그대로입니다. 그래서 강도는 기울기의 크기보다 점들이 직선 주위에 얼마나 모여 있는지로 판단하는 것이 정확합니다.</p>`),{sub:'시뮬레이션에서 r을 0.3, 0.6, 0.9로 바꾸며 점들이 모이는 모습을 비교해 보세요.'})+
    `<details class="guide more-sim"><summary><span>더 알아보기</span>숨은 원인 찾기: 상관관계 ≠ 인과 관계</summary><p class="more-sub">1단계에서 배운 상관관계와 인과 관계를 상관관계 계수로 확인해 봅니다. 여름 60일 동안의 아이스크림 판매량과 물놀이 사고 수입니다 (가상의 데이터).</p><div class="ctls"><label class="chk"><input type="checkbox" data-in="hidden">기온으로 색칠하기</label></div>${slot('hidden')}</details>`,
  calc(){const c=corrPts(),p=pearson(c.xs,c.ys);
    return `# 피어슨 상관관계 계수 (점 ${c.xs.length}개)
x 평균 = ${fx(p.mx,4)}
y 평균 = ${fx(p.my,4)}
Σ(x−x̄)(y−ȳ) = ${fx(p.sxy,2)}
Σ(x−x̄)²     = ${fx(p.sxx,2)}
Σ(y−ȳ)²     = ${fx(p.syy,2)}

r = Σ(x−x̄)(y−ȳ) ÷ √( Σ(x−x̄)² × Σ(y−ȳ)² )
  = ${fx(p.sxy,2)} ÷ √( ${fx(p.sxx,2)} × ${fx(p.syy,2)} )
  = ${fx(p.r,4)}

# 해석
${Math.abs(p.r)<.3?'|r| < 0.3 → 약한 상관관계':p.r>0?'r > 0 → 양의 상관관계 (x가 커지면 y도 커짐)':'r < 0 → 음의 상관관계 (x가 커지면 y는 작아짐)'}${A().outlier?`\n\n# 빨간 점 (${fx(A().ox)}, ${fx(A().oy)}) 을 넣었을 때\n점 하나만으로 r이 달라질 수 있습니다`:''}

# 더 알아보기: 숨은 원인 (60일)
아이스크림 ↔ 물놀이 사고  r = ${fx(pearson(HID.map(d=>d.ice),HID.map(d=>d.acc)).r,3)}
${A().hidden?[1,2,3].map(b=>{const g=HID.filter(d=>band(d.t)===b);return`${['','18°C 미만 ','18~27°C  ','27°C 이상 '][b]} ${g.length}일  r = ${fx(pearson(g.map(d=>d.ice),g.map(d=>d.acc)).r,3)}`}).join('\n')+'\n# 같은 기온끼리는 관계가 약해집니다\n# → 상관관계 ≠ 인과 관계':'# 기온으로 색칠하기를 켜 보세요'}`}},
{title:'오차와 평가 지표',short:'오차',page:'112쪽',quiz:'err',
  goal:'예측값과 실젯값의 차이인 오차를 눈으로 확인하고, 평균절대오차와 평균제곱오차가 어떻게 다르게 계산되는지 비교합니다.',
  concept:'<p><b>오차</b>는 예측값과 실젯값의 차이로, 그 값이 작을수록 좋은 회귀 분석 모델입니다. 오차에 절댓값을 취한 후 평균을 구한 <b>평균절대오차(MAE)</b>와 오차에 제곱을 취한 후 평균을 구한 <b>평균제곱오차(MSE)</b>를 가장 많이 사용합니다.</p>',
  render:()=>card('오차의 표현',`<div class="ctls">${range('ea','기울기 a',-1,2,.05)}${range('eb','절편 b',-5,5,.1)}</div><div class="ctls"><label class="chk"><input type="checkbox" data-in="sq">오차의 제곱을 넓이로 보기</label>${btn('ebest','평균제곱오차가 가장 작은 직선')}</div>${slot('err')}`,{sub:'보라 점(실젯값)은 위아래로 끌 수 있고, 주황 직선(회귀 분석 모델)은 슬라이더로 바꿉니다.'}),
  calc(){const e=errStat(),{ys}=A();
    return `# 회귀 분석 모델 ${eqStr(A().ea,A().eb)}
# 오차 = 실젯값 − 예측값
${E_X.map((x,k)=>`x=${x}: ${ys[k]} − ${fx(e.pr[k])} = ${fx(e.er[k])}`).join('\n')}

# 평균절대오차 (MAE)
(|${e.er.map(v=>fx(v)).join('| + |')}|) ÷ 4
= (${e.er.map(v=>fx(Math.abs(v))).join(' + ')}) ÷ 4
= ${fx(sum(e.er.map(Math.abs)),3)} ÷ 4 = ${fx(e.mae,4)}

# 평균제곱오차 (MSE)
(${e.er.map(v=>`(${fx(v)})²`).join(' + ')}) ÷ 4
= (${e.er.map(v=>fx(v*v)).join(' + ')}) ÷ 4
= ${fx(sum(e.er.map(v=>v*v)),3)} ÷ 4 = ${fx(e.mse,4)}`}},
{title:'결정 계수와 두 가지 회귀 분석',short:'결정 계수',page:'113쪽',quiz:'r2',
  goal:'점들이 흩어진 정도에 따라 결정 계수가 어떻게 바뀌는지 먼저 확인하고, 통계적 회귀 분석과 기계학습 기반 회귀 분석의 차이와 데이터를 나누어 평가하는 방법을 알아봅니다.',
  concept:'<p><b>결정 계수(R²)</b>는 회귀 모델의 설명력을 나타내는 지표로, 독립 변수가 종속 변수를 얼마나 설명하고 있는지 나타냅니다. 0에서 1 사이의 값이며, 1에 가까울수록 정확도가 높습니다.</p><p><b>통계적 회귀 분석</b>은 변수 사이의 관계를 설명하는 데 중점을 두고 결정 계수로, <b>기계학습 기반 회귀 분석</b>은 예측을 잘하는 데 중점을 두고 평가 데이터의 평균제곱오차로 모델을 평가합니다.</p>',
  render:()=>card('결정 계수',`<div class="ctls">${range('noise','흩어진 정도',0,6,.05)}${seg('r2pre',[['0','R² = 1'],['2.2','R² ≈ 0.5'],['5.5','R² ≈ 0']],'')}</div><div class="ctls">${seg('r2view',[['sse','회귀선과의 차이 (SSE)'],['sst','평균선과의 차이 (SST)']],A().r2view)}</div>${slot('r2')}`,{sub:'점 40개가 y = 2x + 1 주위에 흩어진 정도를 바꿉니다.'})+
    card('통계적 회귀 분석과 기계학습 기반 회귀 분석','<div class="def-split"><p class="def-lead">회귀 분석은 <mark class="st">변수와 변수 사이의 관계를 파악</mark>하고, <mark class="ml">목적에 맞는 결과를 예측</mark>하기 위해 사용하는 분석 방법입니다.</p><ul class="def-map"><li class="st"><mark>변수와 변수 사이의 관계를 파악</mark><span>=</span><b>통계적 회귀 분석</b></li><li class="ml"><mark>목적에 맞는 결과를 예측</mark><span>=</span><b>기계학습 기반 회귀 분석</b></li></ul></div>'+cmpFig()+'<div class="tbl-wrap">'+gridTable(['구분','통계적 회귀 분석','기계학습 기반 회귀 분석'],[['<b>목표</b>','변수 사이의 <b>관계를 설명</b>','새 데이터의 값을 <b>잘 예측</b>'],['<b>묻는 질문</b>','광고비가 100만 원 늘면 매출액이 얼마나 늘까? 그 관계를 믿을 수 있나?','다음 달 광고비가 이만큼이면 매출액은 얼마일까?'],['<b>데이터 사용</b>','가진 데이터를 <b>전부</b> 써서 관계를 분석','<b>학습 데이터</b>로 모델을 만들고, 따로 떼어 둔 <b>평가 데이터</b>로 성능 확인'],['<b>대표 평가 지표</b>','<b>결정 계수(R²)</b> · 모델이 데이터를 얼마나 설명하는가 (1에 가까울수록 좋음)','<b>평균제곱오차(MSE)</b> · 처음 보는 데이터에서 오차가 얼마나 작은가 (작을수록 좋음)'],['<b>관심 있는 것</b>','기울기 · 절편의 <b>의미</b> (해석)','예측값의 <b>정확도</b> (성능)']],{left:1})+'</div>'+insight('스스로 점검','데이터의 특성에 따라 다른 성능 지표와 함께 데이터 분석 결과를 평가해야 할 수도 있습니다. 두 방법은 상황에 따라 서로 보완하여 사용합니다.'),{sub:'교과서 113쪽 · 위에서 본 결정 계수는 통계적 회귀 분석의 평가 지표입니다. 두 방법의 차이를 살펴본 뒤, 아래에서 학습·평가 데이터로 기계학습 방식을 확인해 보세요.'})+
    card('기계학습 기반 회귀 분석: 학습 데이터와 평가 데이터',`<div class="btns">${btn('resplit','↻ 무작위로 다시 나누기',{cls:'primary'})}</div>${slot('ml')}`,{sub:'데이터 30개를 학습 70%, 평가 30%로 나눕니다. 나눌 때마다 평가 오차가 어떻게 달라지는지 기록해 보세요.'}),
  calc(){const st=r2Stat(),sp=mlSplit(A().seed);
    return `# 결정 계수 (점 40개, 흩어진 정도 σ = ${fx(A().noise**2,2)})
회귀 직선 ${eqStr(st.f.a,st.f.b,3)}
y 평균 = ${fx(st.my,3)}
SST = Σ(y − ȳ)²  = ${fx(st.sst,2)}   # 평균선과의 차이
SSE = Σ(y − ŷ)²  = ${fx(st.sse,2)}   # 회귀선과의 차이
R² = 1 − SSE ÷ SST
   = 1 − ${fx(st.sse,2)} ÷ ${fx(st.sst,2)}
   = ${fx(st.r2,4)}

# 기계학습 기반 회귀 분석
학습 데이터 21개 → ${eqStr(sp.f.a,sp.f.b,3)}
학습 MSE = ${fx(sp.trMSE,4)}
평가 데이터 9개
평가 MSE = ${fx(sp.teMSE,4)}   # 성능 평가에 사용`}}
];
const GROUPS={0:'시작',1:'01 회귀 분석 모델'};
function FLOW(){const a=A(),c=corrPts(),e=errStat(),st=r2Stat(),ok=PAIRS.filter((p,k)=>a.roles[k]===p[2]).length;
  return[
    {steps:[1,0],l:'독립 · 종속 변수',b:`${ok} / 4 구분`,s:'상관관계 vs 인과 관계',d:quizDone('var'),wait:!ok},
    {steps:[2],l:'선형 회귀 분석',b:a.showMy?eqStr(a.a,a.b,2):'직선 그리기 전',s:a.showMy?`내 직선 MSE ${fx(mseOf(AD_X,AD_Y,a.a,a.b))}`:'광고비 · 매출액 9개',d:quizDone('lin'),wait:!a.showMy},
    {steps:[3],l:'상관관계 분석',b:`r = ${fx(pearson(c.xs,c.ys).r)}`,s:a.hidden?'숨은 원인: 기온':Math.abs(a.r)<.3?'약한 상관관계':a.r>0?'양의 상관관계':'음의 상관관계',d:quizDone('cor')},
    {steps:[4],l:'오차',b:`MSE ${fx(e.mse)}`,s:`MAE ${fx(e.mae)}`,d:quizDone('err')},
    {steps:[5],l:'결정 계수 · 평가',b:`R² = ${fx(st.r2)}`,s:`평가 MSE ${fx(mlSplit(a.seed).teMSE)}`,d:quizDone('r2')}]}
const KEYNOTE=`# 회귀 분석 모델 핵심 정리 (교과서 110–113쪽)

# 1. 독립 변수와 종속 변수
독립 변수 = 다른 변수에 영향을 주는 변수   (예: 광고비)
종속 변수 = 영향을 받으며 예측하려는 변수 (예: 매출액)
상관관계 = 함께 변하는 관계
인과 관계 = 원인과 결과의 관계 (상관관계만으로는 알 수 없음)

# 2. 선형 회귀 분석
y = ax + b   (x: 독립 변수, y: 종속 변수)
독립 변수 1개 → 단순 선형 회귀 분석
독립 변수 2개 이상 → 다중 선형 회귀 분석

# 3. 상관관계 계수 (피어슨)
−1 ≤ r ≤ 1
r > 0 → 양의 상관관계,  r < 0 → 음의 상관관계
|r| < 0.3 → 약한 상관관계

# 4. 오차 = 실젯값 − 예측값
MAE = |오차|의 평균
MSE = 오차²의 평균
→ 작을수록 좋은 회귀 분석 모델

# 5. 결정 계수
R² = 1 − SSE ÷ SST,  0 ≤ R² ≤ 1
1에 가까울수록 설명력이 높음
통계적 회귀 분석 → 결정 계수로 평가
기계학습 기반 회귀 분석 → 평가 데이터의 MSE로 평가`;
