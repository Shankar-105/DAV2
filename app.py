
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Dispatch / Delivery Intelligence", page_icon="▰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#f4f0e9;--ink:#20211f;--muted:#77746e;--paper:#fffdf8;--line:#ded8ce;--red:#e65d43;--blue:#3157a4;--yellow:#f4c95d}
.stApp{background:var(--bg);color:var(--ink)}
.block-container{max-width:1440px;padding:2rem 3.5rem 4rem}
*{font-family:Manrope,sans-serif}
.mono{font-family:"DM Mono",monospace!important}
.mast{display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid var(--ink);padding-bottom:15px;margin-bottom:32px}
.logo{font-weight:800;letter-spacing:-.04em;font-size:21px}
.logo span{color:var(--red)}
.issue{font-family:"DM Mono";font-size:11px;letter-spacing:.08em}
.hero{display:grid;grid-template-columns:1.5fr .7fr;gap:30px;margin-bottom:30px}
.kicker{font-family:"DM Mono";font-size:11px;text-transform:uppercase;letter-spacing:.16em;color:var(--red);margin-bottom:12px}
h1{font-size:clamp(45px,6vw,82px)!important;line-height:.91!important;letter-spacing:-.075em!important;margin:0!important;font-weight:800!important}
.hero p{max-width:760px;color:var(--muted);font-size:17px;line-height:1.7;margin-top:24px}
.stamp{background:var(--ink);color:#fffdf8;padding:25px;border-radius:2px;min-height:220px;display:flex;flex-direction:column;justify-content:space-between}
.stamp .big{font-size:64px;line-height:.9;font-weight:800;letter-spacing:-.07em}
.stamp small{color:#c9c6be;font-family:"DM Mono";font-size:11px}
nav{margin-bottom:30px}
div[data-testid="stHorizontalBlock"] button{border-radius:2px!important}
.nav button{background:transparent!important;border:0!important;border-bottom:2px solid transparent!important;border-radius:0!important;font-family:"DM Mono"!important;font-size:12px!important}
.rule{height:1px;background:var(--ink);opacity:.25;margin:25px 0}
.section{font-size:28px;font-weight:800;letter-spacing:-.05em;margin:28px 0 5px}
.sub{color:var(--muted);margin-bottom:20px}
.card{background:var(--paper);border:1px solid var(--line);padding:20px;border-radius:2px}
.metric{background:var(--paper);border-top:3px solid var(--ink);padding:18px 19px;min-height:115px}
.metric .label{font-family:"DM Mono";font-size:10px;text-transform:uppercase;color:var(--muted);letter-spacing:.12em}
.metric .value{font-size:30px;font-weight:800;letter-spacing:-.05em;margin-top:12px}
.metric .note{font-size:11px;color:var(--muted);margin-top:4px}
.note-box{background:#e9e3d8;border-left:4px solid var(--red);padding:17px 19px;line-height:1.6}
.clean-row{display:grid;grid-template-columns:70px 1fr 150px;gap:15px;padding:15px 0;border-bottom:1px solid var(--line)}
.clean-num{font-family:"DM Mono";font-size:12px;color:var(--red)}
.clean-title{font-weight:800}
.clean-desc{font-size:13px;color:var(--muted);margin-top:4px}
.clean-action{font-family:"DM Mono";font-size:11px;text-align:right}
.footer{font-family:"DM Mono";font-size:10px;color:#97938b;text-align:center;margin-top:40px}
@media(max-width:900px){.block-container{padding:1rem}.hero{grid-template-columns:1fr}h1{font-size:52px!important}}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_dataset(n=12000, seed=2407):
    rng=np.random.default_rng(seed)
    zones=rng.choice(["Central","North","East","South","West"],n,p=[.24,.18,.20,.22,.16])
    order_type=rng.choice(["Standard","Priority","Scheduled"],n,p=[.67,.23,.10])
    weather=rng.choice(["Clear","Cloudy","Rain","Storm"],n,p=[.52,.25,.18,.05])
    hour=rng.integers(8,24,n)
    weekend=rng.integers(0,2,n)
    distance=np.clip(rng.gamma(2.2,2.0,n),.5,15)
    prep=np.clip(rng.normal(18,6,n),5,45)
    experience=np.clip(rng.normal(2.6,1.8,n),.1,8)
    traffic=np.clip(rng.normal(48,21,n),0,100)
    rating=np.clip(rng.normal(4.15,.45,n),2.5,5)
    value=np.clip(rng.lognormal(6.9,.48,n),180,3500)
    items=np.clip(rng.poisson(2.6,n)+1,1,10)
    peak=((hour>=12)&(hour<=14)|(hour>=19)&(hour<=21)).astype(int)
    zone_effect=pd.Series(zones).map({"Central":1.5,"North":3.0,"East":2.0,"South":3.8,"West":2.7}).to_numpy()
    weather_effect=pd.Series(weather).map({"Clear":0,"Cloudy":1.5,"Rain":5.5,"Storm":10}).to_numpy()
    type_effect=pd.Series(order_type).map({"Standard":0,"Priority":-2.5,"Scheduled":-1.0}).to_numpy()
    delivery=(7.5+distance*2.15+prep*.82+traffic*.105-experience*.9+(4.5-rating)*3
              +weather_effect+zone_effect+peak*4+weekend*1.5+type_effect+items*.65+rng.normal(0,3.7,n))
    df=pd.DataFrame({
        "order_id":[f"FD-{i:05d}" for i in range(1,n+1)],
        "zone":zones,"order_type":order_type,"distance_km":distance.round(2),
        "prep_time_min":prep.round(1),"rider_experience_years":experience.round(1),
        "traffic_index":traffic.round(1),"weather":weather,"restaurant_rating":rating.round(2),
        "order_value":value.round(0),"item_count":items,"hour":hour,"weekend":weekend,
        "peak_hour":peak,"delivery_time_min":np.maximum(delivery,8).round(1)
    })
    # Deliberate missing values for the cleaning stage.
    for col, frac in [("traffic_index",.02),("restaurant_rating",.015),("prep_time_min",.01)]:
        idx=rng.choice(df.index,size=int(n*frac),replace=False)
        df.loc[idx,col]=np.nan
    return df

@st.cache_resource
def train_models(clean):
    target="delivery_time_min"
    X=clean.drop(columns=[target,"order_id"])
    y=clean[target]
    cat=["zone","order_type","weather"]
    num=[c for c in X.columns if c not in cat]
    pre=ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
        ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])
    split=int(len(clean)*.8)
    Xtr,Xte=X.iloc[:split],X.iloc[split:]
    ytr,yte=y.iloc[:split],y.iloc[split:]
    specs={
        "Ridge Regression":Ridge(alpha=8.0),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=180,max_depth=3,learning_rate=.045,loss="huber",random_state=42)
    }
    fitted={}; preds=[]; pred_map={}
    for name,model in specs.items():
        pipe=Pipeline([("pre",pre),("model",model)])
        pipe.fit(Xtr,ytr); p=np.maximum(pipe.predict(Xte),0)
        fitted[name]=pipe; pred_map[name]=p
        preds.append({"Model":name,"MAE":mean_absolute_error(yte,p),"RMSE":mean_squared_error(yte,p)**.5,"R²":r2_score(yte,p)})
    return fitted,pd.DataFrame(preds).sort_values("R²",ascending=False).reset_index(drop=True),yte.reset_index(drop=True),pred_map

def metric(label,value,note):
    return f'<div class="metric"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>'

raw=generate_dataset()
missing_before=raw.isna().sum()
clean=raw.copy()
for c in clean.select_dtypes(include=np.number).columns:
    clean[c]=clean[c].fillna(clean[c].median())
duplicates=int(clean.duplicated().sum())
clean=clean.drop_duplicates().reset_index(drop=True)
models,metrics,ytest,predmap=train_models(clean)
best=metrics.iloc[0]

st.markdown('<div class="mast"><div class="logo">dispatch<span>/</span>intel</div><div class="issue">FIELD NOTE 07 · DELIVERY OPERATIONS</div></div>',unsafe_allow_html=True)

pages=["Briefing","The Dataset","Cleaning Desk","Patterns","Model Trials","Estimator"]
if "page" not in st.session_state: st.session_state.page=pages[0]
nav=st.columns(len(pages))
for i,p in enumerate(pages):
    with nav[i]:
        if st.button(p,key=f"n{i}",use_container_width=True): st.session_state.page=p

if st.session_state.page=="Briefing":
    st.markdown('<div class="hero"><div><div class="kicker">Food delivery · operations analytics</div><h1>Know why an order<br>takes longer.</h1><p>A synthetic but realistic delivery-operations dataset built to study how distance, kitchen preparation, traffic, weather and peak hours shape delivery time.</p></div><div class="stamp"><small>WORKING DATASET</small><div class="big">12,000</div><small>ORDERS · 16 VARIABLES · REPRODUCIBLE SEED</small></div></div>',unsafe_allow_html=True)
    cs=st.columns(4)
    cards=[("Records","12,000","synthetic orders"),("Missing cells",f"{int(raw.isna().sum().sum()):,}","before cleaning"),("Duplicate rows",str(duplicates),"checked after imputation"),("Best R²",f"{best['R²']:.3f}",best["Model"])]
    for c,x in zip(cs,cards):
        with c: st.markdown(metric(*x),unsafe_allow_html=True)
    st.markdown('<div class="section">Operations snapshot</div><div class="sub">The fastest way to understand the shape of the delivery problem.</div>',unsafe_allow_html=True)
    d=clean.groupby("hour",as_index=False)["delivery_time_min"].mean()
    fig=px.line(d,x="hour",y="delivery_time_min",markers=True,title="Average delivery time by hour")
    fig.update_traces(line_color="#e65d43",marker_color="#20211f")
    st.plotly_chart(fig.update_layout(height=420,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=55,b=10)),use_container_width=True)
    a,b=st.columns(2)
    with a:
        z=clean.groupby("zone",as_index=False)["delivery_time_min"].mean()
        fig=px.bar(z,x="zone",y="delivery_time_min",title="Average delivery time by service zone")
        fig.update_traces(marker_color="#3157a4"); st.plotly_chart(fig.update_layout(height=360,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"),use_container_width=True)
    with b:
        w=clean.groupby("weather",as_index=False)["delivery_time_min"].mean()
        fig=px.bar(w,x="weather",y="delivery_time_min",title="Weather effect")
        fig.update_traces(marker_color="#f4c95d"); st.plotly_chart(fig.update_layout(height=360,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"),use_container_width=True)

elif st.session_state.page=="The Dataset":
    st.markdown('<div class="kicker">01 · source design</div><div class="section">The Dataset</div><div class="sub">A fully reproducible synthetic food-delivery dataset. No external download is required.</div>',unsafe_allow_html=True)
    st.markdown('<div class="note-box"><b>What is simulated?</b> Each row represents one food-delivery order. The target is delivery time in minutes. The generator creates realistic relationships between distance, preparation time, traffic, weather, rider experience and peak periods.</div>',unsafe_allow_html=True)
    st.markdown("<div class='section'>Raw records</div>",unsafe_allow_html=True)
    st.dataframe(raw.head(120),use_container_width=True,height=480,hide_index=True)
    st.markdown("<div class='section'>Field guide</div>",unsafe_allow_html=True)
    guide=pd.DataFrame([
        ["distance_km","Numerical","Delivery distance"],
        ["prep_time_min","Numerical","Restaurant preparation time"],
        ["traffic_index","Numerical","0–100 traffic pressure"],
        ["weather","Categorical","Clear / Cloudy / Rain / Storm"],
        ["rider_experience_years","Numerical","Rider experience"],
        ["peak_hour","Binary","High-demand time indicator"],
        ["delivery_time_min","Target","Minutes to deliver"],
    ],columns=["Field","Type","Meaning"])
    st.dataframe(guide,use_container_width=True,hide_index=True)

elif st.session_state.page=="Cleaning Desk":
    st.markdown('<div class="kicker">02 · preprocessing</div><div class="section">Cleaning Desk</div><div class="sub">A visible audit trail from raw records to model-ready data.</div>',unsafe_allow_html=True)
    c=st.columns(4)
    cards=[("Rows before",f"{len(raw):,}","raw dataset"),("Missing cells",f"{int(missing_before.sum()):,}","three numeric fields"),("Rows after",f"{len(clean):,}","after duplicate check"),("Remaining nulls",f"{int(clean.isna().sum().sum()):,}","ready for analysis")]
    for col,x in zip(c,cards):
        with col: st.markdown(metric(*x),unsafe_allow_html=True)
    ledger=missing_before[missing_before>0].rename("Missing Before").reset_index().rename(columns={"index":"Column"})
    ledger["Method"]="Median imputation"
    ledger["Missing After"]=[int(clean[col].isna().sum()) for col in ledger["Column"]]
    st.dataframe(ledger,use_container_width=True,hide_index=True)
    st.markdown(f"""
    <div class="clean-row"><div class="clean-num">01</div><div><div class="clean-title">Detect missing values</div><div class="clean-desc">Count null cells for every field before any model training.</div></div><div class="clean-action">AUDIT</div></div>
    <div class="clean-row"><div class="clean-num">02</div><div><div class="clean-title">Median imputation</div><div class="clean-desc">Fill missing numerical values using the median of that field, preserving all orders.</div></div><div class="clean-action">3 COLUMNS</div></div>
    <div class="clean-row"><div class="clean-num">03</div><div><div class="clean-title">Duplicate check</div><div class="clean-desc">Check complete duplicate rows and remove them if present.</div></div><div class="clean-action">{duplicates} FOUND</div></div>
    <div class="clean-row"><div class="clean-num">04</div><div><div class="clean-title">Model preprocessing</div><div class="clean-desc">Numeric features are scaled; categorical features are imputed and one-hot encoded inside the training pipeline.</div></div><div class="clean-action">PIPELINE</div></div>
    """,unsafe_allow_html=True)
    st.markdown("<div class='section'>Clean sample</div>",unsafe_allow_html=True)
    st.dataframe(clean.head(120),use_container_width=True,height=440,hide_index=True)

elif st.session_state.page=="Patterns":
    st.markdown('<div class="kicker">03 · exploratory analysis</div><div class="section">Patterns</div><div class="sub">The variables that move with delivery time.</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        fig=px.scatter(clean.sample(2500,random_state=42),x="distance_km",y="delivery_time_min",color="weather",opacity=.6,title="Distance vs delivery time")
        st.plotly_chart(fig.update_layout(height=420,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"),use_container_width=True)
    with b:
        fig=px.box(clean,x="order_type",y="delivery_time_min",color="order_type",title="Delivery time by order type")
        st.plotly_chart(fig.update_layout(height=420,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False),use_container_width=True)
    corr=clean.select_dtypes(include=np.number).corr().round(2)
    fig=px.imshow(corr,text_auto=True,zmin=-1,zmax=1,color_continuous_scale="RdBu_r",title="Correlation matrix")
    st.plotly_chart(fig.update_layout(height=650,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"),use_container_width=True)

elif st.session_state.page=="Model Trials":
    st.markdown('<div class="kicker">04 · machine learning</div><div class="section">Model Trials</div><div class="sub">Two models. The same cleaned data. One 80/20 holdout.</div>',unsafe_allow_html=True)
    c=st.columns(3)
    for col,(lab,val,note) in zip(c,[("Winner",best["Model"],"highest R²"),("Best MAE",f"{metrics.MAE.min():.2f}","lower is better"),("Best RMSE",f"{metrics.RMSE.min():.2f}","lower is better")]):
        with col: st.markdown(metric(lab,val,note),unsafe_allow_html=True)
    view=metrics.copy()
    for x in ["MAE","RMSE"]: view[x]=view[x].round(2)
    view["R²"]=view["R²"].round(3)
    st.dataframe(view,use_container_width=True,hide_index=True)
    melt=metrics.melt("Model",value_vars=["MAE","RMSE","R²"],var_name="Metric",value_name="Score")
    fig=px.bar(melt,x="Metric",y="Score",color="Model",barmode="group",title="Model scorecard")
    st.plotly_chart(fig.update_layout(height=410,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"),use_container_width=True)
    st.markdown(f'<div class="note-box"><b>Selection rule:</b> {best["Model"]} is the winner because it has the strongest R² while also being competitive on absolute error. R² rewards explanatory power; MAE and RMSE measure prediction error, where lower is better.</div>',unsafe_allow_html=True)
    comp=pd.DataFrame({"Actual":ytest,"Ridge Regression":predmap["Ridge Regression"],"Gradient Boosting":predmap["Gradient Boosting"]}).tail(300).reset_index(drop=True)
    fig=go.Figure()
    for col in comp.columns: fig.add_trace(go.Scatter(y=comp[col],mode="lines",name=col))
    fig.update_layout(title="Actual vs predicted · final 300 test orders",height=430,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)

else:
    st.markdown('<div class="kicker">05 · scenario tool</div><div class="section">Estimator</div><div class="sub">Create a delivery scenario and let both trained models estimate the time.</div>',unsafe_allow_html=True)
    a,b=st.columns([.85,1.15])
    with a:
        distance=st.slider("Distance (km)",.5,15.,4.,.1)
        prep=st.slider("Restaurant prep time (min)",5.,45.,18.,.5)
        traffic=st.slider("Traffic index",0.,100.,50.,1.)
        experience=st.slider("Rider experience (years)",.1,8.,2.5,.1)
        rating=st.slider("Restaurant rating",2.5,5.,4.2,.1)
        value=st.slider("Order value",180,3500,900,20)
        items=st.slider("Items",1,10,3)
        hour=st.slider("Hour",8,23,19)
        zone=st.selectbox("Zone",["Central","North","East","South","West"])
        order_type=st.selectbox("Order type",["Standard","Priority","Scheduled"])
        weather=st.selectbox("Weather",["Clear","Cloudy","Rain","Storm"])
        weekend=st.selectbox("Weekend",["No","Yes"])
        run=st.button("Run estimate",type="primary",use_container_width=True)
    with b:
        if run:
            peak=int((12<=hour<=14) or (19<=hour<=21))
            row=pd.DataFrame([{"zone":zone,"order_type":order_type,"distance_km":distance,"prep_time_min":prep,"rider_experience_years":experience,"traffic_index":traffic,"weather":weather,"restaurant_rating":rating,"order_value":value,"item_count":items,"hour":hour,"weekend":int(weekend=="Yes"),"peak_hour":peak}])
            p1=max(0,float(models["Ridge Regression"].predict(row)[0])); p2=max(0,float(models["Gradient Boosting"].predict(row)[0]))
            avg=(p1+p2)/2
            st.markdown(metric("Consensus estimate",f"{avg:.0f} min","average of both models"),unsafe_allow_html=True)
            c1,c2=st.columns(2)
            with c1: st.markdown(metric("Ridge",f"{p1:.0f} min","model estimate"),unsafe_allow_html=True)
            with c2: st.markdown(metric("Gradient Boosting",f"{p2:.0f} min","model estimate"),unsafe_allow_html=True)
            fig=go.Figure(go.Indicator(mode="gauge+number",value=avg,title={"text":"Expected delivery time"},gauge={"axis":{"range":[0,100]},"bar":{"color":"#e65d43"},"steps":[{"range":[0,25],"color":"#e7f0e6"},{"range":[25,45],"color":"#f4e9c8"},{"range":[45,100],"color":"#f4d2ca"}]}))
            fig.update_layout(height=350,paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.markdown('<div class="card"><div class="kicker">ESTIMATION STUDIO</div><div style="font-size:30px;font-weight:800;letter-spacing:-.05em">Build a scenario.</div><p style="color:#77746e;line-height:1.7">Change the operational conditions on the left. The two models will turn them into an estimated delivery time.</p></div>',unsafe_allow_html=True)

st.markdown('<div class="footer">dispatch/intel · synthetic food-delivery operations study · 12,000 records</div>',unsafe_allow_html=True)
