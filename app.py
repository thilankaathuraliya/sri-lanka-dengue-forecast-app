"""Sri Lankan Dengue forecasting MSc demonstration.
Historical one-step-ahead Jan 2021 only. Not a public-health warning system.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'dengue_catboost_improved.cbm'
META_PATH = ROOT / 'dengue_model_metadata.json'
DATA_PATH = ROOT / 'data' / 'dengue_history.xlsx'
EVAL_PATH = ROOT / 'dengue_city_month_predictions.csv'

st.set_page_config(page_title='Sri Lanka Dengue Forecast', page_icon='🦟', layout='wide')
st.title('Sri Lankan Dengue Prediction Dashboard')
st.caption('MSc Machine Learning assignment | CatBoost + explainability | Historical research demonstration')
st.warning('Experimental historical forecasting only. Not a real-time dengue alert, medical advice, or a validated public-health forecasting service.')

@st.cache_data
def read_history():
    raw = pd.read_excel(DATA_PATH, sheet_name='All Data', usecols='A:C')
    raw.columns = ['Date','City','Value']
    raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce').dt.to_period('M').dt.to_timestamp()
    raw['City'] = raw['City'].astype('string').str.strip().astype(str)
    raw['Value'] = pd.to_numeric(raw['Value'], errors='coerce')
    if raw[['Date','City','Value']].isna().any().any() or raw.duplicated(['City','Date']).any() or (raw.Value<0).any():
        raise ValueError('History contains missing, duplicate, or negative observations.')
    return raw.sort_values(['City','Date']).reset_index(drop=True)

@st.cache_resource
def load_model():
    m = CatBoostRegressor()
    m.load_model(str(MODEL_PATH))
    return m

def make_row(city, history, target_date):
    g = history.loc[history.City.eq(city)].sort_values('Date')
    if len(g)<12:
        raise ValueError('At least 12 consecutive months of history required.')
    if g.Date.max() != target_date - pd.DateOffset(months=1):
        raise ValueError('Prediction must be exactly one month after the last observed month.')
    recent = g.tail(12)
    expected = pd.date_range(recent.Date.iloc[0], periods=12, freq='MS')
    if not recent.Date.reset_index(drop=True).equals(pd.Series(expected)):
        raise ValueError('History contains gaps in the latest 12 months.')
    v = g.Value.to_numpy(dtype=float)
    month = target_date.month
    row = {'City':city, 'month_sin':np.sin(2*np.pi*month/12), 'month_cos':np.cos(2*np.pi*month/12)}
    for lag in [1,2,3,6,12]: row[f'lag_{lag}']=float(v[-lag])
    for w in [3,6,12]:
        row[f'rolling_mean_{w}']=float(v[-w:].mean())
        row[f'rolling_std_{w}']=float(v[-w:].std(ddof=0))
    row['recent_change']=row['lag_1']-row['lag_2']
    row['ratio_recent_to_year']=row['lag_1']/(row['lag_12']+1)
    return row

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error('The exported Colab model files are missing. Place dengue_catboost_improved.cbm and dengue_model_metadata.json in the app folder, then restart.')
    st.code('dengue_catboost_improved.cbm\ndengue_model_metadata.json')
    st.stop()
try:
    history=read_history()
    metadata=json.loads(META_PATH.read_text(encoding='utf-8'))
    model=load_model()
    features=metadata['features']
    categorical=metadata.get('categorical_features',['City'])
    if 'City' not in features or any(c not in features for c in categorical):
        raise ValueError('Invalid feature metadata.')
    last=history.Date.max()
    forecast_month=last+pd.DateOffset(months=1)
    if forecast_month.strftime('%Y-%m') != metadata.get('forecast_month'):
        raise ValueError('Dataset end date does not match the exported model forecast metadata.')
except Exception as exc:
    st.error(f'Could not initialize the application: {exc}')
    st.stop()

cities=sorted(history.City.unique())
city=st.sidebar.selectbox('Select location',cities)
st.sidebar.markdown('**Forecast mode:** One month ahead from last recorded data')
st.sidebar.caption(f'Last observed month: {last:%B %Y}')
st.sidebar.caption(f'Forecast month: {forecast_month:%B %Y}')
st.sidebar.caption(f'Model trained through: {metadata.get("train_end", "not recorded")}')
st.sidebar.caption('The model uses historical lag features; it does not predict multiple future months recursively.')

row=make_row(city,history,forecast_month)
X=pd.DataFrame([row])[features]
raw=float(model.predict(X)[0])
pred=max(0.0,float(np.expm1(raw)) if metadata.get('log_target',False) else raw)
baseline=row['lag_1']
city_hist=history[history.City.eq(city)].sort_values('Date')

st.subheader(f'{city} — {forecast_month:%B %Y} historical forecast')
a,b,c=st.columns(3)
a.metric('CatBoost predicted cases',f'{pred:,.0f}')
b.metric('Previous-month baseline',f'{baseline:,.0f}')
c.metric('Last observed month',f'{last:%b %Y}')
st.caption('Case predictions are rounded for display. A model trained only through 2018 is being applied to features derived from observations through December 2020.')

st.subheader('Recorded dengue cases')
fig=px.line(city_hist,x='Date',y='Value',markers=False,labels={'Value':'Recorded cases','Date':'Month'},title=f'Historical reported dengue cases — {city}')
fig.add_trace(go.Scatter(x=[forecast_month],y=[pred],mode='markers',marker={'size':12,'color':'orange'},name='Jan 2021 CatBoost forecast'))
fig.add_trace(go.Scatter(x=[forecast_month],y=[baseline],mode='markers',marker={'size':10,'color':'green'},name='Previous-month baseline'))
st.plotly_chart(fig,use_container_width=True)

st.subheader('How this prediction was produced')
st.write('The model uses the last observed monthly counts and seasonal signals. All rolling statistics use months before the forecast date.')
with st.expander('View input features'):
    st.dataframe(X.T.rename(columns={0:'Value'}),use_container_width=True)

st.subheader('Local SHAP-style feature contributions (CatBoost SHAP values)')
try:
    sv=model.get_feature_importance(Pool(X,cat_features=categorical),type='ShapValues')[0]
    values=sv[:-1]
    labels=features
    explain=pd.DataFrame({'Feature':labels,'Contribution':values}).sort_values('Contribution')
    chart=px.bar(explain,x='Contribution',y='Feature',orientation='h',color='Contribution',color_continuous_scale='RdBu_r',color_continuous_midpoint=0)
    st.plotly_chart(chart,use_container_width=True)
    if metadata.get('log_target',False):
        st.caption('SHAP contributions are on the log(1 + cases) model-output scale, before inverse transformation.')
    else:
        st.caption('Positive SHAP contributions increase the raw predicted case count; negative contributions decrease it. The displayed prediction is clipped at zero if necessary.')
    st.caption('These explanations describe model behavior, not causal effects.')
except Exception as exc:
    st.info(f'Per-prediction explanations could not be calculated: {exc}')

st.subheader('Historical evaluation (2019–2020)')
st.caption('Exploratory evaluation only: these years were previously inspected during development and are not an independent blind test.')
metrics=pd.DataFrame([
    {'Method':'Revised CatBoost','MAE':105.995,'RMSE':242.278,'R²':0.704},
    {'Method':'Previous-month baseline','MAE':97.037,'RMSE':237.311,'R²':0.716},
    {'Method':'Same-month-last-year baseline','MAE':259.912,'RMSE':557.244,'R²':-0.566},
])
st.dataframe(metrics,hide_index=True,use_container_width=True)
st.info('The previous-month baseline outperformed CatBoost in this historical evaluation. The app displays both methods transparently.')
if EVAL_PATH.exists():
    eval_df=pd.read_csv(EVAL_PATH,parse_dates=['Date'])
    if {'Date','City','Value','CatBoost','Previous month'}.issubset(eval_df.columns):
        series=eval_df.loc[eval_df.City.eq(city)].sort_values('Date')
        fig2=go.Figure()
        for name in ['Value','CatBoost','Previous month']:
            fig2.add_trace(go.Scatter(x=series.Date,y=series[name],mode='lines',name=name))
        fig2.update_layout(title=f'Historical one-step-ahead predictions — {city}',xaxis_title='Month',yaxis_title='Cases')
        st.plotly_chart(fig2,use_container_width=True)

st.subheader('Limitations and responsible use')
st.write('The data ends in December 2020; this dashboard cannot provide a current forecast. The dataset lacks weather, vector surveillance, and intervention variables. Abrupt outbreaks are difficult to forecast. Location labels reflect the original dataset and should be verified against its source. Predictions must not guide medical or public-health decisions.')
st.download_button('Download location history (CSV)',data=city_hist.to_csv(index=False),file_name='dengue_history_'+city.replace(' ','_')+'.csv',mime='text/csv')
