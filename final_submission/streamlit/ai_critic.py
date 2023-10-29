# -------------------------------------------------------------------------------- #
# Import Libraries
# -------------------------------------------------------------------------------- #
import streamlit as st

import pandas as pd
import datetime

# Custome libraries
import helper


st.set_page_config(page_title="Main Dashboard", page_icon="📊")
st.title('AI Critic Dashboard')
st.sidebar.success("Select a demo above.")

@st.cache_data
def load_data(excel_filepath, nrows=None):

    if nrows== None:
        data = pd.read_excel(excel_filepath)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)
    else:
        data = pd.read_excel(excel_filepath, nrows=nrows)
        lowercase = lambda x: str(x).lower()
        data.rename(lowercase, axis='columns', inplace=True)

    return data


data_load_state = st.text('Loading data...')
data = helper.load_data('../inputs/preprocessed_230604.xlsx')
data_load_state.text(f"Done! (using cached data)")

# -------------------------------------------------------------------------------- #
# Build KPIs
# -------------------------------------------------------------------------------- #

# Total post Flagged
ttl_post = data.non_compliant.value_counts()[1]
# Total Agent Flagged
ttl_agent = 11
# Average No. of Posts
avg_post = ttl_post / ttl_agent


kpi1, kpi2, kpi3 = st.columns(3)

kpi1.metric(
    label="Total Post Flagged",
    value=round(ttl_post),
    delta=round(ttl_post) - 10,
    delta_color="inverse",
)

kpi2.metric(
    label="Total Agent Flagged",
    value=int(ttl_agent),
    delta=-10 + ttl_agent,
    delta_color="inverse",
)

kpi3.metric(
    label="Average No. of Posts",
    value=f"$ {round(avg_post,2)} ",
    delta=round(avg_post) - 10,
    delta_color="inverse",
)

# -------------------------------------------------------------------------------- #
# Build Charts
# -------------------------------------------------------------------------------- #

st.subheader("Annual Post Count")
# Preprocess Data
data['month_literal'] = data['month'].apply(lambda monthinteger: datetime.date(1900, monthinteger, 1).strftime('%B'))
post_cnt = pd.DataFrame(data.groupby(['month_literal']).sum()['non_compliant'].sort_index())
post_cnt['total'] = data.groupby(['month_literal']).count()['non_compliant'].values
post_cnt['compliant'] = post_cnt['total'] - post_cnt['non_compliant']
# Plot chart
st.plotly_chart(helper.plot_barchart(post_cnt), use_container_width=True)

st.subheader("Post Count by Channel")
# Preprocess Data
channel_cnt = pd.DataFrame(data.groupby(['channel']).sum()['non_compliant'].sort_index())
channel_cnt['total'] = data.groupby(['channel']).count()['non_compliant'].values
channel_cnt['compliant'] = channel_cnt['total'] - channel_cnt['non_compliant']
# Plot chart
st.plotly_chart(helper.plot_barchart(channel_cnt), use_container_width=True)

st.subheader("Post Count by Breach")
# Preprocess Data
breach_cnt = pd.DataFrame(data.breach.value_counts()).sort_values(by='breach', ascending=True)
# Plot chart
st.plotly_chart(helper.plot_barh(breach_cnt), use_container_width=True)
