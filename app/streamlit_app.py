"""
Reddit Tech Sentiment Dashboard — Final Mix
=============================================
Neon chartreuse accent (#C8FF00) · Glassmorphism cards · Gradient mesh bg
Pill-style tabs · DM Sans + JetBrains Mono · Violin plots · Topic lollipops

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SentimentPulse — Reddit NLP Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design Tokens ────────────────────────────────────────────────────────────

ACCENT     = "#C8FF00"
ACCENT_DIM = "rgba(200,255,0,0.08)"
ACCENT_MED = "rgba(200,255,0,0.18)"
BG         = "#080c14"
BG_CARD    = "rgba(14,18,30,0.65)"
BG_SIDEBAR = "#0a0e18"
BORDER     = "rgba(255,255,255,0.06)"
TXT        = "#e8eaed"
TXT2       = "#64748b"
TXT3       = "#2d3548"
POS        = "#34d399"
NEG        = "#f87171"
NEU        = "#94a3b8"

PLOTLY = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TXT, family="DM Sans, Segoe UI, sans-serif", size=12),
    margin=dict(t=32, b=28, l=44, r=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)",
               linecolor=BORDER),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.06)",
               linecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#0f1520", font_size=12, font_color=TXT, bordercolor=BORDER),
)

SUB_COLORS = [ACCENT, "#a78bfa", "#34d399", "#f97316", "#f472b6",
              "#facc15", "#38bdf8", "#818cf8", "#4ade80", "#fb923c"]


# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Global ── */
    .stApp {{
        background: {BG};
        background-image:
            radial-gradient(ellipse 70% 50% at 8% 0%, rgba(160,200,0,0.04) 0%, transparent 55%),
            radial-gradient(ellipse 60% 45% at 92% 100%, rgba(200,255,0,0.025) 0%, transparent 50%);
        font-family: 'DM Sans', sans-serif;
    }}
    header[data-testid="stHeader"] {{
        background: rgba(8,12,20,0.8);
        backdrop-filter: blur(14px);
        border-bottom: 1px solid {BORDER};
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background: {BG_SIDEBAR};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] h2 {{
        color: {TXT2};
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-top: 1.1rem;
    }}
    section[data-testid="stSidebar"] .stMarkdown p {{
        color: {TXT2};
        font-size: 0.85rem;
    }}

    /* ── Metric Cards — glass ── */
    div[data-testid="stMetric"] {{
        background: {BG_CARD};
        backdrop-filter: blur(16px);
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 20px 14px;
        transition: border-color 0.25s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: {ACCENT_MED};
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {TXT2} !important;
        font-size: 0.7rem !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}
    div[data-testid="stMetricValue"] {{
        color: {TXT} !important;
        font-size: 1.6rem !important;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    div[data-testid="stMetricDelta"] {{
        font-size: 0.75rem;
    }}

    /* ── Tabs — pill style ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 3px;
        background: {BG_CARD};
        backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {TXT2};
        font-weight: 500;
        font-size: 0.82rem;
        border-radius: 9px;
        padding: 8px 18px;
        background: transparent;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background: {ACCENT};
        color: #000 !important;
        font-weight: 700;
    }}
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {{
        display: none;
    }}

    /* ── Section labels ── */
    .sec {{
        color: {TXT};
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .sec::before {{
        content: '';
        width: 3px;
        height: 16px;
        background: {ACCENT};
        border-radius: 2px;
    }}

    /* ── Dataframe ── */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ── Inputs ── */
    div[data-baseweb="select"] > div {{
        background: rgba(14,18,30,0.5);
        border-color: {BORDER};
    }}
    div[data-baseweb="popover"] > div {{
        background: #0f1520;
    }}

    /* ── Radio ── */
    .stRadio > div {{ gap: 0.5rem; }}
    .stRadio [data-baseweb="radio"] label {{
        color: {TXT2} !important;
        font-size: 0.82rem;
    }}

    /* ── Divider ── */
    hr {{ border-color: {BORDER}; margin: 1.2rem 0; }}

    /* ── Footer ── */
    .foot {{
        text-align: center;
        color: {TXT3};
        font-size: 0.75rem;
        padding: 2.5rem 0 1rem;
        border-top: 1px solid {BORDER};
        margin-top: 3rem;
    }}
    .foot a {{ color: {ACCENT}; text-decoration: none; }}
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading data...")
def load_data():
    """Load real analyzed data from posts_final.parquet."""
    data_path = PROJECT_ROOT / "data" / "processed" / "posts_final.parquet"
    if data_path.exists():
        df = pd.read_parquet(data_path)
        df["created_utc"] = pd.to_datetime(df["created_utc"], utc=True)
        return df
    else:
        st.error(f"Data file not found: {data_path}\nRun notebooks 01–04 first.")
        st.stop()


df = load_data()
label_col = "vader_label" if "vader_label" in df.columns else "sentiment_label"


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
        <div style="display:flex; align-items:center; gap:11px; margin-bottom:26px;
                    padding-bottom:18px; border-bottom:1px solid {BORDER};">
            <div style="width:36px; height:36px; border-radius:10px; background:{ACCENT};
                        display:flex; align-items:center; justify-content:center;
                        font-size:17px; font-weight:800; color:#000;">⚡</div>
            <div>
                <div style="font-size:1.08rem; font-weight:700; color:{TXT};
                            letter-spacing:-0.02em;">SentimentPulse</div>
                <div style="font-size:0.65rem; color:{TXT2}; letter-spacing:0.06em;
                            text-transform:uppercase;">Reddit NLP Dashboard</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## Data Filters")

    subreddits = sorted(df["subreddit"].unique().tolist())
    sel_subs = st.multiselect("Subreddits", subreddits, default=subreddits)

    min_d = df["created_utc"].min().date()
    max_d = df["created_utc"].max().date()
    date_range = st.date_input("Period", value=(min_d, max_d),
                               min_value=min_d, max_value=max_d)

    sel_sent = st.multiselect("Sentiment", ["positive", "neutral", "negative"],
                              default=["positive", "neutral", "negative"])

    min_score = st.slider("Minimum Score", 0, 500, 0, step=5)

    st.markdown("---")
    st.markdown(f"""
        <div style="background:{BG_CARD}; backdrop-filter:blur(12px);
                    border:1px solid {BORDER}; border-radius:12px;
                    padding:14px 16px; font-size:0.82rem; line-height:1.65;">
            <span style="color:{TXT}; font-weight:600;">Shril Patel</span><br>
            <span style="color:{TXT2};">NLP · Sentiment Analysis · Topic Modeling</span><br>
            <a href="https://github.com/ZeroZulu" style="color:{ACCENT}; text-decoration:none;">
                GitHub ↗</a>
            <span style="color:{TXT3};"> · </span>
            <a href="https://linkedin.com/in/shril-patel-020504284" style="color:{ACCENT}; text-decoration:none;">
                LinkedIn ↗</a>
        </div>
    """, unsafe_allow_html=True)


# ─── Filtering ────────────────────────────────────────────────────────────────

df_f = df[df["subreddit"].isin(sel_subs)].copy()
if len(date_range) == 2:
    df_f = df_f[(df_f["created_utc"].dt.date >= date_range[0]) &
                (df_f["created_utc"].dt.date <= date_range[1])]
df_f = df_f[df_f[label_col].isin(sel_sent)]
df_f = df_f[df_f["score"] >= min_score]


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(f"""
    <div style="margin-bottom:4px;">
        <span style="display:inline-block; width:8px; height:8px; border-radius:50%;
                     background:{ACCENT}; margin-right:10px; vertical-align:middle;"></span>
        <span style="font-size:1.45rem; font-weight:800; color:{TXT};
                     letter-spacing:-0.03em;">SentimentPulse Dashboard</span>
    </div>
    <div style="font-size:0.82rem; color:{TXT2}; margin-bottom:18px; padding-left:18px;">
        {len(df_f):,} posts · {df_f['subreddit'].nunique()} subreddits ·
        {df_f['created_utc'].min().strftime('%b %Y')} – {df_f['created_utc'].max().strftime('%b %Y')}
    </div>
""", unsafe_allow_html=True)


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Sentiment Trends", "Topics", "Subreddits", "Data Explorer"
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    avg_s = df_f["vader_compound"].mean()
    pct_pos = (df_f[label_col] == "positive").mean() * 100
    pct_neg = (df_f[label_col] == "negative").mean() * 100

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Posts", f"{len(df_f):,}")
    m2.metric("Avg Sentiment", f"{avg_s:+.3f}",
              delta="Positive" if avg_s > 0 else "Negative")
    m3.metric("% Positive", f"{pct_pos:.1f}%")
    m4.metric("Median Score", f"{df_f['score'].median():,.0f}")
    m5.metric("Avg Comments", f"{df_f['num_comments'].mean():,.1f}")

    st.markdown("")

    # ── Charts row ──
    cl, cr = st.columns([1.1, 0.9])

    with cl:
        st.markdown('<div class="sec">Sentiment Breakdown</div>', unsafe_allow_html=True)
        sc = df_f[label_col].value_counts()

        fig = go.Figure()
        fig.add_trace(go.Pie(
            values=sc.values, labels=sc.index, hole=0.6,
            marker=dict(
                colors=[POS if n == "positive" else NEG if n == "negative" else NEU
                        for n in sc.index],
                line=dict(color=BG, width=3),
            ),
            textinfo="label+percent",
            textfont=dict(size=11, color=TXT),
            hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
        ))
        fig.add_annotation(
            text=(f"<b style='font-size:20px'>{len(df_f):,}</b><br>"
                  f"<span style='font-size:10px;color:{TXT2}'>posts</span>"),
            showarrow=False, font=dict(size=16, color=TXT, family="JetBrains Mono"),
        )
        fig.update_layout(**PLOTLY, height=370, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.markdown('<div class="sec">Subreddit Volume</div>', unsafe_allow_html=True)
        sub_c = df_f["subreddit"].value_counts().sort_values(ascending=True)
        fig = go.Figure(go.Bar(
            y=sub_c.index, x=sub_c.values, orientation="h",
            marker=dict(
                color=[SUB_COLORS[i % len(SUB_COLORS)] for i in range(len(sub_c))],
                cornerradius=5,
            ),
            text=[f"{v:,}" for v in sub_c.values],
            textposition="inside",
            textfont=dict(size=10, family="JetBrains Mono", color="#000"),
            hovertemplate="%{y}: %{x:,}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY, height=370, showlegend=False)
        fig.update_layout(yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                          xaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)

    # ── Score distribution ──
    st.markdown('<div class="sec">Sentiment Score Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df_f, x="vader_compound", nbins=70,
                       color_discrete_sequence=[ACCENT],
                       labels={"vader_compound": "VADER Compound Score"})
    fig.update_layout(**PLOTLY, height=240, bargap=0.02)
    fig.update_traces(marker_line_width=0, opacity=0.8)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: SENTIMENT TRENDS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="sec">Sentiment Over Time</div>', unsafe_allow_html=True)

    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
    freq_label = st.radio("Granularity", list(freq_map.keys()), index=1, horizontal=True)
    freq = freq_map[freq_label]

    df_ts = df_f.copy()
    df_ts["period"] = df_ts["created_utc"].dt.tz_localize(None).dt.to_period(
        freq.replace("ME", "M")).astype(str)

    trend = (df_ts.groupby(["period", "subreddit"])
             .agg(avg_sentiment=("vader_compound", "mean"),
                  post_count=("id", "count"))
             .reset_index())

    fig = px.line(trend, x="period", y="avg_sentiment", color="subreddit",
                  color_discrete_sequence=SUB_COLORS,
                  markers=True,
                  labels={"period": "", "avg_sentiment": "Avg Sentiment", "subreddit": ""})
    fig.add_hline(y=0, line_dash="dot", line_color=TXT3, line_width=1,
                  annotation_text="Neutral", annotation_font_size=10,
                  annotation_font_color=TXT2)
    fig.update_layout(**PLOTLY, height=420)
    fig.update_traces(line_width=2.2, marker_size=5)
    st.plotly_chart(fig, use_container_width=True)

    # ── Volume stacked area ──
    st.markdown('<div class="sec">Volume by Subreddit</div>', unsafe_allow_html=True)
    vol = df_ts.groupby(["period", "subreddit"]).size().reset_index(name="count")
    fig = px.area(vol, x="period", y="count", color="subreddit",
                  color_discrete_sequence=SUB_COLORS,
                  labels={"period": "", "count": "Posts", "subreddit": ""})
    fig.update_layout(**PLOTLY, height=280)
    fig.update_traces(line_width=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Sentiment heatmap: subreddit × period ──
    if len(sel_subs) > 1:
        st.markdown('<div class="sec">Sentiment Heatmap · Subreddit × Period</div>',
                    unsafe_allow_html=True)
        hm_data = (df_ts.groupby(["subreddit", "period"])["vader_compound"]
                   .mean().reset_index()
                   .pivot(index="subreddit", columns="period", values="vader_compound")
                   .fillna(0))
        fig = px.imshow(hm_data.values,
                        x=hm_data.columns.tolist(),
                        y=hm_data.index.tolist(),
                        color_continuous_scale=[[0, NEG], [0.5, "#1a1d2e"], [1, POS]],
                        labels={"color": "Sentiment"}, aspect="auto")
        fig.update_layout(**PLOTLY, height=max(200, len(hm_data) * 38),
                          coloraxis_showscale=True)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: TOPICS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    if "topic_name" in df_f.columns:
        # ── Topic volume ──
        st.markdown('<div class="sec">Discovered Topics</div>', unsafe_allow_html=True)
        tc = df_f["topic_name"].value_counts()
        fig = go.Figure(go.Bar(
            x=tc.values, y=tc.index, orientation="h",
            marker=dict(
                color=[SUB_COLORS[i % len(SUB_COLORS)] for i in range(len(tc))],
                cornerradius=4,
            ),
            text=[f"{v:,}" for v in tc.values],
            textposition="inside",
            textfont=dict(size=10, family="JetBrains Mono", color="#000"),
        ))
        fig.update_layout(**PLOTLY, height=max(420, len(tc) * 34))
        fig.update_layout(yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                          xaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True)

        # ── Topic sentiment — lollipop ──
        st.markdown('<div class="sec">Topic Sentiment</div>', unsafe_allow_html=True)
        ts = df_f.groupby("topic_name")["vader_compound"].mean().sort_values()
        overall_avg = df_f["vader_compound"].mean()

        fig = go.Figure()
        for i, (name, val) in enumerate(ts.items()):
            c = POS if val > overall_avg else NEG if val < overall_avg * 0.6 else NEU
            # Stem line
            fig.add_trace(go.Scatter(
                x=[overall_avg, val], y=[name, name], mode="lines",
                line=dict(color=c, width=1.5), showlegend=False,
                hoverinfo="skip",
            ))
            # Dot
            fig.add_trace(go.Scatter(
                x=[val], y=[name], mode="markers",
                marker=dict(size=10, color=c, line=dict(width=2, color=BG)),
                hovertemplate=f"{name}<br>Mean: {val:+.3f}<extra></extra>",
                showlegend=False,
            ))
        fig.add_vline(x=overall_avg, line_dash="dash", line_color=ACCENT, line_width=1,
                      annotation_text=f"overall avg {overall_avg:+.3f}",
                      annotation_font_size=10, annotation_font_color=TXT2)
        fig.update_layout(**PLOTLY, height=max(400, len(ts) * 32))
        fig.update_layout(xaxis=dict(title="Avg VADER Compound"),
                          yaxis=dict(gridcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, use_container_width=True)

        # ── Top 5 topics over time ──
        st.markdown('<div class="sec">Top 5 Topics Over Time</div>', unsafe_allow_html=True)
        top5 = df_f["topic_name"].value_counts().head(5).index.tolist()
        df_top = df_f[df_f["topic_name"].isin(top5)].copy()
        df_top["month"] = df_top["created_utc"].dt.tz_localize(None).dt.to_period("M").astype(str)
        tvol = df_top.groupby(["month", "topic_name"]).size().reset_index(name="count")
        fig = px.line(tvol, x="month", y="count", color="topic_name",
                      color_discrete_sequence=SUB_COLORS, markers=True,
                      labels={"month": "", "count": "Posts", "topic_name": "Topic"})
        fig.update_layout(**PLOTLY, height=340)
        fig.update_traces(line_width=2, marker_size=4)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Topic data not available. Run notebook 04 first.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: SUBREDDITS
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="sec">Sentiment Distribution</div>', unsafe_allow_html=True)
        fig = px.violin(df_f, x="subreddit", y="vader_compound", color="subreddit",
                        color_discrete_sequence=SUB_COLORS, box=True, points=False,
                        labels={"vader_compound": "Sentiment", "subreddit": ""})
        fig.update_layout(**PLOTLY, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="sec">Engagement vs Sentiment</div>', unsafe_allow_html=True)
        n = min(2000, len(df_f))
        samp = df_f.sample(n, random_state=42)
        fig = px.scatter(samp, x="vader_compound",
                         y=np.log1p(samp["score"]),
                         color="subreddit", opacity=0.4,
                         color_discrete_sequence=SUB_COLORS,
                         labels={"vader_compound": "Sentiment", "y": "log(Score+1)"})
        fig.update_layout(**PLOTLY, height=400)
        fig.update_traces(marker_size=4)
        st.plotly_chart(fig, use_container_width=True)

    # ── Stats table ──
    st.markdown('<div class="sec">Subreddit Statistics</div>', unsafe_allow_html=True)
    stats = (df_f.groupby("subreddit").agg(
        posts=("id", "count"),
        avg_sentiment=("vader_compound", "mean"),
        median_score=("score", "median"),
        avg_comments=("num_comments", "mean"),
        pct_positive=(label_col, lambda x: (x == "positive").mean()),
    ).round(3).sort_values("posts", ascending=False).reset_index())

    st.dataframe(
        stats.style.format({
            "avg_sentiment": "{:+.3f}",
            "median_score": "{:,.0f}",
            "avg_comments": "{:,.1f}",
            "pct_positive": "{:.1%}",
        }).background_gradient(subset=["avg_sentiment"], cmap="RdYlGn", vmin=-0.2, vmax=0.4),
        use_container_width=True, hide_index=True,
    )

    # ── Activity heatmap ──
    if "hour_of_day" in df_f.columns and "day_of_week" in df_f.columns:
        st.markdown('<div class="sec">Activity Heatmap · Hour × Day</div>',
                    unsafe_allow_html=True)
        hm = (df_f.groupby(["day_of_week", "hour_of_day"]).size()
              .reset_index(name="count")
              .pivot(index="day_of_week", columns="hour_of_day", values="count").fillna(0))
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig = px.imshow(hm.values, x=list(range(24)), y=days[:len(hm)],
                        color_continuous_scale=[[0, BG], [0.4, "#2a3a00"], [1, ACCENT]],
                        labels={"x": "Hour", "y": "", "color": "Posts"})
        fig.update_layout(**PLOTLY, height=250, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown('<div class="sec">Data Explorer</div>', unsafe_allow_html=True)

    ec1, ec2, ec3 = st.columns([3, 1, 1])
    with ec1:
        q = st.text_input("🔍 Search", placeholder="Filter by title...",
                          label_visibility="collapsed")
    with ec2:
        sort_col = st.selectbox("Sort by", ["score", "num_comments", "vader_compound",
                                            "created_utc"], label_visibility="collapsed")
    with ec3:
        asc = st.checkbox("Ascending", False)

    df_ex = df_f.copy()
    if q:
        df_ex = df_ex[df_ex["title"].str.contains(q, case=False, na=False)]
    df_ex = df_ex.sort_values(sort_col, ascending=asc)

    cols = ["subreddit", "title", "score", "num_comments", "vader_compound", label_col, "created_utc"]
    cols = [c for c in cols if c in df_ex.columns]
    if "topic_name" in df_ex.columns:
        cols.insert(-1, "topic_name")

    st.dataframe(
        df_ex[cols].head(400).style.format({
            "vader_compound": "{:+.3f}",
            "score": "{:,}",
            "num_comments": "{:,}",
        }).map(
            lambda x: (f"background:rgba(52,211,153,0.12); color:{POS}"
                       if x == "positive"
                       else f"background:rgba(248,113,113,0.12); color:{NEG}"
                       if x == "negative" else ""),
            subset=[label_col],
        ),
        use_container_width=True, hide_index=True, height=560,
    )
    st.caption(f"Showing {min(400, len(df_ex)):,} of {len(df_ex):,} filtered posts")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown(f"""
    <div class="foot">
        SentimentPulse · {len(df):,} posts analyzed ·
        VADER + DistilBERT + LDA · Streamlit & Plotly ·
        <a href="https://github.com/ZeroZulu">Shril Patel</a>
    </div>
""", unsafe_allow_html=True)
