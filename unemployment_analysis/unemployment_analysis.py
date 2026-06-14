"""
Unemployment Analysis with Python
===================================
CodeAlpha Data Science Internship — Task 2
Author: Gemechu Ejeta Atomsa

Objective:
    Analyze unemployment rate data across Indian states (2019–2020),
    investigate the impact of Covid-19, identify regional and seasonal
    patterns, and present policy-relevant insights.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── 1. Load & Clean Data ─────────────────────────────────────────────────────
df = pd.read_csv('data/Unemployment_in_India.csv')
df.columns = df.columns.str.strip()
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'].str.strip(), format='%d-%m-%Y')
df = df.sort_values('Date').reset_index(drop=True)

# Derived columns
df['Month']  = df['Date'].dt.to_period('M')
df['Period'] = df['Date'].apply(
    lambda x: 'Pre-Covid' if x < pd.Timestamp('2020-03-01') else 'Covid-19'
)

unemp_col  = 'Estimated Unemployment Rate (%)'
labour_col = 'Estimated Labour Participation Rate (%)'
employ_col = 'Estimated Employed'

print("=" * 60)
print("UNEMPLOYMENT ANALYSIS IN INDIA (2019–2020)")
print("=" * 60)
print(f"Dataset shape : {df.shape}")
print(f"Date range    : {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"States covered: {df['Region'].nunique()}")
print(f"Areas         : {df['Area'].unique()}")

# ── 2. Key Statistics ────────────────────────────────────────────────────────
pre  = df[df['Period'] == 'Pre-Covid'][unemp_col].mean()
post = df[df['Period'] == 'Covid-19'][unemp_col].mean()
peak = df.loc[df[unemp_col].idxmax()]

print(f"\n── Covid-19 Impact ──────────────────────────────────")
print(f"Pre-Covid avg unemployment  : {pre:.2f}%")
print(f"Covid-19 avg unemployment   : {post:.2f}%")
print(f"Increase                    : +{post - pre:.2f} percentage points")
print(f"Peak unemployment           : {peak[unemp_col]:.2f}% "
      f"({peak['Region']}, {peak['Date'].date()})")

# ── 3. Aggregations ───────────────────────────────────────────────────────────
monthly_avg = df.groupby('Date')[unemp_col].mean().reset_index()
state_covid = (df[df['Period'] == 'Covid-19']
               .groupby('Region')[unemp_col].mean()
               .sort_values(ascending=False))
area_monthly = df.groupby(['Date', 'Area'])[unemp_col].mean().reset_index()
corr = df[[unemp_col, employ_col, labour_col]].corr()

print(f"\n── Top 5 Worst-Hit States (Covid Period) ────────────")
print(state_covid.head(5).to_string())

print(f"\n── Rural vs Urban (Covid Period) ────────────────────")
for area in ['Rural', 'Urban']:
    avg = df[(df['Period'] == 'Covid-19') & (df['Area'] == area)][unemp_col].mean()
    print(f"  {area}: {avg:.2f}%")

# ── 4. Figure 1 — National Trend & Covid Impact ───────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Unemployment Analysis in India (2019–2020)',
             fontsize=18, fontweight='bold')

# (A) National monthly unemployment trend
ax = axes[0, 0]
bar_colors = ['#d62728' if d >= pd.Timestamp('2020-03-01') else '#1f77b4'
              for d in monthly_avg['Date']]
ax.bar(monthly_avg['Date'], monthly_avg[unemp_col],
       color=bar_colors, width=20, alpha=0.85)
ax.axvline(pd.Timestamp('2020-03-01'), color='black',
           linestyle='--', linewidth=1.5)
blue_p = mpatches.Patch(color='#1f77b4', label=f'Pre-Covid avg: {pre:.1f}%')
red_p  = mpatches.Patch(color='#d62728', label=f'Covid-19 avg: {post:.1f}%')
ax.legend(handles=[blue_p, red_p], loc='upper left', fontsize=9)
ax.set_title('National Monthly Unemployment Rate', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Unemployment Rate (%)')
ax.tick_params(axis='x', rotation=45)

# (B) Top 10 worst-hit states during Covid
ax = axes[0, 1]
top10 = state_covid.head(10)
top10.plot(kind='barh', ax=ax, color='#d62728', alpha=0.85)
ax.set_title('Top 10 States — Avg Unemployment During Covid-19',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Avg Unemployment Rate (%)')
ax.set_ylabel('')
for i, v in enumerate(top10.values):
    ax.text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=9)
ax.invert_yaxis()

# (C) Rural vs Urban unemployment trend
ax = axes[1, 0]
for area, grp in area_monthly.groupby('Area'):
    color = '#2ca02c' if area == 'Rural' else '#9467bd'
    ax.plot(grp['Date'], grp[unemp_col], marker='o', markersize=3,
            label=area, color=color, linewidth=2)
ax.axvline(pd.Timestamp('2020-03-01'), color='black',
           linestyle='--', linewidth=1.5, label='Lockdown')
ax.set_title('Rural vs Urban Unemployment Trend', fontsize=13, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend()
ax.tick_params(axis='x', rotation=45)

# (D) Pre-Covid vs Covid distribution by area
ax = axes[1, 1]
sns.boxplot(data=df, x='Area', y=unemp_col, hue='Period',
            palette={'Pre-Covid': '#1f77b4', 'Covid-19': '#d62728'}, ax=ax)
ax.set_title('Pre-Covid vs Covid-19 Distribution by Area',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Area')
ax.set_ylabel('Unemployment Rate (%)')
ax.legend(title='Period')

plt.tight_layout()
plt.savefig('outputs/fig1_trend_covid.png', dpi=150, bbox_inches='tight')
print("\nFig 1 saved → outputs/fig1_trend_covid.png")

# ── 5. Figure 2 — Regional Heatmap & Correlations ────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
fig2.suptitle('Regional Patterns & Feature Correlations',
              fontsize=16, fontweight='bold')

# (A) State × Month heatmap
pivot = df.groupby(['Region', 'Month'])[unemp_col].mean().unstack()
pivot.columns = [str(c) for c in pivot.columns]
ax = axes2[0]
sns.heatmap(pivot, cmap='YlOrRd', ax=ax, linewidths=0.3,
            cbar_kws={'label': 'Unemployment Rate (%)'})
ax.set_title('State-wise Unemployment Heatmap (Monthly)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('State')
ax.tick_params(axis='x', rotation=45, labelsize=7)
ax.tick_params(axis='y', labelsize=8)

# (B) Feature correlation matrix
ax = axes2[1]
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            square=True, linewidths=0.5,
            xticklabels=['Unemployment\nRate', 'Employed', 'Labour\nParticipation'],
            yticklabels=['Unemployment\nRate', 'Employed', 'Labour\nParticipation'])
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/fig2_regional_corr.png', dpi=150, bbox_inches='tight')
print("Fig 2 saved → outputs/fig2_regional_corr.png")

print("\n── Policy Insights ──────────────────────────────────")
print("1. Covid-19 nearly doubled the national unemployment rate.")
print("2. Urban areas were hit harder than rural during lockdowns.")
print("3. Puducherry, Jharkhand & Bihar showed the highest spikes.")
print("4. Labour participation dropped sharply — many left the workforce.")
print("5. Unemployment is negatively correlated with labour participation.")
