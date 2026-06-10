"""
=============================================================
  LOAN APPROVAL CLASSIFICATION — PIPELINE COMPLETO
  Empresa Fictícia: CredTech Brasil
=============================================================
  1. Carregamento & Visão Geral
  2. Análise Exploratória dos Dados (EDA)
  3. Engenharia de Atributos
  4. Pré-processamento
  5. Modelagem (Random Forest + XGBoost comparativo)
  6. Métricas de Avaliação
  7. Visualizações Profissionais
=============================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, matthews_corrcoef
)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ────────────────── Paleta profissional ──────────────────
PALETTE = {
    'bg':       '#0F1117',
    'card':     '#1A1D2E',
    'accent1':  '#4F8EF7',
    'accent2':  '#F7924F',
    'accent3':  '#4FF7A0',
    'accent4':  '#F74F7C',
    'text':     '#E8EAF6',
    'muted':    '#6B7280',
    'grid':     '#2A2D3E',
}

plt.rcParams.update({
     'figure.facecolor':  PALETTE['bg'],
    'axes.facecolor':    PALETTE['card'],
    'axes.edgecolor':    PALETTE['grid'],
    'axes.labelcolor':   PALETTE['text'],
    'xtick.color':       PALETTE['muted'],
    'ytick.color':       PALETTE['muted'],
    'text.color':        PALETTE['text'],
    'grid.color':        PALETTE['grid'],
    'grid.linewidth':    0.6,
    'font.family':       'DejaVu Sans',
    'axes.titlesize':    13,
    'axes.labelsize':    11,
    'axes.titlepad':     12,
    'figure.dpi':        130,
})

 
# =============================================================
# 1. CARREGAMENTO
# =============================================================

print("\n" + "="*60)
print("LOAN APPROVAL --- CredTech BR")
print("="*60)

df = pd.read_csv('loan_data.csv')
print(f"\n📊 Shape: {df.shape}")
print(f"\n{'-'*40}")
print(df.dtypes)
print(f"\n{'-'*40}")
print("Valores Nulos")
print(df.isnull().sum())

print(f"\n{'-'*40}")
print("Dsitribuição do Target")
vc = df['aprovado'].value_counts()
print(vc)
print(f"\nDesbalanceamento: {vc[0]/vc[1]:.2f}:1  (negado:aprovado)")


# =============================================================
# 2. EDA — ANÁLISE EXPLORATÓRIA
# =============================================================
print("\n\n⚡ Gerando visualizações EDA...")

# ── Fig 1: Visão Geral do Dataset ──
fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor(PALETTE['bg'])
fig.suptitle("CredTech BR - Análise e Aprovação de Empréstimos",
             fontsize=18, fontweight='bold', color=PALETTE['text'], y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)


# 2.1 Distribução de Target
ax1 = fig.add_subplot((gs[0, 0]))
colors = [PALETTE['accent4'], PALETTE['accent3']]
bars = ax1.bar(["Negado (0)", "Aprovado (1)"],
               [vc[0], vc[1]], color=colors, width=0.55, edgecolor='none')
ax1.set_title("Distribuição do Target", fontweight="bold")
ax1.set_ylabel("Quantidade")
for bar, val in zip(bars, [vc[0], vc[1]]):
    pct = val / len(df) * 100
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
             f'{val:,}\n({pct:.1f}%)', ha='center', fontsize=9.5,
             color=PALETTE['text'], fontweight='bold')
ax1.set_ylim(0, vc[1] * 1.22)
ax1.grid(axis='y', alpha=0.3)
ax1.set_axisbelow(True)


# 2.2 Score de Crédito por Aprovação
ax2 = fig.add_subplot(gs[0, 1])
for val, color, label in [(0, PALETTE['accent4'], 'Negado'), (1, PALETTE['accent3'], 'Aprovado')]:
    ax2.hist(df[df['aprovado']==val]['score_credito'], bins=30, alpha=0.7,
             color=color, label=label, edgecolor='none')
ax2.set_title("Score de Crédito", fontweight='bold')
ax2.set_xlabel("Score")
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)


# 2.3 Renda Mensal (log)
ax3 = fig.add_subplot(gs[0, 2])
for val, color, label in [(0, PALETTE['accent4'], 'Negado'), (1, PALETTE['accent3'], 'Aprovado')]:
    ax3.hist(np.log1p(df[df['aprovado']==val]['renda_mensal']), bins=30, alpha=0.7,
             color=color, label=label, edgecolor='none')
ax3.set_title("Renda Mensal (log)", fontweight='bold')
ax3.set_xlabel('log(Renda)')
ax3.legend(fontsize=9)
ax3.grid(axis='y', alpha=0.3)


# 2.4 Inadimplências
ax4 = fig.add_subplot(gs[0, 3])
inad_rate = df.groupby('inadimplencias')['aprovado'].mean()
bars4 = ax4.bar(inad_rate.index, inad_rate.values,
                color=[PALETTE['accent1'], PALETTE['accent2'], PALETTE['accent3'], PALETTE['accent4']],
                edgecolor='none', width=0.6)
ax4.set_title("Taxa de Aprovação x Inadimplências", fontweight='bold')
ax4.set_xlabel("Qtd. Inadimplências")
ax4.set_ylabel("Taxa de Aprovação")
ax4.set_ylim(0, 1.1)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
for bar in bars4:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.1%}', ha='center', fontsize=9, color=PALETTE['text'])
ax4.grid(axis='y', alpha=0.3)


# 2.5 Histórico de Pagamento
ax5 = fig.add_subplot(gs[1, 0:2])
order = ["Excelente", "Bom", "Regular", "Ruim", "Péssimo"]
hp_rate = df.groupby("historico_pagamento")["aprovado"].mean().reindex(order)
bar_color = [PALETTE['accent3'], PALETTE['accent1'], PALETTE['accent2'],
              PALETTE['accent4'], '#B91C1C']
bars5 = ax5.barh(order, hp_rate.values, color=bar_color, edgecolor='none', height=0.6)
ax5.set_title('Taxa de Aprovação por Histórico de Pagamento', fontweight='bold')
ax5.set_xlabel('Taxa de Aprovação')
ax5.set_xlim(0, 1.15)
ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
for bar in bars5:
    ax5.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.1%}', va='center', fontsize=9.5, color=PALETTE['text'])
ax5.grid(axis='x', alpha=0.3)


# 2.6 Tipo de Emprego
ax6 = fig.add_subplot(gs[1, 2:4])
te_rate = df.groupby("tipo_emprego")["aprovado"].mean().sort_values(ascending=True)
bars6 = ax6.barh(te_rate.index, te_rate.values,
                 color=PALETTE['accent1'], edgecolor='none', height=0.6)
ax6.set_title('Taxa de Aprovação por Tipo de Emprego', fontweight='bold')
ax6.set_xlabel('Taxa de Aprovação')
ax6.set_xlim(0, 1.15)
ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
for bar in bars6:
    ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
             f'{bar.get_width():.1%}', va='center', fontsize=9.5, color=PALETTE['text'])
ax6.grid(axis='x', alpha=0.3)


# 2.7 Correlação heatmap (numéricas)
ax7 = fig.add_subplot(gs[2, 0:2])
num_cols = ['idade', 'anos_emprego', 'renda_mensal', 'valor_emprestimo',
            'score_credito', 'divida_total', 'inadimplencias', 'aprovado']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, ax=ax7,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            linewidths=0.5, linecolor=PALETTE['bg'],
            cbar_kws={'shrink': 0.8})
ax7.set_title("Correlação Entre as Variáveis Númericas", fontweight='bold')
ax7.tick_params(axis='x', rotation=45)


# 2.8 Taxa aprovação por região
ax8 = fig.add_subplot(gs[2, 2])
reg_rate = df.groupby("regiao")["aprovado"].mean().sort_values(ascending=False)
bars8 = ax8.bar(reg_rate.index, reg_rate.values,
                color=PALETTE['accent2'], edgecolor='none', width=0.6)
ax8.set_title('Aprovação por Região', fontweight='bold')
ax8.set_ylabel('Taxa de Aprovação')
ax8.set_ylim(0, 1.15)
ax8.xaxis.set_tick_params(rotation=30)
ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
for bar in bars8:
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{bar.get_height():.0%}', ha='center', fontsize=9, color=PALETTE['text'])
ax8.grid(axis='y', alpha=0.3)


# 2.9 Boxplot Score × Aprovação
ax9 = fig.add_subplot(gs[2, 3])
data_neg = df[df['aprovado']==0]["score_credito"]
data_pos = df[df['aprovado']==1]["score_credito"]
bp = ax9.boxplot([data_neg, data_pos],
                 patch_artist=True, notch=True,
                 medianprops={'color': PALETTE['bg'], 'linewidth': 2})
bp['boxes'][0].set_facecolor(PALETTE['accent4'])
bp['boxes'][1].set_facecolor(PALETTE['accent3'])
ax9.set_xticks([1, 2])
ax9.set_xticklabels(['Negado', 'Aprovado'])
ax9.set_title('Score Crédito × Decisão', fontweight='bold')
ax9.set_ylabel('Score de Crédito')
ax9.grid(axis='y', alpha=0.3)
 
plt.savefig('eda_overview.png', bbox_inches='tight', dpi=130,
            facecolor=PALETTE['bg'])
plt.close()
print("   ✓ eda_overview.png")


# =============================================================
# 3. ENGENHARIA DE ATRIBUTOS
# =============================================================
print("\n🔧 Engenharia de Atributos...")

df_feat = df.copy()

# Ratios Financeiros
df_feat['ratio_divida_renda'] = df_feat['divida_total'] / (df_feat['renda_mensal'] + 1)
df_feat['ratio_emprestimo_renda'] = df_feat['valor_emprestimo'] / (df_feat['renda_mensal'] + 1)
df_feat['capacidade_pagamento'] = df_feat['renda_mensal'] / (df_feat['valor_emprestimo'] / 60 + 1)


# Score Normalizado
df_feat['score_normalizado'] = (df_feat['score_credito'] - 300) / 550


# Risco Composto
df_feat['risco_composto'] = (
    df_feat['inadimplencias'] * 2
    + (df_feat['historico_pagamento'].map(
        {'Excelente': -2, 'Bom': -1, 'Regular': 0, 'Ruim': 2, 'Pessimo': 4}
    ))
    + df_feat['ratio_divida_renda'].clip(0, 10)
)


# Estabilidade Financeira
df_feat['estabilidade'] = (
    df_feat['anos_emprego'] * 0.3
    + df_feat['tempo_conta_bancaria'] * 0.2
    + df_feat['possui_imovel'] * 3
    + df_feat['possui_veiculo'] * 1
    + (df_feat['tipo_emprego'] == 'Servidor_Publico').astype(int) * 3
)


# Faixas Etarias
df_feat['faixa_etaria'] = pd.cut(df_feat['idade'],
                                 bins=[17, 25, 35, 45, 55, 75],
                                 labels=['18-25', '26-35', '36-45', '46-55', '56+'])


# Log Transforms
df_feat['log_renda'] = np.log1p(df_feat['renda_mensal'])
df_feat['log_divida'] = np.log1p(df_feat['divida_total'])
df_feat['log_emprestimo'] = np.log1p(df_feat['valor_emprestimo'])

new_features = ['ratio_divida_renda', 'ratio_emprestimo_renda', 'capacidade_pagamento',
                'score_normalizado', 'risco_composto', 'estabilidade',
                'log_renda', 'log_divida', 'log_emprestimo']
print(f"✓ {len(new_features)} novos atributos criados")
print(f"Features totais: {df_feat.shape[1]} colunas")


# =============================================================
# 4. PRÉ-PROCESSAMENTO
# =============================================================
print("\n⚙️  Pré-processamento...")

# Encoding
cat_cols = ['estado_civil', 'nivel_educacao', 'tipo_emprego', 'regiao',
            'historico_pagamento', 'finalidade_emprestimo', 'faixa_etaria']

le = LabelEncoder()
for col in cat_cols:
    df_feat[col] = le.fit_transform(df_feat[col].astype(str))


# Drop colunas redundantes após log transform
drop_cols = ['renda_mensal', 'divida_total', 'valor_emprestimo']
df_model = df_feat.drop(columns=drop_cols)

X = df_model.drop(columns=['aprovado'])
y = df_model['aprovado']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Train class dist: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")


# =============================================================
# 5. MODELAGEM — RANDOM FOREST COM SMOTE
# =============================================================
print("\n🌲 Treinando Random Forest com SMOTE...")

smote = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"   Após SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_res, y_res)
print("   ✓ Modelo treinado!")


# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_roc = cross_val_score(rf, X_res, y_res, cv=cv, scoring='roc_auc')
cv_f1 = cross_val_score(rf, X_res, y_res, cv=cv, scoring='f1')
print(f"\n CV ROC-AUC: {cv_roc.mean():.4f} ± {cv_roc.std():.4f}")
print(f"CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")


# Comparativo com Logistic Regression
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_res, y_res)


# =============================================================
# 6. MÉTRICAS
# =============================================================
print("\n📈 Calculando Métricas...")

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_pred_lr = lr.predict(X_test)
y_prob_lr = lr.predict_proba(X_test)[:, 1]

def print_metrics(name, y_true, y_pred, y_prob):
    print(f"\n ── {name} ──")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"MCC: {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"Avg Prec: {average_precision_score(y_true, y_prob):.4f}")
    print(classification_report(y_true, y_pred, target_names=['Negado', 'Aprovado']))

print_metrics("Random Forest", y_test, y_pred_rf, y_prob_rf)
print_metrics("Logistic Regression", y_test, y_pred_lr, y_prob_lr)


# =============================================================
# 7. VISUALIZAÇÕES DE RESULTADOS
# =============================================================
print("\n🎨 Gerando visualizações de resultados...")

fig2 = plt.figure(figsize=(22, 16))
fig2.patch.set_facecolor(PALETTE['bg'])
fig2.suptitle('CredTech Brasil  ·  Resultados do Modelo — Random Forest',
              fontsize=18, fontweight='bold', color=PALETTE['text'], y=0.98)
gs2 = gridspec.GridSpec(3, 4, figure=fig2, hspace=0.45, wspace=0.4)


# 7.1 Confusion Matrix RF
ax = fig2.add_subplot(gs2[0, 0])
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negado', 'Aprovado'], yticklabels=['Negado', 'Aprovado'],
            linewidths=1, linecolor=PALETTE['bg'],
            annot_kws={'size': 14, 'weight': 'bold'})
ax.set_title("Matriz de Confusão\n(Random Forest)", fontweight='bold')
ax.set_ylabel("Real")
ax.set_xlabel("Predito")


# 7.2 ROC Curves
ax = fig2.add_subplot(gs2[0, 1])
for model_name, y_prob, color in [
    ("Random Forest", y_prob_rf, PALETTE['accent3']),
    ("Logistic Reg", y_prob_lr, PALETTE['accent2'])]:
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{model_name} (AUC={auc:.3f})")
ax.plot([0,1], [0,1], "--", color=PALETTE['muted'], lw=1)
ax.fill_between(*roc_curve(y_test, y_prob_rf)[:2],
                alpha=0.1, color=PALETTE['accent3'])
ax.set_title('Curva ROC', fontweight='bold')
ax.set_xlabel('FPR (1 - Especificidade)')
ax.set_ylabel('TPR (Sensibilidade)')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])


# 7.3 Precision-Recall
ax = fig2.add_subplot(gs2[0, 2])
for model_name, y_prob, color in [
    ("Random Forest", y_prob_rf, PALETTE['accent3']),
    ("Logistic Reg", y_prob_lr, PALETTE['accent2'])]:
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(rec, prec, color=color, lw=2, label=f"{model_name} (AP={ap:.3f})")
baseline = y_test.mean()
ax.axhline(baseline, color=PALETTE['muted'], linestyle="--", lw=1, label=f"Baseline: ({baseline:.2f})")
ax.fill_between(*precision_recall_curve(y_test, y_prob_rf)[1::-1],
                alpha=0.1, color=PALETTE['accent3'])
ax.set_title("Curva Precision Recall", fontweight='bold')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)


# 7.4 Métricas Comparativas
ax = fig2.add_subplot(gs2[0, 3])
metrics_names = ['ROC-AUC', 'F1', 'Precision', 'Recall', 'MCC']
rf_scores = [
    roc_auc_score(y_test, y_prob_rf),
    f1_score(y_test, y_pred_rf),
    precision_score(y_test, y_pred_rf),
    recall_score(y_test, y_pred_rf),
    (matthews_corrcoef(y_test, y_pred_rf) + 1) / 2  # normalize to 0-1
]
lr_scores = [
    roc_auc_score(y_test, y_prob_lr),
    f1_score(y_test, y_pred_lr),
    precision_score(y_test, y_pred_lr),
    recall_score(y_test, y_pred_lr),
    (matthews_corrcoef(y_test, y_pred_lr) + 1) / 2
]
x = np.arange(len(metrics_names))
width = 0.35
ax.bar(x - width/2, rf_scores, width, label='Random Forest', color=PALETTE['accent3'],
       edgecolor='none', alpha=0.85)
ax.bar(x + width/2, lr_scores, width, label='Logistic Reg', color=PALETTE['accent2'],
       edgecolor='none', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(metrics_names, fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_title('Comparativo de Métricas', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_axisbelow(True)
for i, (v1, v2) in enumerate(zip(rf_scores, lr_scores)):
    ax.text(i - width/2, v1 + 0.02, f'{v1:.2f}', ha='center', fontsize=7.5, color=PALETTE['text'])
    ax.text(i + width/2, v2 + 0.02, f'{v2:.2f}', ha='center', fontsize=7.5, color=PALETTE['text'])


# 7.5 Feature Importance
ax = fig2.add_subplot(gs2[1, 0:2])
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True).tail(18)
colors_fi = [PALETTE['accent1'] if v > feat_imp.quantile(0.75) else PALETTE['muted']
             for v in feat_imp.values]
bars = ax.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor='none', height=0.7)
ax.set_title('Feature Importance — Top 18 Atributos', fontweight='bold')
ax.set_xlabel('Importância Média (Gini)')
ax.grid(axis='x', alpha=0.3)
for bar in bars:
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{bar.get_width():.4f}', va='center', fontsize=8, color=PALETTE['muted'])


# 7.6 Score distribution by class
ax = fig2.add_subplot(gs2[1, 2])
ax.hist(y_prob_rf[y_test == 0], bins=40, alpha=0.7, color=PALETTE['accent4'],
        label='Negado (Real)', density=True, edgecolor='none')
ax.hist(y_prob_rf[y_test == 1], bins=40, alpha=0.7, color=PALETTE['accent3'],
        label='Aprovado (Real)', density=True, edgecolor='none')
ax.axvline(0.5, color='white', linestyle='--', lw=1.5, label='Threshold 0.5')
ax.set_title('Dist. de Probabilidade Predita', fontweight='bold')
ax.set_xlabel('P(Aprovado)')
ax.set_ylabel('Densidade')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)


# 7.7 Threshold Analysis
ax = fig2.add_subplot(gs2[1, 3])
thresholds = np.linspace(0.1, 0.9, 80)
f1s, precs, recs = [], [], []
for t in thresholds:
    yp = (y_prob_rf >= t).astype(int)
    f1s.append(f1_score(y_test, yp, zero_division=0))
    precs.append(precision_score(y_test, yp, zero_division=0))
    recs.append(recall_score(y_test, yp, zero_division=0))
 
ax.plot(thresholds, f1s,   color=PALETTE['accent3'],  lw=2, label='F1')
ax.plot(thresholds, precs, color=PALETTE['accent1'],  lw=2, label='Precision')
ax.plot(thresholds, recs,  color=PALETTE['accent2'],  lw=2, label='Recall')
best_t = thresholds[np.argmax(f1s)]
ax.axvline(best_t, color='white', linestyle='--', lw=1.5, label=f'Melhor Threshold ({best_t:.2f})')
ax.set_title('Análise de Threshold', fontweight='bold')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)


# 7.8 CV scores boxplot
ax = fig2.add_subplot(gs2[2, 0])
cv_data = {
    'ROC-AUC': cross_val_score(rf, X_res, y_res, cv=cv, scoring='roc_auc'),
    'F1':      cross_val_score(rf, X_res, y_res, cv=cv, scoring='f1'),
    'Precision': cross_val_score(rf, X_res, y_res, cv=cv, scoring='precision'),
    'Recall':  cross_val_score(rf, X_res, y_res, cv=cv, scoring='recall'),
}
bp = ax.boxplot(cv_data.values(), patch_artist=True, notch=True,
                medianprops={'color': PALETTE['bg'], 'linewidth': 2})
box_colors = [PALETTE['accent3'], PALETTE['accent1'], PALETTE['accent2'], PALETTE['accent4']]
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.set_xticklabels(cv_data.keys())
ax.set_title('Validação Cruzada (5-Fold)', fontweight='bold')
ax.set_ylabel('Score')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.5, 1.05)


# 7.9 Engenharia de atributos — correlação com target
ax = fig2.add_subplot(gs2[2, 1:3])
eng_feats = ['ratio_divida_renda', 'ratio_emprestimo_renda', 'capacidade_pagamento',
             'score_normalizado', 'risco_composto', 'estabilidade',
             'log_renda', 'log_divida', 'log_emprestimo']
corrs = df_feat[eng_feats + ['aprovado']].corr()['aprovado'][:-1].sort_values()
colors_c = [PALETTE['accent4'] if v < 0 else PALETTE['accent3'] for v in corrs.values]
bars_c = ax.barh(corrs.index, corrs.values, color=colors_c, edgecolor='none', height=0.65)
ax.axvline(0, color=PALETTE['muted'], lw=0.8)
ax.set_title('Correlação dos Atributos Criados com Target', fontweight='bold')
ax.set_xlabel('Correlação de Pearson')
ax.grid(axis='x', alpha=0.3)
for bar in bars_c:
    v = bar.get_width()
    x_pos = v + 0.005 if v >= 0 else v - 0.005
    ha = 'left' if v >= 0 else 'right'
    ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{v:.3f}', va='center',
            ha=ha, fontsize=8.5, color=PALETTE['text'])


# 7.10 Aprovação por faixa etária e educação
ax = fig2.add_subplot(gs2[2, 3])
edu_order = ['Fundamental', 'Medio', 'Superior', 'Pos-Graduacao']
edu_data = df.groupby('nivel_educacao')['aprovado'].mean().reindex(edu_order)
bar_colors_e = [PALETTE['accent2'], PALETTE['accent1'], PALETTE['accent3'], PALETTE['accent3']]
bars_e = ax.bar(edu_data.index, edu_data.values, color=bar_colors_e, edgecolor='none', width=0.6)
ax.set_title('Aprovação por Educação', fontweight='bold')
ax.set_ylabel('Taxa de Aprovação')
ax.set_ylim(0, 1.15)
ax.tick_params(axis='x', rotation=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
for bar in bars_e:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.0%}', ha='center', fontsize=9, color=PALETTE['text'])
ax.grid(axis='y', alpha=0.3)
 
plt.savefig('model_results.png', bbox_inches='tight', dpi=130, facecolor=PALETTE['bg'])
plt.close()
print("   ✓ model_results.png")


# =============================================================
# Summary Card
# =============================================================
print("\n\n" + "="*60)
print("  RESUMO FINAL — RANDOM FOREST")
print("="*60)
print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"  MCC: {matthews_corrcoef(y_test, y_pred_rf):.4f}")
print(f"  Avg Precision: {average_precision_score(y_test, y_prob_rf):.4f}")
print(f"  Melhor Threshold: {best_t:.2f}")
print(f"\n  Atributos mais importantes:")
top5 = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
for feat, val in top5.items():
    print(f"    {feat:<30} {val:.4f}")
print("\n✅ Pipeline completo! Arquivos salvos:")
print("- eda_overview.png")
print("- model_results.png")
print("="*60)