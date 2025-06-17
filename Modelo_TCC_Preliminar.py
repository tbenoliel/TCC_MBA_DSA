# Classificação de risco de interrupção de tratamento das pessoas vivendo com HIV/AIDS no Brasil

#### Autor: Tiago Benoliel   

# Versão: 1.00  

# Data de Criação:    15/05/2025                                        
                                                                           


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import log_loss, cohen_kappa_score, f1_score, RocCurveDisplay,precision_recall_curve
from collections import Counter
import warnings
from IPython.display import display

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Organização dos Bancos de Dados

## Banco geral das PVHA

# Carrega o Banco de dados
PVHA_ult_ano = pd.read_parquet("PVHA_ult_ano.parquet")

# Exclui da base as pessoas que não iniciaram TARV
DF = PVHA_ult_ano[PVHA_ult_ano["Status_ano"].isin(["Tarv", "Interrupção de Tarv"])].copy()
DF.shape

# Verifica a completude de cada variável do banco
for col in DF.columns:
    print(f"A coluna {col} possui {round(1-(len(DF[DF[col].isnull()])/len(DF)),3)*100}% registros preenchidos")
    print()

Colunas_sociodemo = [
    "Pop_genero",
    "Raca_cat2",
    "Escol_cat",
    "Idade_min_cat",
    "Idade_ult_cat"
]

for col in Colunas_sociodemo:
    DF[col].fillna("Nao Informado", inplace = True)
DF["CD4_cat"].fillna("Sem exame CD4 no SUS", inplace = True)
DF["CV_cat"].fillna("Sem exame CV no SUS", inplace = True)
DF["Grupo_instituicao"].fillna("Sem exame no SUS", inplace = True)

DF["TARV_no_ano"] = np.where(DF["Status_ano"] == "Tarv", 1,0)
DF["Interrup_no_ano"] = np.where(DF["Status_ano"] == "Interrupção de Tarv", 1,0)
DF['Interrup_ano_seguinte'] = DF.groupby('Cod_unificado')['Interrup_no_ano'].shift(-1)

DF["Abandono_raz"] = round(DF["Abandono_sum"]/DF["N_dispensas"],3)
DF["Atraso_raz"] = round(DF["Atraso_sum"]/DF["N_dispensas"],3)

## Banco com variáveis Dummies

dummie_cols = [
'Pop_genero',
'Raca_cat2',
'Escol_cat',
'Idade_min_cat',
'Idade_ult_cat',
'reg_res',
'Dias_diag_TARV_cat',
'esquema_cat',
'CD4_cat',
'CV_cat',
'Grupo_instituicao'
]

DF1 = pd.get_dummies(DF, columns = dummie_cols,drop_first = False)

drop_cols = [
"Cod_unificado",
'cod_ibge6_res',
"Status_ano",
"Abandono_sum",
"Atraso_sum",
"N_dispensas",
'Pop_genero_Nao Informado',
'Raca_cat2_Nao Informado',
'Escol_cat_Nao Informado',
'Idade_min_cat_Nao Informado',
'Idade_ult_cat_Nao Informado',
'reg_res_Centro-Oeste',
'esquema_cat_Outro Esquema',
'Dias_diag_TARV_cat_No mesmo dia',
'CD4_cat_Sem exame CD4 no SUS',
'CV_cat_Sem exame CV no SUS',
'Grupo_instituicao_Sem exame no SUS'
]

DF1.drop(columns = drop_cols, inplace = True)

DF1.shape

DF1.columns

DF1.rename(columns = {
    'Grupo_instituicao_Instituições acompanhando entre 500 e 1000 pessoas':'Instituicao 500-1000',
     'Grupo_instituicao_Instituições acompanhando mais de 1000 pessoas':'Instituicao 1000',
    'Grupo_instituicao_Instituições acompanhando menos de 500 pessoas':'Instituicao 500'
}, inplace = True)

## Criação dos bancos anuais

DF_dict = {}
for ano_analise in range(2015,2024):
    Base = DF1[(DF1["ano"] == ano_analise) & (DF1["TARV_no_ano"] == 1) & (DF1["Interrup_ano_seguinte"].isnull() == False)].copy()
    Base.drop(columns = ["ano","TARV_no_ano","Interrup_no_ano"], inplace = True)
    Base.columns = [col.replace('<', '_').replace('/', '_').replace('+', '_')
                           for col in Base.columns]
    DF_dict[str(ano_analise)] = Base
    print(f"Base {ano_analise}:", len(Base))
    print()
print(f"As bases apresentam {Base.shape[1]} colunas")

## Análise exploratória da variável de resposta

# Analisando a proporção de pessoas que interrompem o tratamento no ano seguinte ao longo do período.
anos = []
percentuais = []

for ano, df in DF_dict.items():
    total = len(df)
    if total > 0:
        percentual = (df["Interrup_ano_seguinte"].sum() / total) * 100
    else:
        percentual = 0
    anos.append(int(ano))
    percentuais.append(percentual)

# Ordena os anos (caso as chaves não estejam em ordem)
anos, percentuais = zip(*sorted(zip(anos, percentuais)))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(anos, percentuais, marker='o', linestyle='-')
plt.xlabel("Ano")
plt.ylabel("Interrupção no ano seguinte (%)")
plt.grid(False)
plt.ylim(0, max(percentuais) * 1.1)  # margem superior no gráfico

# Remova a moldura do gráfico
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)
plt.savefig(f"Interrupcao_por_ano.png",dpi=500)
plt.show()

# Calcula a média da interrupção no ano seguinte com o desvio padrão
print(np.mean(percentuais))
print(np.std(percentuais))

# Avalia a correlação entre as variáveis 
for ano, df in DF_dict.items():
    
    # Calcula a matriz de correlação
    corr_matrix = df.corr()
    
    # Define o limiar
    threshold = 0.5
    
    # Zera a diagonal para ignorar a correlação da variável com ela mesma
    corr_no_diag = corr_matrix.copy()
    np.fill_diagonal(corr_no_diag.values, 0)
    
    # Identifica as variáveis que têm pelo menos uma correlação forte
    vars_to_keep = corr_no_diag.columns[
        (abs(corr_no_diag) >= threshold).any()
    ]
    
    # Filtra a matriz
    corr_filtered = corr_matrix.loc[vars_to_keep, vars_to_keep]
    
    # Aplica máscara para deixar os fracos em branco
    mask = abs(corr_filtered) < threshold
    
    # Plota o heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_filtered, annot=True, fmt=".2f", cmap="coolwarm", center=0, mask=mask)
    plt.savefig(f"Correlacao_{ano}.png",dpi=500,bbox_inches = "tight")
    plt.show()

# Exclui a variável Tempo_diag por alta correlação com Tempo TARV
for ano, df in DF_dict.items():

    DF_dict[ano] = DF_dict[ano].drop(columns = ["Tempo_Diag"])

## Treinamento e Avaliação do Modelo

param_grid = {
    'n_estimators': [100],          # Evite exagerar no nº de árvores inicialmente; refine depois
    'max_depth': [3, 5],            # 3 e 5 são bons para evitar overfitting; 7 se quiser explorar maior capacidade
    'learning_rate': [0.1, 0.3],    # 0.05 (mais conservador), 0.1 (padrão), 0.2 (aprendizado mais rápido)
    'subsample': [0.6, 0.8],        # Combate overfitting e reduz custo
    'colsample_bytree': [0.6, 0.8], # Idem acima
    'min_child_weight': [1],        # Para controlar complexidade
    'gamma': [0]                    # Regularização: 0 (nenhuma), 1 (mais conservador)
}

# Acumular dados
X_acumulado = None
y_acumulado = None

dict_metrics_corte = {}

desempenho_geral = []

for ano_treino in range (2015,2023):

    y_treino = DF_dict[f"{ano_treino}"]["Interrup_ano_seguinte"]
    X_treino =  DF_dict[f"{ano_treino}"].drop(columns = ["Interrup_ano_seguinte"])
    
    y_teste = DF_dict[f"{ano_treino+1}"]["Interrup_ano_seguinte"]
    X_teste = DF_dict[f"{ano_treino+1}"].drop(columns = ["Interrup_ano_seguinte"])


    # Acumula
    if X_acumulado is None:
        X_acumulado = X_treino
        y_acumulado = y_treino
    else:
        X_acumulado = pd.concat([X_acumulado, X_treino], axis=0)
        y_acumulado = pd.concat([y_acumulado, y_treino], axis=0)

    # Ajusta o scale_pos_weight
    counter = Counter(y_acumulado)
    scale_pos_weight = counter[0] / counter[1]

    # Configura o modelo base
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )

    # Faz GridSearch
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    # Treina o modelo acumulado
    grid_search.fit(X_acumulado, y_acumulado)

    # Melhor modelo
    best_model = grid_search.best_estimator_
    print(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
    print()

    # Calcula os valores preditos para a base de teste (X_test)
    y_pred_proba = best_model.predict_proba(X_teste)[:,1]
    y_pred = best_model.predict(X_teste)
    
    # Cria e mostra Métricas gerais do modelo
    roc_auc = roc_auc_score(y_teste, y_pred_proba) 
    logloss = log_loss(y_teste, y_pred_proba)

    desempenho_geral.append({
        'Ano': ano_treino,
        'roc_auc': roc_auc,
        'logloss': logloss
    })
    print()
    print(f"roc_auc: {roc_auc:.2f}")
    print()          
    print(f"Log Loss: {logloss:.2f}")
    print()
    print(classification_report(y_teste, y_pred))
    
    # Cria a Curva ROC
    ax = plt.gca()
    RocCurveDisplay.from_estimator(best_model, X_teste, y_teste, ax=ax, alpha=0.8)
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Taxa Falso Positivo')
    plt.ylabel('Taxa Verdadeiro Positivo')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.savefig(f"ROC_{ano_treino}.png",dpi=500)
        
    plt.show()
    print()

    # Gerar thresholds
    thresholds = np.linspace(0, 1, 100)
    
    sens_list = []
    spec_list = []
    prec_list = []
    f1_list = []
    
    for thr in thresholds:
        y_pred_thr = (y_pred_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_teste, y_pred_thr).ravel()
    
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
        
        sens_list.append(sens)
        spec_list.append(spec)
        prec_list.append(prec)
        f1_list.append(f1)
    
    # Plot
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, sens_list, label='Sensibilidade (Recall)', color='green')
    plt.plot(thresholds, spec_list, label='Especificidade', color='blue')
    plt.plot(thresholds, prec_list, label='Precisão', color='orange')
    plt.xlabel('Threshold')
    plt.ylabel('Valor')
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    plt.legend(bbox_to_anchor=(0.9, -0.1), ncols = 3)
    plt.grid(False)
    plt.savefig(f"Sensi_Espec_Prec_{ano_treino}.png",dpi=500,bbox_inches = "tight")
    plt.show()
    
    # Gerar tabela
    df_metrics = pd.DataFrame({
        'Threshold': thresholds,
        'Sensibilidade': sens_list,
        'Especificidade': spec_list,
        'Precisão': prec_list,
        'F1_Score': f1_list
    })

    dict_metrics_corte[ano_treino] = df_metrics

df_desempenho = pd.DataFrame(desempenho_geral)

# Gráfico
fig, ax1 = plt.subplots(figsize=(10, 6))

# Linha 1 - AUC
line1, = ax1.plot(df_desempenho['Ano']+1, df_desempenho['roc_auc'], color='blue', marker='o', label='AUC')
ax1.set_xlabel('Ano')
ax1.set_ylabel('AUC', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_ylim(0, 1)

# Linha 2 - LogLoss no segundo eixo
ax2 = ax1.twinx()
line2, = ax2.plot(df_desempenho['Ano']+1, df_desempenho['logloss'], color='red', marker='s', label='LogLoss')
ax2.set_ylabel('LogLoss', color='red')
ax2.tick_params(axis='y', labelcolor='red')


# Legenda combinada abaixo do gráfico
lines = [line1, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05))

# Ajusta layout para dar espaço à legenda
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # espaço extra para legend

ax1.spines['top'].set_visible(False)
ax2.spines['top'].set_visible(False)


# Salva figura
plt.savefig('AUC_LogLoss_anos.png', dpi=500,bbox_inches = "tight")

plt.show()

# Define o threshold de interesse
thresholds_interesse = [0.5, 0.6, 0.7, 0.8, 0.9]

# Itera pelos anos armazenados
for corte in thresholds_interesse:
    
    # Lista para armazenar resultados
    historico_threshold = []
    for ano, df in dict_metrics_corte.items():
        
        # Pega a linha mais próxima do threshold fixo
        linha_proxima = df.iloc[(df['Threshold'] - corte).abs().argmin()]
        
        # Armazena os dados
        historico_threshold.append({
            'Ano': ano+1,
            'Threshold': linha_proxima['Threshold'],
            'Sensibilidade': linha_proxima['Sensibilidade'],
            'Especificidade': linha_proxima['Especificidade'],
            'Precisão': linha_proxima['Precisão'],
            'F1_Score': linha_proxima['F1_Score']
        })
    
    # Converte para DataFrame
    df_evolucao = pd.DataFrame(historico_threshold).sort_values("Ano")

    plt.figure(figsize=(10,6))
    plt.plot(df_evolucao['Ano'], df_evolucao['Sensibilidade'], marker='o', label='Sensibilidade (Recall)')
    plt.plot(df_evolucao['Ano'], df_evolucao['Especificidade'], marker='o', label='Especificidade')
    plt.plot(df_evolucao['Ano'], df_evolucao['Precisão'], marker='o', label='Precisão')
    plt.plot(df_evolucao['Ano'], df_evolucao['F1_Score'], marker='o', label='F1 Score')
    
    plt.xlabel("Ano")
    plt.ylabel("Valor da Métrica")
    plt.ylim(0, 1)
    plt.grid(False)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legenda horizontal
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(f'Metricas_no_corte_{corte}.png', dpi=500,bbox_inches = "tight")
    plt.show()