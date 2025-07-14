
import pandas as pd
import os

# Seus arquivos
files = {
    'jobs': r'C:\Users\e178454\Desktop\partic\FIAP ML\selecao\data\jobs.csv',
    'prospects': r'C:\Users\e178454\Desktop\partic\FIAP ML\selecao\data\prospects.csv',
    'applicants': r'C:\Users\e178454\Desktop\partic\FIAP ML\selecao\data\applicants.csv'
}

def safe_load_csv(file_path, name):
    """Carrega CSV de forma ultra-segura"""
    print("Processando arquivo:", name)
    
    if not os.path.exists(file_path):
        print("ERRO: Arquivo nao encontrado")
        return None
    
    # Estrategia 1: Latin-1 direto
    try:
        df = pd.read_csv(file_path, encoding='latin-1')
        print("SUCESSO com latin-1")
        print("Shape:", df.shape)
        print("Colunas:", len(df.columns))
        return df
    except Exception as e:
        print("Falhou latin-1:", str(e)[:50])
    
    # Estrategia 2: CP1252
    try:
        df = pd.read_csv(file_path, encoding='cp1252')
        print("SUCESSO com cp1252")
        print("Shape:", df.shape)
        return df
    except Exception as e:
        print("Falhou cp1252:", str(e)[:50])
    
    # Estrategia 3: Limpeza manual
    try:
        print("Tentando limpeza manual...")
        
        # Ler como bytes
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        # Converter para string removendo caracteres ruins
        clean_data = ""
        for byte in raw_data:
            if byte < 128:  # Apenas ASCII
                clean_data += chr(byte)
            elif byte == 195:  # Parte de caracteres UTF-8 validos
                clean_data += chr(byte)
            else:
                clean_data += " "  # Substituir por espaco
        
        # Salvar arquivo limpo
        temp_file = name + "_clean.csv"
        with open(temp_file, 'w', encoding='ascii', errors='ignore') as f:
            f.write(clean_data)
        
        # Carregar arquivo limpo
        df = pd.read_csv(temp_file)
        print("SUCESSO com limpeza manual")
        print("Shape:", df.shape)
        
        # Remover arquivo temporario
        os.remove(temp_file)
        return df
        
    except Exception as e:
        print("Limpeza manual falhou:", str(e)[:50])
    
    print("TODOS OS METODOS FALHARAM")
    return None

# EXECUTAR
print("INICIANDO CARREGAMENTO SEGURO")
print("="*40)

results = {}

for name, path in files.items():
    print("\n" + "-"*30)
    df = safe_load_csv(path, name)
    if df is not None:
        results[name] = df

print("\n" + "="*40)
print("RESULTADO FINAL:")

if len(results) == 3:
    print("SUCESSO TOTAL! Todos os 3 arquivos carregados")
    
    # Criar variaveis
    df_jobs = results['jobs']
    df_prospects = results['prospects']  
    df_applicants = results['applicants']
    
    print("\nResumo:")
    print("df_jobs:", df_jobs.shape)
    print("df_prospects:", df_prospects.shape)
    print("df_applicants:", df_applicants.shape)
    
    print("\nColunas jobs:", list(df_jobs.columns))
    print("Colunas prospects:", list(df_prospects.columns))
    print("Colunas applicants:", list(df_applicants.columns))
    
    print("\nPrimeiras linhas jobs:")
    print(df_jobs.head(2))
    
else:
    print("Parcialmente carregados:", len(results), "/3")
    for name in results:
        print("OK:", name, results[name].shape)

print("\nAgora voce pode usar:")
print("df_jobs.info()")
print("df_prospects.describe()")  
print("df_applicants.head()")
