# Exemplo de uso (opcional, para teste)

import json
from pathlib import Path

from adapters.log.logger import logger
from core.heuristic_exploration import GherkinStepMatcherExploration

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Heurística para encontrar uso de arquivos .feature com processamento paralelo')
    parser.add_argument('--clone-dir', required=True, help='Diretório onde os repositórios estão clonados')
    parser.add_argument('--data-file', required=True, help='Arquivo JSON com dados dos repositórios')
    parser.add_argument('--output', default='resultados_heuristica.json', help='Arquivo de saída para os resultados')
    parser.add_argument('--debug', action='store_true', help='Ativa modo de debug com logs detalhados')
    parser.add_argument('--workers', type=int, default=None, help='Número de processos paralelos (padrão: número de CPUs - 1)')
    
    args = parser.parse_args()

    # Configurar diretório de clone e arquivo de dados
    CLONE_DIR = Path(args.clone_dir) # Ajuste conforme necessário
    DATA_FILE = args.data_file # Ajuste conforme necessário
    OUTPUT_FILE = args.output

    # Criar diretório de clone se não existir
    CLONE_DIR.mkdir(parents=True, exist_ok=True)

    # Instanciar e executar o matcher
    matcher = GherkinStepMatcherExploration(clone_dir=CLONE_DIR, debug=True, min_confidence=0.5)
    matcher.data = matcher._load_data(args.data_file)

    # Verificar se o arquivo de dados foi carregado
    if not matcher._load_data(args.data_file):
         logger.error("Falha ao carregar dados. Abortando análise.")
    else:
        analysis_results = matcher.analyze_repositories(args.data_file)

        # Salvar resultados em JSON
        if analysis_results:
            try:
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(analysis_results, f, indent=4, ensure_ascii=False)
                logger.info(f"\n💾 Resultados da análise salvos em: {OUTPUT_FILE}")
            except Exception as e:
                logger.error(f"Erro ao salvar resultados em {OUTPUT_FILE}: {e}")
        else:
            logger.info("Nenhum resultado de análise gerado.")

