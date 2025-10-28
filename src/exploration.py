import json
import re
from pathlib import Path
from statistics import mean, median, mode, StatisticsError
from adapters.log.logger import logger
from core.heuristic_exploration import GherkinStepMatcherExploration


def normalize_name(name: str) -> str:
    """Normaliza nome removendo caracteres especiais"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())


def filter_max_confidence(data: list) -> list:
    """
    Processa lista mantendo apenas o used_in com maior confidence para cada item.
    
    Args:
        data: Lista de dicionários com resultados da análise
        
    Returns:
        Lista processada com apenas o used_in de maior confidence
    """
    processed_data = []
    
    for item in data:
        if "used_in" in item and item["used_in"]:
            max_used = max(item["used_in"], key=lambda u: u.get("confidence", 0))
            item_copy = item.copy()
            item_copy["used_in"] = [max_used]
            processed_data.append(item_copy)
        else:
            processed_data.append(item)
    
    logger.info(f"✅ Filtrado {len(processed_data)} itens mantendo maior confidence")
    return processed_data


def calculate_confidence_statistics(data: list) -> dict:
    """
    Calcula estatísticas sobre os valores de confidence.
    
    Args:
        data: Lista de dicionários com resultados
        
    Returns:
        Dicionário com estatísticas (mean, median, mode, min, max, count)
    """
    confidences = []
    
    for item in data:
        if "used_in" in item and item["used_in"]:
            for used in item["used_in"]:
                conf = used.get("confidence", 0)
                if conf > 0:
                    confidences.append(conf)
    
    if not confidences:
        logger.warning("⚠️  Nenhum valor de confidence encontrado")
        return {
            "count": 0,
            "mean": 0,
            "median": 0,
            "mode": 0,
            "min": 0,
            "max": 0
        }
    
    stats = {
        "count": len(confidences),
        "mean": mean(confidences),
        "median": median(confidences),
        "min": min(confidences),
        "max": max(confidences)
    }
    
    try:
        stats["mode"] = mode(confidences)
    except StatisticsError:
        # Sem moda única, usar a mediana como fallback
        stats["mode"] = stats["median"]
        logger.warning("⚠️  Múltiplas modas encontradas, usando mediana como fallback")
    
    return stats


def filter_by_metric(data: list, metric: str = "median", custom_threshold: float = None) -> list:
    """
    Filtra resultados baseado em uma métrica estatística de confidence.
    
    Args:
        data: Lista de dicionários com resultados
        metric: Métrica a usar como threshold ('mean', 'median', 'mode')
        custom_threshold: Valor customizado para threshold (sobrescreve a métrica)
        
    Returns:
        Lista filtrada com apenas itens acima do threshold
    """
    # Calcular estatísticas
    stats = calculate_confidence_statistics(data)
    
    if stats["count"] == 0:
        logger.warning("⚠️  Nenhum dado para filtrar")
        return data
    
    # Determinar threshold
    if custom_threshold is not None:
        threshold = custom_threshold
        logger.info(f"📊 Usando threshold customizado: {threshold:.4f}")
    else:
        threshold = stats.get(metric, stats["median"])
        logger.info(f"📊 Usando threshold baseado em '{metric}': {threshold:.4f}")
    
    # Exibir estatísticas
    logger.info("📈 Estatísticas de confidence:")
    logger.info(f"   • Count: {stats['count']}")
    logger.info(f"   • Mean: {stats['mean']:.4f}")
    logger.info(f"   • Median: {stats['median']:.4f}")
    logger.info(f"   • Mode: {stats['mode']:.4f}")
    logger.info(f"   • Min: {stats['min']:.4f}")
    logger.info(f"   • Max: {stats['max']:.4f}")
    
    # Filtrar dados
    filtered_data = []
    removed_count = 0
    
    for item in data:
        if "used_in" not in item or not item["used_in"]:
            # Manter itens sem matches
            filtered_data.append(item)
            continue
        
        # Filtrar used_in acima do threshold
        filtered_used_in = [
            used for used in item["used_in"] 
            if used.get("confidence", 0) >= threshold
        ]
        
        if filtered_used_in:
            item_copy = item.copy()
            item_copy["used_in"] = filtered_used_in
            filtered_data.append(item_copy)
        else:
            removed_count += 1
    
    logger.info(f"✅ Filtrados {len(filtered_data)}/{len(data)} itens (removidos: {removed_count})")
    
    return filtered_data


def enrich_with_file_contents(data: list, clone_dir: Path) -> list:
    """
    Enriquece os dados com o conteúdo dos arquivos .feature e arquivos matched.
    Remove itens que não conseguirem carregar ambos os arquivos com sucesso.
    
    Args:
        data: Lista de dicionários com resultados
        clone_dir: Diretório base onde os repositórios estão clonados
        
    Returns:
        Lista enriquecida com feature_content e matched_content (apenas itens completos)
    """
    enriched_data = []
    success_count = 0
    feature_errors = 0
    matched_errors = 0
    removed_incomplete = 0
    
    for entry in data:
        language = entry.get("language", "")
        repository = entry.get("repository", "")
        author_repo = repository.replace("/", "_")
        
        # Construir caminho do repositório
        repo_path = clone_dir / language / normalize_name(author_repo)
        
        # Caminho do arquivo .feature
        feature_file = entry.get("feature_file", "")
        feature_path = repo_path / feature_file
        
        # Caminho do arquivo matched (primeiro used_in)
        matched_file = ""
        if "used_in" in entry and entry["used_in"]:
            matched_file = entry["used_in"][0].get("file", "")
        matched_path = repo_path / matched_file if matched_file else None
        
        # Flags para rastrear sucesso
        feature_success = False
        matched_success = False
        
        # Ler conteúdo do feature file
        try:
            entry["feature_content"] = feature_path.read_text(encoding="utf-8")
            feature_success = True
        except UnicodeDecodeError:
            # Tentar com encoding alternativo
            try:
                entry["feature_content"] = feature_path.read_text(encoding="latin-1")
                feature_success = True
                logger.debug(f"✓ Arquivo lido com encoding latin-1: {feature_path}")
            except Exception as e:
                entry["feature_content"] = f"[ERRO ao ler feature_file: {e}]"
                feature_errors += 1
                logger.warning(f"⚠️  Erro ao ler {feature_path}: {e}")
        except Exception as e:
            entry["feature_content"] = f"[ERRO ao ler feature_file: {e}]"
            feature_errors += 1
            logger.warning(f"⚠️  Erro ao ler {feature_path}: {e}")
        
        # Ler conteúdo do matched file
        if matched_path and matched_path.exists():
            try:
                entry["matched_content"] = matched_path.read_text(encoding="utf-8")
                matched_success = True
            except UnicodeDecodeError:
                # Tentar com encoding alternativo
                try:
                    entry["matched_content"] = matched_path.read_text(encoding="latin-1")
                    matched_success = True
                    logger.debug(f"✓ Arquivo lido com encoding latin-1: {matched_path}")
                except Exception as e:
                    entry["matched_content"] = f"[ERRO ao ler matched_file: {e}]"
                    matched_errors += 1
                    logger.warning(f"⚠️  Erro ao ler {matched_path}: {e}")
            except Exception as e:
                entry["matched_content"] = f"[ERRO ao ler matched_file: {e}]"
                matched_errors += 1
                logger.warning(f"⚠️  Erro ao ler {matched_path}: {e}")
        else:
            entry["matched_content"] = "[Arquivo matched não encontrado]"
            matched_errors += 1
        
        # Apenas adicionar se AMBOS os arquivos foram lidos com sucesso
        if feature_success and matched_success:
            enriched_data.append(entry)
            success_count += 1
        else:
            removed_incomplete += 1
            logger.debug(
                f"✗ Item removido (feature: {feature_success}, matched: {matched_success}): "
                f"{repository}/{feature_file}"
            )
    
    logger.info(f"📚 Enriquecimento concluído: {success_count}/{len(data)} itens completos")
    if removed_incomplete > 0:
        logger.warning(f"⚠️  {removed_incomplete} itens removidos (faltando feature ou matched content)")
    if feature_errors > 0:
        logger.warning(f"⚠️  {feature_errors} erros ao ler feature files")
    if matched_errors > 0:
        logger.warning(f"⚠️  {matched_errors} erros ao ler matched files")
    
    return enriched_data


def save_results(data: list, output_file: str):
    """
    Salva os resultados em arquivo JSON.
    
    Args:
        data: Lista de dados a serem salvos
        output_file: Caminho do arquivo de saída
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Resultados salvos em: {output_file}")
    except Exception as e:
        logger.error(f"❌ Erro ao salvar resultados em {output_file}: {e}")
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Heurística para encontrar uso de arquivos .feature com processamento paralelo'
    )
    parser.add_argument(
        '--clone-dir', 
        required=True, 
        help='Diretório onde os repositórios estão clonados'
    )
    parser.add_argument(
        '--data-file', 
        required=True, 
        help='Arquivo JSON com dados dos repositórios'
    )
    parser.add_argument(
        '--output', 
        default='resultados_heuristica.json', 
        help='Arquivo de saída para os resultados (padrão: resultados_heuristica.json)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Ativa modo de debug com logs detalhados'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=None, 
        help='Número de processos paralelos (padrão: número de CPUs - 1)'
    )
    parser.add_argument(
        '--filter-confidence', 
        action='store_true', 
        help='Mantém apenas o used_in com maior confidence'
    )
    parser.add_argument(
        '--filter-metric',
        choices=['mean', 'median', 'mode'],
        default=None,
        help='Filtra resultados baseado em métrica estatística (mean, median, mode)'
    )
    parser.add_argument(
        '--filter-threshold',
        type=float,
        default=None,
        help='Threshold customizado para filtro de confidence (sobrescreve --filter-metric)'
    )
    parser.add_argument(
        '--enrich-contents', 
        action='store_true', 
        help='Adiciona conteúdo dos arquivos feature e matched'
    )
    parser.add_argument(
        '--min-confidence',
        type=float,
        default=0.5,
        help='Confiança mínima para considerar um match (padrão: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Configurar diretórios e arquivos
    CLONE_DIR = Path(args.clone_dir)
    DATA_FILE = args.data_file
    OUTPUT_FILE = args.output
    
    # Criar diretório de clone se não existir
    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Iniciando análise de repositórios...")
    logger.info(f"📂 Clone dir: {CLONE_DIR}")
    logger.info(f"📄 Data file: {DATA_FILE}")
    logger.info(f"💾 Output file: {OUTPUT_FILE}")
    
    # Instanciar e executar o matcher
    matcher = GherkinStepMatcherExploration(
        clone_dir=CLONE_DIR, 
        debug=args.debug, 
        min_confidence=args.min_confidence
    )
    
    # Carregar dados
    if not matcher._load_data(DATA_FILE):
        logger.error("❌ Falha ao carregar dados. Abortando análise.")
        exit(1)
    
    # Executar análise
    logger.info("🔍 Executando análise dos repositórios...")
    analysis_results = matcher.analyze_repositories(DATA_FILE)
    
    if not analysis_results:
        logger.warning("⚠️  Nenhum resultado de análise gerado.")
        exit(0)
    
    logger.info(f"✅ Análise concluída: {len(analysis_results)} resultados gerados")
    
    # Aplicar filtro de confidence se solicitado
    if args.filter_confidence:
        logger.info("🔧 Aplicando filtro de max confidence...")
        analysis_results = filter_max_confidence(analysis_results)
    
    # Aplicar filtro por métrica estatística se solicitado
    if args.filter_metric or args.filter_threshold is not None:
        logger.info("🔧 Aplicando filtro por métrica estatística...")
        analysis_results = filter_by_metric(
            analysis_results, 
            metric=args.filter_metric or "median",
            custom_threshold=args.filter_threshold
        )
    
    # Enriquecer com conteúdo dos arquivos se solicitado
    if args.enrich_contents:
        logger.info("📚 Enriquecendo com conteúdo dos arquivos...")
        analysis_results = enrich_with_file_contents(analysis_results, CLONE_DIR)
    
    # Salvar resultados
    save_results(analysis_results, OUTPUT_FILE)
    
    # Estatísticas finais
    logger.info("\n" + "="*50)
    logger.info("📊 ESTATÍSTICAS FINAIS")
    logger.info("="*50)
    logger.info(f"Total de resultados: {len(analysis_results)}")
    
    if analysis_results:
        with_matches = sum(1 for r in analysis_results if r.get("used_in"))
        logger.info(f"Com matches: {with_matches}")
        logger.info(f"Sem matches: {len(analysis_results) - with_matches}")
        
        if args.enrich_contents:
            # Todos devem ter ambos os conteúdos (feature + matched)
            logger.info(f"Itens completos (feature + matched): {len(analysis_results)}")
    
    logger.info("="*50)
    logger.info("✅ Processamento concluído com sucesso!")