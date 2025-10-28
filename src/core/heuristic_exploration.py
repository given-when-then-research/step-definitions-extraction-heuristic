from adapters.log.logger import logger

import logging
import json

import re
from pathlib import Path
from typing import List, Dict, Set, Any, Tuple

import time
from multiprocessing import Pool, cpu_count

from datetime import datetime
import os

# Cria diretório para logs
os.makedirs("logs", exist_ok=True)

# Nome do arquivo com data e hora
log_filename = datetime.now().strftime("logs/gherkin_%Y%m%d_%H%M%S.log")

# Cria o logger específico
logger = logging.getLogger("gherkin_heuristica_exploration")
logger.setLevel(logging.DEBUG)  # <- captura todos os níveis de log

# Evita adicionar múltiplos handlers em execuções repetidas
if not logger.handlers:
    # Formatação padrão
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Handler para arquivo
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # <- inclui DEBUG
    file_handler.setFormatter(formatter)

    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # <- inclui DEBUG
    console_handler.setFormatter(formatter)

    # Adiciona os handlers ao logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def _normalize_name(name: str) -> str:
    """Normaliza nome removendo caracteres especiais"""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())

class GherkinStepMatcherExploration:
    """
    Classe principal para análise e mapeamento entre arquivos .feature e step definitions.
    Implementa múltiplas heurísticas com sistema de pontuação ponderada e processamento paralelo.
    """

    # Constantes para pesos das heurísticas
    WEIGHT_EXACT_FILENAME_MATCH = 10.0
    WEIGHT_PARTIAL_FILENAME_MATCH = 5.0
    WEIGHT_FEATURE_NAME_IN_CONTENT = 7.0
    WEIGHT_SCENARIO_NAME_IN_CONTENT = 6.0
    WEIGHT_STEP_TEXT_EXACT_MATCH = 9.0
    WEIGHT_STEP_TEXT_REGEX_MATCH = 8.0
    WEIGHT_STEP_ANNOTATION_MATCH = 8.5
    WEIGHT_DOMAIN_KEYWORD_MATCH = 4.0
    WEIGHT_IMPORT_STATEMENT_MATCH = 3.0
    WEIGHT_DIRECTORY_STRUCTURE_MATCH = 0.5
    WEIGHT_EXPLICIT_FEATURE_REFERENCE = 10.0
    WEIGHT_RUNTEST_REFERENCE = 10.0

    # Padrões de expressões regulares para diferentes frameworks
    STEP_ANNOTATION_PATTERNS = {
        'java': [
            r'@(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)'
        ],
        'kotlin': [
            r'@(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)',
            r'(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)\s*\{'
        ],
        'scala': [
            r'@(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)',
            r'(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)',
            r'(?:given|when|then|and|but)\s*\(\s*"(.+?)"\s*\)'
        ],
        'nodejs': [
            # regex literal JS
            r'\b(?:Given|When|Then|And|But)\s*\(\s*/(.+?)/[igmsuy]*\s*,',
            r'\b(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*,',
            r'\b(?:defineStep|step)\s*\(\s*"(.+?)"\s*,',
            r'\bthis\.(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*,'
        ],
        'ruby': [
            r'\b(?:Given|When|Then|And|But)\s*\(\s*/(.+?)/\s*\)',
            r'\b(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)',
            r'\b(?:given|when|then|and|but)\s*\(\s*"(.+?)"\s*\)'
        ],
        'go': [
            r'\bctx\.Step\s*\(\s*"(.+?)"\s*,',  # Godog
            r'\bs\.Step\s*\(\s*"(.+?)"\s*,',
            r'\b(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*,'
        ],
        'python': [
            r'@(?:given|when|then|step)\s*\(\s*"(.+?)"\s*\)',  # behave
            # raw string optional
            r'@(?:given|when|then|step)\s*\(\s*r?"(.+?)"\s*\)'
        ],
        'android': [
            r'@(?:Step|Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)',
            r'(?:onGiven|onWhen|onThen)\s*\(\s*"(.+?)"\s*\)',
            r'(?:given|when|then)\s*\{\s*"(.+?)"\s*\}'
        ],
        'lua': [
            r'\bstep_definitions\.(?:given|when|then)\s*\(\s*"(.+?)"\s*,',
            r'\b(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*,',
            r'\b(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)'
        ],
        'cpp': [
            r'\bGIVEN\s*\(\s*"(.+?)"\s*\)',
            r'\bWHEN\s*\(\s*"(.+?)"\s*\)',
            r'\bTHEN\s*\(\s*"(.+?)"\s*\)',
            r'\bREGEX_PARAM\s*\(\s*"(.+?)"\s*\)',
            r'\b(?:Given|When|Then|And|But)\s*\(\s*"(.+?)"\s*\)'
        ],
        'ocaml': [
            r'\b(?:step|given|when|then)_[a-z]+\s+~text:\s*"(.+?)"',
            r'\blet\s+\w+\s*=\s*(?:step|given|when|then)_[a-z]+\s+~text:\s*"(.+?)"'
        ]
    }

    # Extensões de arquivos para cada linguagem
    LANGUAGE_EXTENSIONS = {
        'java': ['.java'],
        'kotlin': ['.kt', '.kts'],
        'scala': ['.scala'],
        'nodejs': ['.js', '.ts'],
        'ruby': ['.rb'],
        'go': ['.go'],
        'android': ['.java', '.kt'],
        'lua': ['.lua'],
        'cpp': ['.cpp', '.cc', '.cxx', '.c++', '.hpp', '.h', '.hxx', '.h++'],
        'ocaml': ['.ml', '.mli']
    }

    # Extensões de arquivos de step definitions por linguagem
    STEP_FILE_EXTENSIONS = {
        'java': ['Steps.java', 'StepDefinitions.java', 'StepDefs.java', 'Stepdefs.java'],
        'kotlin': ['Steps.kt', 'StepDefinitions.kt', 'StepDefs.kt', 'Stepdefs.kt'],
        'scala': ['Steps.scala', 'StepDefinitions.scala', 'StepDefs.scala'],
        'nodejs': [
            'steps.js', 'stepDefinitions.js', 'step-definitions.js', 'step_definitions.js',
            'steps.ts', 'stepDefinitions.ts', 'step-definitions.ts', 'step_definitions.ts'
        ],
        'ruby': ['_steps.rb', 'steps.rb', 'step_definitions.rb', 'stepdefs.rb'],
        'go': ['_steps.go', 'steps.go', 'step_definitions.go'],
        'android': ['Steps.java', 'StepDefinitions.java', 'Steps.kt', 'StepDefinitions.kt'],
        'lua': ['_steps.lua', 'steps.lua', 'step_definitions.lua'],
        'cpp': ['steps.cpp', 'step_definitions.cpp', 'steps.cc', 'steps.h', 'steps.hpp'],
        'ocaml': ['steps.ml', 'step_definitions.ml', 'steps_binding.ml']
    }

    # Padrões de importação para frameworks BDD por linguagem
    IMPORT_PATTERNS = {
        'java': [
            r'import\s+(?:[a-zA-Z0-9_.]+\.)?cucumber',
            r'import\s+(?:[a-zA-Z0-9_.]+\.)?gherkin',
            r'import\s+io\.cucumber'
        ],
        'kotlin': [
            r'import\s+(?:[a-zA-Z0-9_.]+\.)?cucumber',
            r'import\s+io\.cucumber',
            r'import\s+(?:[a-zA-Z0-9_.]+\.)?gherkin'
        ],
        'scala': [
            r'import\s+(?:[a-zA-Z0-9_.]+\.)?cucumber',
            r'import\s+io\.cucumber'
        ],
        'nodejs': [
            r'(?:import|require)\s+[\'"](?:@cucumber|cucumber)[\'"]',
            r'(?:import|require)\s+[\'"](?:.*?)cucumber(?:.*?)[\'"]',
            r'from\s+[\'"](?:@cucumber|cucumber)[\'"]'
        ],
        'ruby': [
            r'require\s+[\'"]cucumber[\'"]',
            r'require\s+[\'"](?:.*?)cucumber(?:.*?)[\'"]'
        ],
        'go': [
            r'import\s+[\'"]github\.com/cucumber/godog[\'"]',
            r'import\s+[\'"](?:.*?)godog(?:.*?)[\'"]'
        ],
        'android': [
            r'import\s+(?:[a-zA-Z0-9_.]+\.)?cucumber',
            r'import\s+(?:[a-zA-Z0-9_.]+\.)?gherkin',
            r'import\s+(?:[a-zA-Z0-9_.]+\.)?android\.test'
        ],
        'lua': [
            r'require\s+[\'"]cucumber[\'"]',
            r'require\s+[\'"](?:.*?)cucumber(?:.*?)[\'"]'
        ],
        'cpp': [
            r'#include\s+[<\"](?:.*?)cucumber(?:.*?)[>\"]',
            r'#include\s+[<\"](?:.*?)gherkin(?:.*?)[>\"]',
            r'#include\s+[<\"](?:.*?)steps(?:.*?)[>\"]'
        ],
        'ocaml': [
            r'open\s+Cucumber',
            r'open\s+Gherkin',
            r'module\s+Cucumber'
        ]
    }

    # Novos padrões para detectar referências explícitas a arquivos .feature
    EXPLICIT_FEATURE_PATTERNS = [
        # Padrão para runTest e variações
        r'runTest\s*\(\s*[\'"]`([^\'"]`]+\.feature)[\'"]`\)',
        r'runTest\s*\(\s*[\'"]`([^\'"]`]+)[\'"]`\)',

        # Padrões para referências diretas a arquivos .feature
        r'[\'"]`([^\'"]`]+\.feature)[\'"]`',
        r'[\'"]`([^\'"]`/]+/[^\'"]`/]+\.feature)[\'"]`',
        r'[\'"]`((?:\.\.?/)?[^\'"]`]+\.feature)[\'"]`',

        # Padrões para funções que carregam features
        r'loadFeature\s*\(\s*[\'"]`([^\'"]`]+)[\'"]`\)',
        r'parseFeature\s*\(\s*[\'"]`([^\'"]`]+)[\'"]`\)',
        r'readFeature\s*\(\s*[\'"]`([^\'"]`]+)[\'"]`\)',

        # Padrões para configurações
        r'features?\s*:\s*[\'"]`([^\'"]`]+\.feature)[\'"]`',
        r'specs?\s*:\s*[\'"]`([^\'"]`]+\.feature)[\'"]`',
        r'featureFile\s*:\s*[\'"]`([^\'"]`]+)[\'"]`',

        # Padrões para caminhos
        r'path\.join\s*\([^)]*[\'"]`([^\'"]`]+\.feature)[\'"]`\)',
        r'__dirname\s*\+\s*[\'"]`([^\'"]`]+\.feature)[\'"]`',
        r'fs\.readFile\s*\(\s*[\'"]`([^\'"]`]+\.feature)[\'"]`\)',

        # Padrões para arrays
        r'features\s*:\s*\[\s*[\'"]`([^\'"]`]+\.feature)[\'"]`\)',
        r'specs\s*:\s*\[\s*[\'"]`([^\'"]`]+\.feature)[\'"]`\)',

        # Padrões para require/import
        r'require\s*\(\s*[\'"]`([^\'"]`]+\.feature)[\'"]`\s*\)',
        r'import\s+[^\'"]`]+\s+from\s+[\'"]`([^\'"]`]+\.feature)[\'"]`',
    ]

    def __init__(self, clone_dir: Path, min_confidence: float = 0.3, debug: bool = False, num_workers: int = None):
        """
        Inicializa o analisador de correspondência Gherkin.

        Args:
            clone_dir: Diretório base onde os repositórios estão clonados
            min_confidence: Confiança mínima para considerar uma correspondência válida
            debug: Ativa modo de debug com logs detalhados
            num_workers: Número de processos paralelos (None = usar todos os CPUs disponíveis)
        """
        self.clone_dir = clone_dir
        self.min_confidence = min_confidence
        self.debug = debug
        self.num_workers = num_workers if num_workers is not None else max(
            1, cpu_count() - 1)

        if debug:
            logger.setLevel(logging.DEBUG)

        self.feature_cache = {}  # Cache para conteúdo de arquivos .feature
        self.file_cache = {}  # Cache para conteúdo de arquivos

        # Extensões de arquivo a serem analisadas (todas as extensões suportadas)
        self.all_extensions = []
        for exts in self.LANGUAGE_EXTENSIONS.values():
            self.all_extensions.extend(exts)

        # Adiciona extensões de arquivos de configuração
        '''
        self.all_extensions.extend(['.json', '.yml', '.yaml', '.xml', '.conf', '.config', '.properties'])
        '''

        # Remove duplicatas
        self.all_extensions = list(set(self.all_extensions))

    def _get_file_content(self, file_path: Path) -> str:
        """
        Obtém o conteúdo de um arquivo, utilizando cache para evitar leituras repetidas.

        Args:
            file_path: Caminho do arquivo

        Returns:
            Conteúdo do arquivo como string
        """
        path_str = str(file_path)
        if path_str in self.file_cache:
            return self.file_cache[path_str]

        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
                self.file_cache[path_str] = content
                return content
        except Exception as e:
            if self.debug:
                logger.warning(f"Erro ao ler arquivo {file_path}: {e}")
            return ""

    def _extract_gherkin_steps(self, feature_content: str) -> List[str]:
        """
        Extrai os passos Gherkin de um arquivo .feature.

        Args:
            feature_content: Conteúdo do arquivo .feature

        Returns:
            Lista de passos Gherkin (Given/When/Then/And/But)
        """
        # Padrão para capturar passos Gherkin
        step_pattern = re.compile(
            r'^\s*(?:Given|When|Then|And|But)\s+(.+?)$',
            re.MULTILINE | re.IGNORECASE
        )
        return [match.group(1).strip() for match in step_pattern.finditer(feature_content)]

    def _extract_feature_name(self, feature_content: str) -> str:
        """
        Extrai o nome da feature de um arquivo .feature.

        Args:
            feature_content: Conteúdo do arquivo .feature

        Returns:
            Nome da feature
        """
        feature_pattern = re.compile(
            r'^\s*Feature:\s*(.+?)$',
            re.MULTILINE
        )
        match = feature_pattern.search(feature_content)
        return match.group(1).strip() if match else ""

    def _extract_scenario_names(self, feature_content: str) -> List[str]:
        """
        Extrai os nomes dos cenários de um arquivo .feature.

        Args:
            feature_content: Conteúdo do arquivo .feature

        Returns:
            Lista de nomes de cenários
        """
        scenario_pattern = re.compile(
            r'^\s*(?:Scenario|Scenario Outline):\s*(.+?)$',
            re.MULTILINE
        )
        return [match.group(1).strip() for match in scenario_pattern.finditer(feature_content)]

    def _extract_domain_keywords(self, feature_content: str) -> Set[str]:
        """
        Extrai palavras-chave de domínio do conteúdo da feature.

        Args:
            feature_content: Conteúdo do arquivo .feature

        Returns:
            Conjunto de palavras-chave de domínio
        """
        # Extrai feature name, scenario names e steps
        feature_name = self._extract_feature_name(feature_content)
        scenario_names = self._extract_scenario_names(feature_content)
        steps = self._extract_gherkin_steps(feature_content)

        # Combina todos os textos
        all_text = ' '.join([feature_name] + scenario_names + steps)

        # Remove palavras comuns de Gherkin e stopwords
        stopwords = {'given', 'when', 'then', 'and', 'but', 'the', 'a', 'an', 'is', 'are',
                     'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'from'}

        # Tokeniza e filtra palavras significativas (substantivos, verbos, adjetivos)
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_]{2,}\b', all_text.lower())
        return {word for word in words if word not in stopwords}

    def _find_all_files(self, repo_folder: Path) -> List[Path]:
        """
        Encontra todos os arquivos relevantes no repositório.

        Args:
            repo_folder: Pasta do repositório

        Returns:
            Lista de caminhos para arquivos relevantes
        """
        all_files = []

        # Busca por arquivos com extensões relevantes
        for ext in self.all_extensions:
            try:
                pattern = f"**/*{ext}"
                matches = list(repo_folder.glob(pattern))
                all_files.extend(matches)
                if self.debug:
                    logger.debug(
                        f"Encontrados {len(matches)} arquivos com extensão {ext}")
            except Exception as e:
                logger.warning(
                    f"Erro ao buscar arquivos com extensão {ext}: {e}")

        # Remove duplicatas
        all_files = list(set(all_files))

        # Prioriza arquivos de teste e configuração
        test_files = [f for f in all_files if any(term in str(f).lower() for term in
                                                  ['test', 'spec', 'step', 'feature', 'gherkin', 'cucumber', 'config', 'conf'])]

        # Coloca arquivos prioritários no início da lista
        prioritized_files = test_files + \
            [f for f in all_files if f not in test_files]

        return prioritized_files

    def _normalize_path(self, path: str) -> str:
        """
        Normaliza um caminho para comparação.

        Args:
            path: Caminho a ser normalizado

        Returns:
            Caminho normalizado
        """
        # Converte para string se for Path
        if isinstance(path, Path):
            path = str(path)

        # Normaliza separadores de diretório
        path = path.replace('\\', '/')

        # Remove ./ do início
        if path.startswith('./'):
            path = path[2:]

        return path.lower()

    def _get_all_possible_paths(self, feature_path: Path, feature_rel_path: str) -> List[str]:
        """
        Gera todas as possíveis variações de caminho para um arquivo .feature.

        Args:
            feature_path: Caminho absoluto para o arquivo .feature
            feature_rel_path: Caminho relativo do arquivo .feature

        Returns:
            Lista de possíveis caminhos normalizados
        """
        paths = []

        # Nome do arquivo com extensão
        feature_name = feature_path.name
        paths.append(self._normalize_path(feature_name))

        # Nome do arquivo sem extensão
        feature_stem = feature_path.stem
        paths.append(self._normalize_path(feature_stem))

        # Caminho relativo completo
        paths.append(self._normalize_path(feature_rel_path))

        # Caminho relativo sem extensão
        rel_stem = str(Path(feature_rel_path).with_suffix(''))
        paths.append(self._normalize_path(rel_stem))

        # Partes do caminho relativo
        rel_parts = Path(feature_rel_path).parts
        if len(rel_parts) > 1:
            # Último diretório + nome do arquivo
            last_dir_file = str(Path(rel_parts[-2]) / rel_parts[-1])
            paths.append(self._normalize_path(last_dir_file))

            # Apenas o último diretório
            paths.append(self._normalize_path(rel_parts[-2]))

            # Caminho relativo sem o primeiro diretório
            if len(rel_parts) > 2:
                without_first = str(Path(*rel_parts[1:]))
                paths.append(self._normalize_path(without_first))

        # Remove duplicatas
        return list(set(paths))

    def _find_step_files(self, repo_folder: Path, language: str) -> List[Path]:
        """
        Encontra arquivos de step definitions em um repositório.

        Args:
            repo_folder: Pasta do repositório
            language: Linguagem de programação

        Returns:
            Lista de caminhos para arquivos de step definitions
        """
        step_files = []

        # Obtém extensões para a linguagem
        extensions = self.LANGUAGE_EXTENSIONS.get(language, [])
        step_patterns = self.STEP_FILE_EXTENSIONS.get(language, [])

        # Busca por arquivos com extensões da linguagem
        for ext in extensions:
            # Busca recursiva em todas as pastas
            for file_path in repo_folder.glob(f'**/*{ext}'):
                # Verifica se é um arquivo de step definition pelo nome
                file_name = file_path.name
                if any(pattern in file_name.lower() for pattern in ['step', 'steps', 'stepdefinition', 'stepdefs']):
                    step_files.append(file_path)
                else:
                    # Verifica conteúdo para identificar step definitions
                    content = self._get_file_content(file_path)
                    patterns = self.STEP_ANNOTATION_PATTERNS.get(language, [])
                    if any(re.search(pattern, content) for pattern in patterns):
                        step_files.append(file_path)

        return step_files

    def _normalize_gherkin_step(self, step: str) -> str:
        """
        Normaliza um passo Gherkin para comparação (mais robusto).

        Args:
            step: Texto do passo Gherkin

        Returns:
            Passo normalizado
        """
        # Remove espaços extras e converte para minúsculas
        normalized = re.sub(r'\s+', ' ', step.lower().strip())

        # Substitui placeholders comuns por um token genérico <PARAM>
        # Exemplos: "<param>", "{param}", "([^"]*)", "(.*)", etc.
        normalized = re.sub(r'<[^>]+>', '<PARAM>', normalized)
        normalized = re.sub(r'\{[^}]+\}', '<PARAM>', normalized)
        normalized = re.sub(r'"[^"]*"', '<PARAM>',
                            normalized)  # Strings entre aspas
        # Strings entre aspas simples
        normalized = re.sub(r'\'\'[^"]*\'\'', '<PARAM>', normalized)
        # Strings entre crases
        normalized = re.sub(r'`[^`]*`', '<PARAM>', normalized)
        normalized = re.sub(r'\b\d+\b', '<PARAM>',
                            normalized)  # Números inteiros
        normalized = re.sub(r'\b\d+\.\d+\b', '<PARAM>',
                            normalized)  # Números decimais
        # Regex placeholders (tentativa de generalizar)
        normalized = re.sub(r'\(\.\*\??\)', '<PARAM>',
                            normalized)  # (.*?) ou (.*)
        normalized = re.sub(r'\(\[\^\"\]\*\)',
                            '<PARAM>', normalized)  # ([^"]*)
        normalized = re.sub(r'\(\\d\+\)', '<PARAM>', normalized)  # (\d+)

        # Remove pontuação no final (opcional, pode afetar precisão)
        # normalized = re.sub(r'[.,;:!?]$', '', normalized)

        # Normaliza espaços ao redor de <PARAM>
        normalized = re.sub(r'\s*<PARAM>\s*', ' <PARAM> ', normalized).strip()

        return normalized

    def _convert_gherkin_to_regex(self, step: str) -> str:
        """
        Converte um passo Gherkin em uma expressão regular.

        Args:
            step: Texto do passo Gherkin

        Returns:
            Expressão regular correspondente
        """
        # Normaliza o passo primeiro para identificar placeholders
        normalized_step = self._normalize_gherkin_step(step)

        # Escapa caracteres especiais de regex
        regex_str = re.escape(normalized_step)

        # Substitui o token <PARAM> normalizado por um grupo de captura genérico
        regex_str = regex_str.replace(re.escape('<PARAM>'), '(.*?)')

        # Adiciona âncoras de início e fim com flexibilidade para espaços
        return r'^\s*' + regex_str + r'\s*$'

    def _extract_step_patterns_from_file(self, step_file: Path, language: str) -> List[str]:
        """
        Extrai padrões de step definitions de um arquivo.

        Args:
            step_file: Caminho do arquivo de step definition
            language: Linguagem de programação

        Returns:
            Lista de padrões de step extraídos
        """
        content = self._get_file_content(step_file)
        patterns = []

        # Obtém os padrões de regex para a linguagem
        regex_patterns = self.STEP_ANNOTATION_PATTERNS.get(language, [])

        # Extrai todos os padrões de step do arquivo
        for pattern in regex_patterns:
            try:
                matches = re.finditer(pattern, content)
                for match in matches:
                    step_pattern = match.group(1)
                    patterns.append(step_pattern)
            except re.error as e:
                if self.debug:
                    logger.warning(
                        f"Erro de regex ao extrair step pattern '{pattern}' em {step_file}: {e}")

        return patterns

    def _calculate_filename_match_score(self, feature_path: Path, file_path: Path) -> float:
        """
        Calcula a pontuação de correspondência baseada em nomes de arquivos.

        Args:
            feature_path: Caminho do arquivo .feature
            file_path: Caminho do arquivo a ser analisado

        Returns:
            Pontuação de correspondência
        """
        feature_stem = feature_path.stem.lower()
        file_stem = file_path.stem.lower()

        # Remove sufixos comuns de arquivos de step
        for suffix in ['steps', 'stepdefinitions', 'stepdefs', 'step', '_steps', 'test', '_test', 'spec', '_spec']:
            # Usar regex para remover sufixo apenas no final e evitar remover partes do nome
            file_stem = re.sub(f'{suffix}$', '', file_stem)

        # Correspondência exata (ignorando case e sufixos)
        if feature_stem == file_stem:
            return self.WEIGHT_EXACT_FILENAME_MATCH

        # Correspondência parcial (um é substring do outro)
        if feature_stem in file_stem or file_stem in feature_stem:
            # Dar prioridade se a feature for substring do arquivo de step
            if feature_stem in file_stem:
                return self.WEIGHT_PARTIAL_FILENAME_MATCH
            else:
                # Penalizar um pouco se o step for substring da feature
                return self.WEIGHT_PARTIAL_FILENAME_MATCH * 0.7

        return 0.0

    def _check_feature_name_in_content(self, feature_path: Path, file_path: Path) -> float:
        """
        Verifica se o nome da feature aparece no conteúdo do arquivo.

        Args:
            feature_path: Caminho do arquivo .feature
            file_path: Caminho do arquivo a ser analisado

        Returns:
            Pontuação de correspondência
        """
        feature_content = self._get_file_content(feature_path)
        file_content = self._get_file_content(file_path)

        # Extrai o nome da feature
        feature_name = self._extract_feature_name(feature_content)
        if not feature_name:
            return 0.0

        # Verifica se o nome da feature (normalizado) aparece no conteúdo do arquivo (normalizado)
        # Usar word boundaries para evitar matches parciais indesejados
        if re.search(r'\b' + re.escape(feature_name.lower()) + r'\b', file_content.lower()):
            return self.WEIGHT_FEATURE_NAME_IN_CONTENT

        return 0.0

    def _check_scenario_names_in_content(self, feature_path: Path, file_path: Path) -> float:
        """
        Verifica se os nomes dos cenários aparecem no conteúdo do arquivo.

        Args:
            feature_path: Caminho do arquivo .feature
            file_path: Caminho do arquivo a ser analisado

        Returns:
            Pontuação de correspondência
        """
        feature_content = self._get_file_content(feature_path)
        file_content = self._get_file_content(file_path)

        # Extrai os nomes dos cenários
        scenario_names = self._extract_scenario_names(feature_content)
        if not scenario_names:
            return 0.0

        # Verifica se algum nome de cenário (normalizado) aparece no conteúdo do arquivo (normalizado)
        file_content_lower = file_content.lower()
        for scenario_name in scenario_names:
            # Usar word boundaries
            if re.search(r'\b' + re.escape(scenario_name.lower()) + r'\b', file_content_lower):
                return self.WEIGHT_SCENARIO_NAME_IN_CONTENT

        return 0.0

    def _check_step_text_match(self, feature_path: Path, file_path: Path, language: str) -> float:
        """
        Verifica se os passos Gherkin correspondem a step definitions no arquivo .

        Args:
            feature_path: Caminho do arquivo .feature
            file_path: Caminho do arquivo a ser analisado
            language: Linguagem de programação

        Returns:
            Pontuação de correspondência (a maior encontrada entre exata, regex)
        """
        feature_content = self._get_file_content(feature_path)

        # Extrai os passos Gherkin
        gherkin_steps = self._extract_gherkin_steps(feature_content)
        if not gherkin_steps:
            return 0.0

        # Extrai os padrões de step definitions
        step_patterns = self._extract_step_patterns_from_file(
            file_path, language)
        if not step_patterns:
            return 0.0

        max_score = 0.0
        best_match_type = "none"

        # Cache de normalização
        normalized_gherkin_steps = {
            step: self._normalize_gherkin_step(step) for step in gherkin_steps}
        normalized_step_patterns = {pattern: self._normalize_gherkin_step(
            pattern) for pattern in step_patterns}

        # Verifica correspondência exata (após normalização)
        for g_step, norm_g_step in normalized_gherkin_steps.items():
            for s_pattern, norm_s_pattern in normalized_step_patterns.items():
                if norm_g_step == norm_s_pattern:
                    if self.WEIGHT_STEP_TEXT_EXACT_MATCH > max_score:
                        max_score = self.WEIGHT_STEP_TEXT_EXACT_MATCH
                        best_match_type = "exact"
                    break  # Encontrou match exato para este gherkin step
            if best_match_type == "exact":
                continue  # Passa para o próximo gherkin step

        # Se não encontrou exato, verifica correspondência por regex
        if best_match_type != "exact":
            for g_step, norm_g_step in normalized_gherkin_steps.items():
                for s_pattern, norm_s_pattern in normalized_step_patterns.items():
                    try:
                        # Tenta converter o padrão de step definition para regex
                        # Usar o padrão normalizado para regex pode ser mais robusto
                        regex_pattern = self._convert_gherkin_to_regex(
                            s_pattern)
                        if re.fullmatch(regex_pattern, g_step, re.IGNORECASE):
                            if self.WEIGHT_STEP_TEXT_REGEX_MATCH > max_score:
                                max_score = self.WEIGHT_STEP_TEXT_REGEX_MATCH
                                best_match_type = "regex"
                            break  # Encontrou match regex para este gherkin step
                    except re.error:
                        # Ignora padrões de regex inválidos
                        continue
                if best_match_type == "regex":
                    continue

        # Retorna a maior pontuação encontrada
        # Adiciona o tipo de match ao score para debug/análise?
        # return max_score, best_match_type
        return max_score

    def _check_step_annotation_match(self, file_path: Path, language: str) -> float:
        """
        Verifica se o arquivo contém anotações de step definition.

        Args:
            file_path: Caminho do arquivo a ser analisado
            language: Linguagem de programação

        Returns:
            Pontuação de correspondência
        """
        content = self._get_file_content(file_path)

        # Obtém os padrões de regex para a linguagem
        regex_patterns = self.STEP_ANNOTATION_PATTERNS.get(language, [])

        # Verifica se algum padrão de anotação está presente no conteúdo
        for pattern in regex_patterns:
            try:
                if re.search(pattern, content):
                    return self.WEIGHT_STEP_ANNOTATION_MATCH
            except re.error:
                continue  # Ignora regex inválido

        return 0.0

    def _check_domain_keyword_match(self, feature_path: Path, file_path: Path) -> float:
        """
        Verifica se palavras-chave de domínio da feature aparecem no arquivo.

        Args:
            feature_path: Caminho do arquivo .feature
            file_path: Caminho do arquivo a ser analisado

        Returns:
            Pontuação de correspondência
        """
        feature_content = self._get_file_content(feature_path)
        file_content = self._get_file_content(file_path)

        # Extrai palavras-chave de domínio
        domain_keywords = self._extract_domain_keywords(feature_content)
        if not domain_keywords:
            return 0.0

        # Conta quantas palavras-chave aparecem no conteúdo do arquivo
        file_content_lower = file_content.lower()
        matches = sum(
            1 for keyword in domain_keywords if keyword in file_content_lower)

        # Calcula a pontuação proporcional ao número de matches
        if matches > 0:
            # Normaliza pela raiz quadrada do número de keywords para diminuir o impacto de features muito grandes
            match_ratio = min(1.0, matches / (len(domain_keywords)**0.5))
            return self.WEIGHT_DOMAIN_KEYWORD_MATCH * match_ratio

        return 0.0

    def _check_import_statement_match(self, file_path: Path, language: str) -> float:
        """
        Verifica se o arquivo contém importações de frameworks BDD.

        Args:
            file_path: Caminho do arquivo a ser analisado
            language: Linguagem de programação

        Returns:
            Pontuação de correspondência
        """
        content = self._get_file_content(file_path)

        # Obtém os padrões de importação para a linguagem
        import_patterns = self.IMPORT_PATTERNS.get(language, [])

        # Verifica se algum padrão de importação está presente no conteúdo
        for pattern in import_patterns:
            try:
                if re.search(pattern, content):
                    return self.WEIGHT_IMPORT_STATEMENT_MATCH
            except re.error:
                continue  # Ignora regex inválido

        return 0.0

    def _check_directory_structure_match(self, feature_path: Path, file_path: Path) -> float:
        """
        Verifica se há correspondência na estrutura de diretórios.

        Args:
            feature_path: Caminho do arquivo .feature
            file_path: Caminho do arquivo a ser analisado

        Returns:
            Pontuação de correspondência
        """
        # Obtém os diretórios pai
        feature_dir = feature_path.parent
        file_dir = file_path.parent

        # Verifica se os diretórios são iguais ou relacionados
        if feature_dir == file_dir:
            return self.WEIGHT_DIRECTORY_STRUCTURE_MATCH

        # Verifica se um diretório é subdiretório do outro (até 2 níveis)
        try:
            if feature_dir == file_dir.parent or feature_dir == file_dir.parent.parent:
                # Step definition dentro de subpasta da feature
                return self.WEIGHT_DIRECTORY_STRUCTURE_MATCH * 0.9
            if file_dir == feature_dir.parent or file_dir == feature_dir.parent.parent:
                # Feature dentro de subpasta do step definition
                return self.WEIGHT_DIRECTORY_STRUCTURE_MATCH * 0.8
        except AttributeError:  # Caso chegue na raiz
            pass
        except Exception as e:
            if self.debug:
                logger.warning(f"Erro ao checar parentesco de diretórios: {e}")

        # Verifica se há partes comuns nos caminhos (excluindo nomes de arquivos)
        feature_parts = set(p.lower() for p in feature_dir.parts)
        file_parts = set(p.lower() for p in file_dir.parts)
        common_parts = feature_parts.intersection(file_parts)

        # Calcula score baseado na proporção de partes comuns
        if len(common_parts) > 1:  # Mais de uma parte em comum (ignora a raiz)
            # Média da proporção de partes comuns em relação a cada caminho
            ratio1 = len(common_parts) / \
                len(feature_parts) if len(feature_parts) > 0 else 0
            ratio2 = len(common_parts) / \
                len(file_parts) if len(file_parts) > 0 else 0
            avg_ratio = (ratio1 + ratio2) / 2
            # Multiplica por 0.7 para dar menos peso que subdiretório direto
            return self.WEIGHT_DIRECTORY_STRUCTURE_MATCH * avg_ratio * 0.7

        return 0.0

    def _check_explicit_feature_reference(self, feature_path: Path, file_path: Path, feature_rel_path: str) -> Tuple[float, str]:
        """
        Verifica se há referências explícitas ao arquivo .feature

        Args:
            feature_path: Caminho do arquivo .feature
            file_path: Caminho do arquivo a ser analisado
            feature_rel_path: Caminho relativo do arquivo .feature

        Returns:
            Tupla (pontuação, tipo de referência)
        """
        content = self._get_file_content(file_path)
        max_score = 0.0
        best_type = ""

        # Gera todas as possíveis variações de caminho para o feature
        possible_feature_paths = self._get_all_possible_paths(
            feature_path, feature_rel_path)

        # Função auxiliar para verificar match (exato, parcial)
        def check_match(ref_path_str: str, weight: float, match_type: str) -> Tuple[float, str]:
            nonlocal max_score, best_type
            current_max_score = 0.0
            current_best_type = ""

            norm_ref_path = self._normalize_path(ref_path_str)

            for feat_path in possible_feature_paths:
                # 1. Match Exato
                if feat_path == norm_ref_path:
                    if weight > current_max_score:
                        current_max_score = weight
                        current_best_type = match_type
                    break  # Prioridade máxima

                # 2. Match Parcial (Feature path contido na referência ou vice-versa)
                # Ser mais estrito: exigir que termine com o nome/caminho da feature
                if norm_ref_path.endswith(feat_path) or feat_path.endswith(norm_ref_path):
                    partial_score = weight * 0.8  # Pontuação ligeiramente menor que exato
                    if partial_score > current_max_score:
                        current_max_score = partial_score
                        current_best_type = f"partial_{match_type}"
                    continue  # Continua buscando exato

            # Atualiza o score máximo global se este match for melhor
            if current_max_score > max_score:
                max_score = current_max_score
                best_type = current_best_type

            return current_max_score, current_best_type

        # Verifica padrões de runTest
        for pattern in self.EXPLICIT_FEATURE_PATTERNS[:2]:
            try:
                matches = re.finditer(pattern, content)
                for match in matches:
                    ref_path = match.group(1)
                    check_match(ref_path, self.WEIGHT_RUNTEST_REFERENCE, "runtest_reference")
            except re.error:
                continue

        # Verifica outros padrões de referência explícita
        for pattern in self.EXPLICIT_FEATURE_PATTERNS[2:]:
            try:
                matches = re.finditer(pattern, content)
                for match in matches:
                    ref_path = match.group(1)
                    check_match(ref_path, self.WEIGHT_EXPLICIT_FEATURE_REFERENCE, "explicit_feature_reference")
            except re.error:
                continue

        # Verifica se o nome do arquivo .feature aparece explicitamente no conteúdo
        # (Mantido como estava, menos propenso a falso positivo)
        feature_name_lower = feature_path.name.lower()
        if feature_name_lower in content.lower():
            lines_with_feature = [line for line in content.lower().split(
                '\n') if feature_name_lower in line]
            for line in lines_with_feature:
                # Verifica se a linha parece uma referência real a um arquivo
                if any(term in line for term in ['require', 'import', 'include', 'load', 'read', 'test', 'feature', 'path', 'file', 'spec', 'run', 'open']):
                    score = self.WEIGHT_EXPLICIT_FEATURE_REFERENCE * 0.7  # Peso menor que ref direta
                    if score > max_score:
                        max_score = score
                        best_type = "feature_name_reference"
                    break  # Encontrou uma linha suspeita

        return max_score, best_type
    
    def _calculate_step_coverage_ratio(self, feature_path: Path, file_path: Path, language: str) -> float:
        """
        Calcula a razão de cobertura dos steps definidos na feature em relação às implementações no arquivo.

        Args:
            feature_path: Caminho para o arquivo .feature
            file_path: Caminho do arquivo de step definition
            language: Linguagem de programação

        Returns:
            Razão de steps implementados (0.0 a 1.0)
        """
        feature_content = self._get_file_content(feature_path)
        gherkin_steps = self._extract_gherkin_steps(feature_content)

        if not gherkin_steps:
            return 0.0

        step_patterns = self._extract_step_patterns_from_file(file_path, language)
        if not step_patterns:
            return 0.0

        normalized_gherkin_steps = [self._normalize_gherkin_step(s) for s in gherkin_steps]
        normalized_step_patterns = [self._normalize_gherkin_step(s) for s in step_patterns]

        implemented = 0
        for g_step in normalized_gherkin_steps:
            match_found = False
            for s_pattern in normalized_step_patterns:
                if g_step == s_pattern:
                    match_found = True
                    break
                else:
                    try:
                        regex_pattern = self._convert_gherkin_to_regex(s_pattern)
                        if re.fullmatch(regex_pattern, g_step, re.IGNORECASE):
                            match_found = True
                            break
                    except re.error:
                        continue
            if match_found:
                implemented += 1

        return implemented / len(gherkin_steps)

    def analyze_feature_usage(self, repo_folder: Path, feature_rel_path: str, language: str) -> List[Dict[str, Any]]:
        """
        Analisa onde um arquivo .feature está sendo utilizado.

        Args:
            repo_folder: Pasta do repositório
            feature_rel_path: Caminho relativo do arquivo .feature
            language: Linguagem de programação

        Returns:
            Lista de resultados com informações sobre o uso do arquivo .feature
        """
        feature_path = repo_folder / feature_rel_path

        # Verifica se o arquivo .feature existe
        if not feature_path.exists():
            logger.warning(f"Arquivo .feature não encontrado: {feature_path}")
            return []

        # Encontra todos os arquivos relevantes no repositório
        all_files = self._find_all_files(repo_folder)

        # Encontra arquivos de step definitions (potenciais candidatos)
        # step_files = self._find_step_files(repo_folder, language) # Pode ser redundante se all_files já prioriza

        matches = []
        analyzed_files = set()

        # Analisa todos os arquivos relevantes
        for file_path in all_files:
            # Pula o próprio arquivo .feature e arquivos já analisados
            if file_path == feature_path or file_path in analyzed_files:
                continue
            analyzed_files.add(file_path)

            # Calcula pontuações para diferentes heurísticas
            scores = {}

            # 1. Verifica referências explícitas (prioridade máxima)
            explicit_score, explicit_type = self._check_explicit_feature_reference(
                feature_path, file_path, feature_rel_path)
            if explicit_score > 0:
                scores[explicit_type] = explicit_score

            # 2. Verifica correspondência de nome de arquivo
            filename_score = self._calculate_filename_match_score(
                feature_path, file_path)
            if filename_score > 0:
                # Adiciona tipo de match no score (exato, parcial)
                if filename_score == self.WEIGHT_EXACT_FILENAME_MATCH:
                    scores["exact_filename_match"] = filename_score
                elif filename_score >= self.WEIGHT_PARTIAL_FILENAME_MATCH * 0.7:
                    scores["partial_filename_match"] = filename_score

            # 3. Verifica correspondência de texto de steps (apenas se for arquivo da linguagem)
            file_ext = file_path.suffix
            if file_ext in self.LANGUAGE_EXTENSIONS.get(language, []):
                step_text_score = self._check_step_text_match(
                    feature_path, file_path, language)
                if step_text_score > 0:
                    # Adiciona tipo de match no score (exato, regex)
                    if step_text_score == self.WEIGHT_STEP_TEXT_EXACT_MATCH:
                        scores["exact_step_text_match"] = step_text_score
                    elif step_text_score == self.WEIGHT_STEP_TEXT_REGEX_MATCH:
                        scores["regex_step_text_match"] = step_text_score

                # 4. Verifica presença de anotações de step
                step_annotation_score = self._check_step_annotation_match(
                    file_path, language)
                if step_annotation_score > 0:
                    scores["step_annotation_match"] = step_annotation_score

                # 5. Verifica importações BDD
                import_score = self._check_import_statement_match(
                    file_path, language)
                if import_score > 0:
                    scores["import_statement_match"] = import_score

            # 6. Verifica nome da feature no conteúdo
            feature_name_score = self._check_feature_name_in_content(
                feature_path, file_path)
            if feature_name_score > 0:
                scores["feature_name_in_content"] = feature_name_score

            # 7. Verifica nomes de cenários no conteúdo
            scenario_name_score = self._check_scenario_names_in_content(
                feature_path, file_path)
            if scenario_name_score > 0:
                scores["scenario_name_in_content"] = scenario_name_score

            # 8. Verifica palavras-chave de domínio
            domain_keyword_score = self._check_domain_keyword_match(
                feature_path, file_path)
            if domain_keyword_score > 0:
                scores["domain_keyword_match"] = domain_keyword_score

            # 9. Verifica estrutura de diretórios
            directory_score = self._check_directory_structure_match(
                feature_path, file_path)
            if directory_score > 0:
                scores["directory_structure_match"] = directory_score
                
                
            # 10. Verifica cobertura de steps
            coverage_ratio = self._calculate_step_coverage_ratio(feature_path, file_path, language)
            if coverage_ratio >= 0.8:
                scores["step_coverage_full"] = 10.0  # cobertura >= 80%
            elif coverage_ratio >= 0.5:
                scores["step_coverage_partial"] = 5.0  # cobertura intermediária
            elif coverage_ratio > 0:
                scores["step_coverage_low"] = 2.0  # cobertura mínima detectada

            # Calcula a pontuação total
            # Usar soma ponderada pode ser melhor, mas soma simples é mais fácil de começar
            total_score = sum(scores.values())

            # Aplica bônus se múltiplas heurísticas fortes combinarem?
            # Ex: Se tiver step_match E filename_match
            strong_heuristics_count = sum(1 for k, v in scores.items() if k in [
                "exact_filename_match", "partial_filename_match",
                "exact_step_text_match", "regex_step_text_match",
                "runtest_reference", "explicit_feature_reference", "config_reference",
                "partial_runtest_reference", "partial_explicit_feature_reference"
            ])

            if strong_heuristics_count >= 2:
                total_score *= 1.1  # Bônus de 10% por combinação forte

            # Se a pontuação total for maior que o mínimo de confiança, adiciona como match
            if total_score >= self.min_confidence:
                # Tenta obter o caminho relativo para melhor visualização
                try:
                    file_rel_path = str(file_path.relative_to(repo_folder))
                except ValueError:
                    file_rel_path = str(file_path)

                matches.append({
                    "file": file_rel_path,
                    "confidence": round(total_score, 2),
                    "reasons": {k: round(v, 2) for k, v in scores.items()}
                })

        # Ordena os matches por pontuação (decrescente)
        matches.sort(key=lambda x: x["confidence"], reverse=True)

        return matches

    def _process_feature(self, args):
        """
        Processa um único arquivo .feature (usado para paralelismo).

        Args:
            args: Tupla (repo_name, language, repo_folder, feature_rel_path)

        Returns:
            Resultado da análise para o arquivo .feature
        """
        repo_name, language, repo_folder, feature_rel_path = args
        feature_path = repo_folder / feature_rel_path

        # Se o arquivo .feature não existir no sistema de arquivos, pula
        if not feature_path.exists():
            logger.warning(
                f"⚠️ Arquivo .feature não encontrado: {feature_path}")
            return None

        logger.info(
            f"📄 Analisando: {feature_rel_path} em {repo_name} ({language})")

        # Analisa onde o arquivo .feature está sendo utilizado
        matches = self.analyze_feature_usage(
            repo_folder, feature_rel_path, language)

        # Cria o resultado
        result = {
            "repository": repo_name,
            "language": language,
            "feature_file": feature_rel_path,  # Usar relativo para consistência
            "used_in": matches
        }

        if not matches:
            logger.info(f"⚪ Nenhum match encontrado para: {feature_rel_path}")
        else:
            logger.info(
                f"✅ Matches encontrados para {feature_rel_path}: {len(matches)}")
            if self.debug:
                # Mostra até 3 matches
                for i, match in enumerate(matches[:3], 1):
                    logger.debug(
                        f"  Match {i}: {match['file']} (score: {match['confidence']}) Reasons: {match['reasons']}")

        return result
    


    def analyze_repositories(self, data_file: str):
        """
        Analisa o uso de arquivos .feature em múltiplos repositórios definidos em um arquivo JSON.
        Utiliza processamento paralelo para acelerar a análise.

        Args:
            data_file: Caminho para o arquivo JSON contendo a estrutura dos repositórios.

        Returns:
            list: Lista de resultados com informações sobre o uso de cada arquivo .feature.
        """
        self.data = self._load_data(data_file)
        if not self.data:
            return []

        results = []
        tasks = []

        # Prepara as tarefas para processamento paralelo
        for language, repos in self.data.items():
            logger.info(f"\n🔎 Preparando tarefas para Linguagem: {language}")
            for repo in repos:
                repo_name = repo.get("name", "NomeNaoDefinido")
                feature_files = repo.get("feature_files", [])
                if not feature_files:
                    logger.warning(
                        f"⚠️ Repositório {repo_name} não possui 'feature_files' definidos.")
                    continue

                logger.info(
                    f"📦 Repositório: {repo_name} ({len(feature_files)} features)")

                # Caminho do repositório clonado localmente
                # Assume uma estrutura como clone_dir/language/repo_name_flat
                repo_folder_name = _normalize_name(repo_name).replace("/", "_")
                repo_folder = self.clone_dir / language / repo_folder_name

                # Verifica se o diretório do repositório existe
                if not repo_folder.is_dir():
                    logger.warning(
                        f"⚠️ Diretório do repositório não encontrado ou não é um diretório: {repo_folder}")
                    continue

                # Adiciona tarefas para cada arquivo .feature
                for feature_rel_path in feature_files:
                    # Garante que feature_rel_path seja relativo ao repo_folder
                    # Se o path no JSON for absoluto ou começar com '/', tenta ajustar
                    if Path(feature_rel_path).is_absolute() or feature_rel_path.startswith('/'):
                        try:
                            feature_rel_path = str(
                                Path(feature_rel_path).relative_to(repo_folder))
                        except ValueError:
                            logger.warning(
                                f"Não foi possível tornar relativo o path {feature_rel_path} para {repo_folder}")
                            # Tenta usar apenas o nome do arquivo como fallback
                            feature_rel_path = Path(feature_rel_path).name

                    tasks.append(
                        (repo_name, language, repo_folder, feature_rel_path))

        if not tasks:
            logger.error(
                "Nenhuma tarefa de análise foi criada. Verifique o arquivo de dados e a estrutura dos diretórios clonados.")
            return []

        # Processa as tarefas em paralelo
        start_time = time.time()
        logger.info(
            f"\n🚀 Iniciando processamento paralelo de {len(tasks)} features com {self.num_workers} workers...")

        processed_results = []
        with Pool(processes=self.num_workers) as pool:
            # Usar imap_unordered para obter resultados assim que estiverem prontos
            for result in pool.imap_unordered(self._process_feature, tasks):
                if result:
                    processed_results.append(result)
                    # Log de progresso
                    logger.info(
                        f"📊 Progresso: {len(processed_results)} / {len(tasks)} features processadas.")

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(
            f"\n⏱️ Tempo total de processamento: {total_time:.2f} segundos ({total_time / len(tasks):.2f} seg/feature em média)")

        # Ordenar resultados finais por repositório e feature file
        processed_results.sort(key=lambda x: (
            x['repository'], x['feature_file']))

        return processed_results

    def _load_data(self, data_file):
        """
        Carrega os dados do arquivo JSON.

        Args:
            data_file (str): Caminho para o arquivo JSON

        Returns:
            dict: Dados carregados do arquivo JSON
        """
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(
                f"Erro: Arquivo de dados não encontrado em '{data_file}'")
            return {}
        except json.JSONDecodeError as e:
            logger.error(
                f"Erro ao decodificar JSON do arquivo '{data_file}': {e}")
            return {}
        except Exception as e:
            logger.error(
                f"Erro inesperado ao carregar arquivo de dados '{data_file}': {e}")
            return {}
