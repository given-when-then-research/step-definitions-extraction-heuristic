# Variáveis comuns
PYTHON      := python3
CLONE_DIR   := ../repositories/
DATA_FILE   := ../dataset/projects_selection.json
OUTPUT_DIR  := ./output     # garanta que este diretório exista

# Heurística exploration
SCRIPT_EXPLORATION   := src/exploration.py
OUTPUT_EXPLORATION   := ./output/exploration.json

# Heurística exploitation
SCRIPT_EXPLOITATION   := src/exploitation.py 
OUTPUT_EXPLOITATION   := ./output/exploitation.json

# Alvo padrão: roda ambas as heurísticas
.PHONY: all
all: run-exploration run-exploitation

# Executa apenas a heurística exploration
.PHONY: run-exploration
run-exploration:
	$(PYTHON) -B $(SCRIPT_EXPLORATION) --clone-dir $(CLONE_DIR) --data-file $(DATA_FILE) --output $(OUTPUT_EXPLORATION)

# Executa apenas a heurística exploitation
.PHONY: run-exploitation
run-exploitation:
	$(PYTHON) -B $(SCRIPT_EXPLOITATION) --clone-dir $(CLONE_DIR) --data-file $(DATA_FILE) --output $(OUTPUT_EXPLOITATION)

# Limpa resultados gerados por ambas as heurísticas
.PHONY: clean
clean:
	rm -f $(OUTPUT_EXPLORATION) $(OUTPUT_EXPLOITATION)