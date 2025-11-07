# Step Definitions Extraction Heuristic

**Exploration and Exploitation Phases**

This repository contains the scripts and heuristics for extracting **BDD step definitions** from repositories selected in the previous stage of projects selection.
It includes two complementary heuristics — **Exploration** and **Exploitation** — which analyze the repositories to identify and enrich step definition mappings used in Behavior-Driven Development (BDD) projects.

---

##  Installation

Clone this repository and create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Then, install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration and Execution

Execution is managed through the provided `Makefile`.
Before running the commands, make sure to **update the path of the dataset** — this file should correspond to the **output of the Step 2 ("Projects Selection")** phase.
For example:

```makefile
DATA_FILE := ../dataset/projects_selection.json
```

If your dataset is stored in another directory or has a different name, adjust the path accordingly.

---

## Running the Heuristics

To execute both heuristics (exploration and exploitation):

```bash
make
```

You can also run each heuristic separately:

### **Exploration Heuristic**

```bash
make run-exploration
```

This mode explores possible step definitions across repositories, filtering and enriching results using metrics like *mean confidence*.

### **Exploitation Heuristic**

```bash
make run-exploitation
```

This mode focuses on high-confidence matches, simplifying and refining the extracted step definitions to maximize precision.

---

## Output

After execution, two datasets will be generated in the `./output` directory:

| File                | Description                                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------------------- |
| `exploration.json`  | Results produced by the **exploration heuristic**, containing broader step mappings with confidence filtering. |
| `exploitation.json` | Results produced by the **exploitation heuristic**, containing only the most confident and simplified matches. |

Make sure the `output` directory exists before running the commands.

---