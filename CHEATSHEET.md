## Cheatsheet Flower

### Creazione e Setup del Progetto

* **`flwr new`**
    Crea un nuovo progetto Flower interattivo partendo da un template (es. PyTorch, scikit-learn, MLX).

    ```bash
    flwr new
    # Oppure specifica direttamente un template per bypassare il menu:
    flwr new @flwrlabs/quickstart-pytorch
    ```

* **`flwr config list`**
    Mostra i profili di connessione (SuperLink) disponibili sul tuo sistema e il percorso in cui è salvato il file di configurazione (`config.toml`).

    ```bash
    flwr config list
    ```

### Configurazione Simulation Runtime

- **`flwr federation simulation-config`**
    Imposta in modo permanente la configurazione di default della Simulation Runtime per il SuperLink locale.

    ```bash
    flwr federation simulation-config --num-supernodes 8
    ```

    Puoi anche configurare le risorse assegnate a ogni `ClientApp`:

    ```bash
    flwr federation simulation-config \
        --num-supernodes 100 \
        --client-resources-num-cpus 4 \
        --client-resources-num-gpus 0.25
    ```

- **`flwr federation simulation-config --help`**
    Mostra tutte le opzioni configurabili per la Simulation Runtime.

    ```bash
    flwr federation simulation-config --help
    ```

### Esecuzione (Run)

Il comando `run` deve essere sempre eseguito dalla cartella in cui risiede il tuo codice sorgente (nello specifico, dove si trova il file `pyproject.toml`).

- **`flwr run .`**
    Avvia la tua Flower App nella directory corrente (`.`) utilizzando la connessione di default (solitamente la simulazione locale). Invia il task al motore, restituisce un **RUN_ID** e termina, lasciando il processo in esecuzione in background.

    ```bash
    flwr run .
    ```

- **`flwr run . <superlink_name>`**
    Avvia l'app usando una specifica connessione SuperLink definita nella Flower config globale (`~/.flwr/config.toml`). Se ometti questo argomento, Flower usa la connessione di default.

    ```bash
    flwr run . local-simulation
    ```

- **`flwr run . --stream`**
    Avvia l'app e **mantiene il terminale in ascolto**, stampando i log del ServerApp in tempo reale. Altamente consigliato durante lo sviluppo.

    ```bash
    flwr run . --stream
    ```

- **Sovrascrittura delle configurazioni**
    Se vuoi passare parametri al volo (che sovrascrivono quelli definiti nel `pyproject.toml`):

    ```bash
    flwr run . --run-config "learning-rate=0.01"
    ```

- **Override della Simulation Runtime per una sola run**
    Se vuoi cambiare temporaneamente numero di SuperNodes o risorse dei client, usa `--federation-config`.

    ```bash
    flwr run . --stream --federation-config="num-supernodes=12"
    ```

    Puoi passare più opzioni nella stessa stringa:

    ```bash
    flwr run . --stream --federation-config="num-supernodes=256 client-resources-num-cpus=1"
    ```

### Monitoraggio (Status & Logs)

Ogni esecuzione genera un `RUN_ID` univoco. Usa i seguenti comandi per controllare lo stato o i log se hai avviato un processo in background.

- **`flwr ls`**
    Elenca tutti i run (passati e presenti) sulla federazione di default e il loro stato attuale (es. *pending, starting, running, finished*).

    ```bash
    flwr ls
    ```

- **`flwr log <run_id>`**
    Mostra i log di un run specifico. Di default, la CLI si mette in "stream" e ti mostra i log in tempo reale fino a quando il run non finisce.

    ```bash
    flwr log 12345678
    ```

- **`flwr log <run_id> --show`**
    Stampa l'intero storico dei log del run tutto in una volta ed esce (senza rimanere in streaming).

    ```bash
    flwr log 12345678 --show
    ```

### Gestione e Interruzione

- **`flwr stop <run_id>`**
    Interrompe in modo sicuro un run attualmente in corso, inviando una richiesta di arresto al SuperLink (il nodo centrale).

    ```bash
    flwr stop 12345678
    ```
