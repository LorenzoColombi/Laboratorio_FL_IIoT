# Laboratorio di Federated Learning per IIoT

README per il laboratorio di Federated Learning per IIoT del corso di Industrial Internet of Things (Università di Ferrara) a.a. 2025-2026. Questo README contiene 2 esercizi. Il primo è un quickstart molto guidato con Flower, PyTorch e CIFAR-10; il secondo applica la stessa struttura al progetto `fed-phish-guard` per il rilevamento federato di URL di phishing.

Entrambi gli esercizi mostrano la divisione tra `ServerApp`, `ClientApp`, configurazione dell'app e configurazione globale Flower.

## Esercizio 1: quickstart con Flower + PyTorch

L'obiettivo di questo laboratorio è costruire una piccola applicazione di Federated Learning con:

- Flower Framework
- Flower Datasets
- PyTorch
- CIFAR-10

### Obiettivo

Alla fine avrai un sistema federato per il riconoscimento delle immagini composto da:

- un `ServerApp` che coordina l'addestramento
- un `ClientApp` eseguito su più nodi o partizioni
- una strategia `FedAvg` per aggregare pesi e metriche

### 1. Preparazione del progetto

#### Clonazione della repository

```bash
git clone <url-della-repository>
cd Laboratorio_FL_IIoT
```

#### Preparazione ambiente

Creazione ambiente virtuale:

```bash
python -m venv flwr-env
source flwr-env/bin/activate
```

#### Installazione FLower

```bash
pip install -U "flwr[simulation]"

```

#### Installazione dipendenze progetto

Il codice Flower/PyTorch usato nel laboratorio si trova nella sottocartella `quickstart-pytorch`. Subito dopo bisogna installare le dipendenze specifiche del progetto:

```bash
cd quickstart-pytorch
pip install -e .
```

Struttura del progetto:

```text
Laboratorio_FL_IIoT/
├── README.md
├── flwr-env/
└── quickstart-pytorch/
    ├── pyproject.toml
    └── pytorchexample/
        ├── __init__.py
        ├── client_app.py
        ├── server_app.py
        └── task.py
```

### 2. Dataset CIFAR-10 e partizionamento

Per simulare un contesto cross-silo, CIFAR-10 viene diviso in partizioni, una per client.
Con `flwr-datasets` è possibile:

- definire un partizionatore IID, ad esempio con 10 partizioni
- caricare la partizione assegnata al client corrente
- dividere localmente i dati in train e validation
- applicare trasformazioni PyTorch
- costruire i `DataLoader` locali

La funzione `load_data(partition_id, num_partitions, batch_size)`, definita in `task.py`, segue questa logica:

1. inizializza una sola volta `FederatedDataset`
2. carica la partizione assegnata al nodo
3. esegue uno split locale train/test, ad esempio 80/20
4. applica `ToTensor()` e `Normalize(...)`
5. restituisce `trainloader` e `valloader`

Codice completo della funzione `load_data`:

```python
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader
```

### 3. Modello e training locale

Nel file `task.py` si definiscono:

- una CNN semplice per CIFAR-10, `Net`
- `train(...)` per l'addestramento locale
- `test(...)` per la validazione locale

Codice completo del modello:

```python
class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### 4. Concetti base di Flower e configurazione: `Message` e `Record`

In Flower, l'intero scambio di informazioni tra server e client avviene tramite l'oggetto `Message`.
Ogni messaggio trasporta al suo interno un `RecordDict`, una struttura dati flessibile progettata per gestire il payload del training federato.

All'interno del `RecordDict`, i dati sono categorizzati in tre tipologie principali di record:

- **`ArrayRecord`**: gestisce tensori e array n-dimensionali; è il record usato per scambiare e aggiornare i pesi del modello
- **`MetricRecord`**: raccoglie le metriche scalari di valutazione, ad esempio *loss*, *accuracy* e *num-examples*
- **`ConfigRecord`**: contiene i parametri di configurazione che il server invia ai client, ad esempio *learning rate*, *batch size* e numero di epoche locali

Questo schema standardizza lo scambio dei dati durante i round federati.

I parametri di configurazione dell'app sono gestiti in modo centralizzato nel file `pyproject.toml`. La sezione `[tool.flwr.app.config]` contiene i parametri letti dal codice dell'app, come numero di round, epoche locali, learning rate e batch size. Le impostazioni di connessione e simulazione Flower, come il numero di client simulati, sono invece nella Flower config globale (`~/.flwr/config.toml`).

1. **Valori di default dell'app (`pyproject.toml`):** I parametri usati da `context.run_config` sono dichiarati nella sezione `[tool.flwr.app.config]`.
2. **Configurazione della simulazione locale:** Il numero di client simulati si imposta con `options.num-supernodes` nella connessione `local-simulation` della Flower config globale.
3. **Sovrascrittura da terminale:** Quando avvii un esperimento, puoi sovrascrivere i default dell'app dinamicamente da riga di comando usando il flag `--run-config`, senza dover modificare il codice sorgente.

```bash
   flwr run . --run-config "num-server-rounds=20 batch-size=64"
```

## 5. Client federato (`client_app.py`)

Nel `ClientApp` si implementano in genere due entrypoint:

- `@app.train()`: riceve i pesi globali, esegue il training locale e restituisce pesi aggiornati più metriche
- `@app.evaluate()`: valuta il modello ricevuto e restituisce le metriche di valutazione

Codice della funzione di training:

```python
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    trainloader, _ = load_data(partition_id, num_partitions, batch_size)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": round(train_loss, 4),
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
```

Codice della funzione di evaluation:

```python
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": round(eval_loss, 4),
        "eval_acc": round(eval_acc * 100, 2),
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
```

## 6. Server federato (`server_app.py`)

Nel `ServerApp`:

1. leggi i parametri da `context.run_config`
2. inizializzi il modello globale, `initial_arrays`
3. configuri la strategia, `FedAvg`
4. avvii `strategy.start(...)`
5. salvi il modello finale su disco

Configurazione usata in questo progetto:

- `options.num-supernodes`: numero di client simulati dalla connessione Flower `local-simulation`, configurata nella Flower config globale
- `num-server-rounds`: numero di round federati da eseguire
- `fraction-evaluate`: frazione di client da coinvolgere in ogni round di evaluation
- `local-epochs`: numero di epoche locali per ogni client selezionato
- `learning-rate`: learning rate da inviare ai client per il training locale
- `batch-size`: dimensione dei batch locali

Codice completo del metodo principale del `ServerApp`:

```python
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = TrackingFedAvg(
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=1,
        min_evaluate_nodes=1 if fraction_evaluate > 0 else 0,
        min_available_nodes=max(1, len(list(grid.get_node_ids()))),
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    save_metric_plots(strategy, result)
    print("\nSaved plots to plots/")

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")

```

## 7. Avvio della simulazione

Una volta attivato l'ambiente e raggiunta la cartella `quickstart-pytorch`, esegui:

```bash
flwr run . --stream
```

Vedrai i round federati con campionamento client, aggregazione del training, aggregazione della evaluation e metriche aggregate.

Per modificare rapidamente la configurazione a runtime:

```bash
flwr run . --stream --run-config "num-server-rounds=5 local-epochs=3"
```

Altrimenti modifica `pyproject.toml`. Per cambiare il numero di client simulati, aggiorna `options.num-supernodes` nella Flower config globale (`~/.flwr/config.toml`), sotto la connessione `local-simulation`.

## 8. Configurazione globale Flower

Flower salva le configurazioni globali delle connessioni SuperLink nel file `~/.flwr/config.toml`. Queste impostazioni non fanno parte del progetto e valgono per l'ambiente dell'utente che esegue il comando.

Per vedere dove si trova il file e quali connessioni sono disponibili:

```bash
flwr config list
```

Nel nostro caso la connessione di default è `local-simulation`, configurata così:

```toml
[superlink]
default = "local-simulation"

[superlink.local-simulation]
address = ":local:"
options.num-supernodes = 8
```

Per cambiare il numero di client simulati, modifica il valore:

```toml
options.num-supernodes = 8
```

Ad esempio, per simulare 12 client:

```toml
options.num-supernodes = 12
```

Dopo la modifica, riesegui normalmente:

```bash
flwr run . --stream
```

I parametri passati con `--run-config`, come `num-server-rounds`, `local-epochs` o `batch-size`, continuano invece a sovrascrivere solo i valori definiti in `[tool.flwr.app.config]` nel `pyproject.toml`.

## 9. Cosa succede dietro le quinte

Per ogni round, in breve:

1. il server seleziona una frazione di client per il training
2. invia messaggi `TRAIN`
3. i client addestrano localmente e rispondono con pesi e metriche
4. il server aggrega i risultati con `FedAvg`
5. il server avvia la valutazione su una frazione di client, spesso tutti
6. il server aggrega le metriche di evaluation

Il ciclo continua fino al numero di round impostato.

## 10. Esercizio opzionale

- personalizzare la strategia di aggregazione oltre `FedAvg`
- aggiungere metriche custom lato client e lato server
- passare dalla simulazione locale a un deployment distribuito

## Esercizio 2: quickstart con Flower + PyTorch

In questo secondo esercizio applichiamo la stessa struttura del quickstart precedente a un caso leggermente più realistico: il progetto `fed-phish-guard`, che addestra un modello PyTorch per classificare URL di phishing in modo federato.

L'obiettivo non è riscrivere tutta l'applicazione, ma riconoscere gli stessi blocchi Flower già visti:

- un `ServerApp`, che inizializza il modello globale e avvia la strategia federata
- un `ClientApp`, che riceve i pesi globali, addestra localmente e restituisce metriche
- una configurazione in `pyproject.toml`, che definisce round, frazioni di client, iperparametri e dataset

La differenza principale rispetto al primo esercizio è il dominio: non lavoriamo più su immagini CIFAR-10, ma su URL trasformati in sequenze numeriche e classificati con una CNN testuale.

### Server federato (`fed-phish-guard/phishguard/server_app.py`)

Nel server leggiamo la configurazione dell'esperimento, costruiamo il modello globale `PhishingCNN`, inizializziamo `FedAvg` e lanciamo i round federati.

```python
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    embed_dim = context.run_config["embed-dim"]
    num_filters = context.run_config["num-filters"]
    dropout = context.run_config["dropout"]

    # Load global model
    global_model = PhishingCNN(
        vocab_size=VOCAB_SIZE,
        embed_dim=embed_dim,
        num_filters=num_filters,
        dropout=dropout,
    )
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        fraction_train=fraction_train,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
```

### Client federato (`fed-phish-guard/phishguard/client_app.py`)

Nel client troviamo di nuovo due entrypoint Flower:

- `@app.train()`: addestra il modello sui dati locali e restituisce pesi aggiornati
- `@app.evaluate()`: valuta il modello sui dati locali e restituisce metriche

```python
@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    model, device = _load_model(msg, context)
    batch_size = context.run_config["batch-size"]
    trainloader, valloader, _, pos_weight = _load_data(context, batch_size, device)

    history, _ = train_fn(
        model,
        trainloader,
        valloader,
        pos_weight,
        lr=context.run_config["learning-rate"],
        device=device,
        num_epochs=context.run_config["local-epochs"],
    )

    summary = summarize_history(history)
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": summary["avg_train_loss"],
        "val_loss": summary["avg_val_loss"],
        "val_f1": summary["avg_val_f1"],
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model, device = _load_model(msg, context)
    batch_size = context.run_config["batch-size"]
    _, _, testloader, pos_weight = _load_data(context, batch_size, device)

    eval_metrics, _, _ = eval_fn(model, testloader, pos_weight, device)

    metrics = {
        "eval_loss": eval_metrics["loss"],
        "eval_acc": eval_metrics["accuracy"],
        "eval_f1": eval_metrics["f1"],
        "num-examples": len(testloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
```

Le funzioni di supporto `_load_model` e `_load_data` sono definite nello stesso file. La prima ricostruisce il modello usando gli iperparametri della run config, la seconda sceglie se caricare dati partizionati per simulazione oppure dati locali per un deployment reale.

Per eseguire l'esercizio:

```bash
cd fed-phish-guard
flwr run . --stream
```

Puoi modificare i parametri dell'esperimento al volo, ad esempio:

```bash
flwr run . --stream --run-config "num-server-rounds=5 local-epochs=2 fraction-train=0.75"
```

## Cheasheet Flower

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

### Esecuzione (Run)

Il comando `run` deve essere sempre eseguito dalla cartella in cui risiede il tuo codice sorgente (nello specifico, dove si trova il file `pyproject.toml`).

- **`flwr run .`**
    Avvia la tua Flower App nella directory corrente (`.`) utilizzando la connessione di default (solitamente la simulazione locale). Invia il task al motore, restituisce un **RUN_ID** e termina, lasciando il processo in esecuzione in background.

    ```bash
    flwr run .
    ```

- **`flwr run . <federation_name>`**
    Avvia l'app su una specifica infrastruttura (federazione) che hai definito nel tuo `config.toml` (es. un server remoto).

    ```bash
    flwr run . remote-deployment
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

### Riferimento ufficiale

Tutorial originale:
<https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html>
