# Laboratorio di Federated Learning per IIoT

README per il laboratorio di Federated Learning per IIoT del corso di Industrial Internet of Things (Università di Ferrara) a.a. 2025-2026. Questo README contiene 2 esercizi. Il primo è un quickstart (molto guidato) con il framework Flower, utilizzando PyTorch e il dataset CIFAR-10.

Il secondo TODO

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

I parametri di configurazione (come il numero di round, la dimensione del batch o il dataset selezionato) sono gestiti in modo centralizzato. Possiamo configurare i parametri di default all'interno del file `pyproject.toml` e sovrascriverli dinamicamente da terminale al momento dell'esecuzione, senza dover modificare il codice sorgente.

1. **Valori di Default (`pyproject.toml`):** Tutti i parametri base sono dichiarati all'interno del file `pyproject.toml` nella sezione `[tool.flwr.app.config]`. 
2. **Sovrascrittura da Terminale:** Quando avvii un esperimento, puoi sovrascrivere questi default dinamicamente da riga di comando usando il flag `--run-config`, senza dover modificare il codice sorgente.

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

Parametri tipici:

- `num-clients`: numero di client da simulare
- `num-server-rounds`: numero di round federati da eseguire
- `fraction-train`: frazione di client da coinvolgere in ogni round di training 
- `fraction-evaluate`: frazione di client da coinvolgere in ogni round di evaluation
- `learning-rate`: learning rate da inviare ai client per il training locale

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

Altrimenti modificare `pyproject.toml`

## 8. Cosa succede dietro le quinte

Per ogni round, in breve:

1. il server seleziona una frazione di client per il training
2. invia messaggi `TRAIN`
3. i client addestrano localmente e rispondono con pesi e metriche
4. il server aggrega i risultati con `FedAvg`
5. il server avvia la valutazione su una frazione di client, spesso tutti
6. il server aggrega le metriche di evaluation

Il ciclo continua fino al numero di round impostato.

## 9. Esercizio opzionale

- personalizzare la strategia di aggregazione oltre `FedAvg`
- aggiungere metriche custom lato client e lato server
- passare dalla simulazione locale a un deployment distribuito

## Esercizio 2: quickstart con Flower + PyTorch

TODO

## Cheasheet rapida Flower

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
    flwr run . --run-config learning_rate=0.01
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
