# Laboratorio di Federated Learning per IIoT

README per il laboratorio di Federated Learning per IIoT del corso di Industrial Internet of Things (Università di Ferrara) a.a. 2025-2026. Questo README contiene 2 esercizi. Il primo è un quickstart (molto guidato) con il framework Flower, utilizzando PyTorch e il dataset CIFAR-10.

Il secondo TODO

## Esercizio 1: quickstart con Flower + PyTorch

L'obiettivo e costruire una piccola applicazione di Federated Learning con:

- Flower Framework
- Flower Datasets
- PyTorch
- Dataset CIFAR-10

### Obiettivo

Alla fine avrai un sistema federato con:

- un `ServerApp` che coordina l'addestramento
- un `ClientApp` eseguito su piu nodi/partizioni
- strategia `FedAvg` per aggregare pesi e metriche

### 1: Preparazione ambiente

Creazione ambiente virtuale:

```bash
python -m venv flwr-env
source flwr-env/bin/activate 
```

Installa Flower con supporto simulazione:

```bash
pip install -U "flwr[simulation]"
```

Crea il progetto di esempio:

Struttura tipica di un progetto federato con Flower:

```text
quickstart-pytorch/
  pyproject.toml
  README.md
  pytorchexample/
 __init__.py
 task.py
 client_app.py
 server_app.py
```

### 2) Dataset CIFAR-10 e partizionamento

Per simulare un contesto cross-silo, CIFAR-10 viene diviso in partizioni (una per client).
Con `flwr-datasets` è possibile:

- definire un partizionatore IID (ad esempio 10 partizioni)
- caricare la partizione del client corrente
- dividere localmente in train/validation
- applicare trasformazioni PyTorch
- costruire `DataLoader` locali

Idea base della funzione, che si trova nel file task.py, `load_data(partition_id, num_partitions, batch_size)`:

1. inizializza una sola volta `FederatedDataset`
2. carica la partizione assegnata al nodo
3. split locale train/test (es. 80/20)
4. applica `ToTensor()` + `Normalize(...)`
5. ritorna `trainloader` e `valloader`

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

### 3) Modello e training locale

Nel file `task.py` si definiscono:

- una CNN semplice per CIFAR-10 (`Net`)
- `train(...)` per addestramento locale
- `test(...)` per validazione locale

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

## 4) Concetti base di Flower: `Message` e `Record`

In Flower, l'intero scambio di informazioni tra Server e Client avviene tramite l'oggetto `Message`. Ogni messaggio trasporta al suo interno un `RecordDict`, una struttura dati flessibile progettata appositamente per gestire il payload del training federato.

All'interno del `RecordDict`, i dati sono categorizzati in tre tipologie principali di record:

- **`ArrayRecord`**: Gestisce tensori e array n-dimensionali. È il record utilizzato per scambiare e aggiornare i **pesi del modello**.
- **`MetricRecord`**: Raccoglie le metriche scalari di valutazione. Viene utilizzato per tracciare e aggregare l'andamento del training (es. *loss*, *accuracy*, *num-examples* processati).
-**`ConfigRecord`**: Contiene i parametri di configurazione. Permette al server di inviare istruzioni o iperparametri dinamici ai client (es. *learning rate*, *batch size*, numero di epoche locali).

Questo schema standardizza lo scambio dati durante i round federati.

### 5) Client federato (`client_app.py`)

Nel `ClientApp` si implementano in genere due entrypoint:

- `@app.train()`: riceve i pesi globali, fa training locale, ritorna pesi aggiornati + metriche
- `@app.evaluate()`: valuta il modello ricevuto, ritorna metriche di valutazione

Codice della funzione di training:

```python

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
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
```

Codice della funzione di evaluation:

```python
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
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
```

### 6) Server federato (`server_app.py`)

Nel `ServerApp`:

1. leggi i parametri da `context.run_config`
2. inizializza il modello globale (`initial_arrays`)
3. configura la strategia (`FedAvg`)
4. avvia `strategy.start(...)`
5. salva il modello finale su disco

Esempio di parametri frequenti:

- `num-server-rounds`
- `fraction-train`
- `fraction-evaluate`
- `learning-rate`

Codice completo del metodo main del `ServerApp`:

```python

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
```

### 7) Avvio simulazione

Esegui:

```bash
flwr run . --stream
```

Vedrai i round federati, con campionamento client, aggregazione training, aggregazione evaluation e metriche aggregate.

Override rapido della config a runtime:

```bash
flwr run . --stream --run-config "num-server-rounds=5 local-epochs=3"
```

### 8) Cosa succede dietro le quinte

Per ogni round, in breve:

1. il server seleziona una frazione di client per training
2. invia messaggi `TRAIN`
3. i client addestrano localmente e rispondono con pesi+metriche
4. il server aggrega (FedAvg)
5. il server avvia valutazione su una frazione (spesso tutti i client)
6. aggrega le metriche di evaluation

Il ciclo continua fino al numero di round impostato.

### 9) Passi successivi opzionali

- personalizzare la strategia federata (oltre FedAvg)
- aggiungere metriche custom client/server
- introdurre scheduler del learning rate per round
- usare un dataset reale gia partizionato per dominio applicativo
- passare da simulazione locale a deployment distribuito

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

## Esecuzione (Run)

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

## Monitoraggio (Status & Logs)
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

## Gestione e Interruzione

- **`flwr stop <run_id>`**
    Interrompe in modo sicuro un run attualmente in corso, inviando una richiesta di arresto al SuperLink (il nodo centrale).

    ```bash
    flwr stop 12345678
    ```

## Riferimento ufficiale

Tutorial di riferimento (originale):
<https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html>
