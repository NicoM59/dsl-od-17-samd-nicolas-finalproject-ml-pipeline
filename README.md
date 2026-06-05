

<style>#mermaid-1780652738519{font-family:sans-serif;font-size:16px;fill:#333;}#mermaid-1780652738519 .error-icon{fill:#552222;}#mermaid-1780652738519 .error-text{fill:#552222;stroke:#552222;}#mermaid-1780652738519 .edge-thickness-normal{stroke-width:2px;}#mermaid-1780652738519 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1780652738519 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1780652738519 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1780652738519 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1780652738519 .marker{fill:#333333;}#mermaid-1780652738519 .marker.cross{stroke:#333333;}#mermaid-1780652738519 svg{font-family:sans-serif;font-size:16px;}#mermaid-1780652738519 .label{font-family:sans-serif;color:#333;}#mermaid-1780652738519 .label text{fill:#333;}#mermaid-1780652738519 .node rect,#mermaid-1780652738519 .node circle,#mermaid-1780652738519 .node ellipse,#mermaid-1780652738519 .node polygon,#mermaid-1780652738519 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-1780652738519 .node .label{text-align:center;}#mermaid-1780652738519 .node.clickable{cursor:pointer;}#mermaid-1780652738519 .arrowheadPath{fill:#333333;}#mermaid-1780652738519 .edgePath .path{stroke:#333333;stroke-width:1.5px;}#mermaid-1780652738519 .flowchart-link{stroke:#333333;fill:none;}#mermaid-1780652738519 .edgeLabel{background-color:#e8e8e8;text-align:center;}#mermaid-1780652738519 .edgeLabel rect{opacity:0.5;background-color:#e8e8e8;fill:#e8e8e8;}#mermaid-1780652738519 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-1780652738519 .cluster text{fill:#333;}#mermaid-1780652738519 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:sans-serif;font-size:12px;background:hsl(80,100%,96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1780652738519:root{--mermaid-font-family:sans-serif;}#mermaid-1780652738519:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1780652738519 flowchart{fill:apa;}</style>


## Architecture du projet

```mermaid
sequenceDiagram
    participant User as Utilisateur
    participant HF as Frontend (Hugging Face)
    participant API as Backend (Render)
    participant Model as Modèle ML

    User->>HF: Saisie du texte
    HF->>API: POST /predict (JSON payload)
    API->>Model: Inférence & Traitement
    Model-->>API: Retourne probabilités
    API-->>HF: Retourne JSON (Résultat + URL source)
    HF-->>User: Affichage dynamique des résultats
```

graph TD
    Data[Données Brutes/Txt] -->|Extract| Airflow[Airflow Orchestration]
    Airflow -->|Transform| Clean[Données Nettoyées]
    Clean -->|Train| Model[Entraînement & Tests]
    Model -->|Log| MLflow[MLflow Registry]
    MLflow -->|Deploy| Render[API FastAPI / Render]
    Render -->|Monitor| Evidently[Evidently AI Monitoring]
    Evidently -->|Feedback| Data


## Flux d'inférence (templExtract
