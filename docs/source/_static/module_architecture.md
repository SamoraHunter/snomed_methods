# Module Architecture Diagram

```mermaid
graph TD
    subgraph "Input Layer"
        A[SNOMED RF2 Files] -->|Path Config| B[SnomedRelations]
        C[SNOMED Description File] --> D[SnomedTermLookup]
        E[MedCAT Model Pack] -->|Optional| F[SnomedRelations]
        G[HuggingFace Model Path] --> H[ClinicalConceptEmbedder]
    end

    subgraph "Core Modules"
        B -->|Get Parents/Children| I[Concept Expansion]
        D -->|Term Search| J[Search Results]
        H -->|Generate Embeddings| K[Vector Index]
        K -->|Build FAISS Index| L[ConceptVectorSearch]
    end

    subgraph "Workflow Layer"
        M[SnomedRelations] -->|expand_codes\(\)| N[Hierarchy Expansion]
        O[ClinicalConceptEmbedder] -->|prepare_concept_text\(\)| P[Text Formatting]
        Q[SemanticSearch] -->|search\(\)| R[Mixed Strategy Search]
    end

    subgraph "Output Layer"
        I --> S[CUI Lists + Names]
        J --> T[Matching Concepts]
        L --> U[Similar Concept Search]
        R --> V[Comprehensive Results]
    end
```
