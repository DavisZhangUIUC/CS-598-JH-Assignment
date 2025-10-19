# Enhancing Knowledge Graph-Based Retrieval-Augmented Generation for Biomedical Question Answering

## 1. Introduction

### Background

Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) models have emerged as powerful tools for biomedical question answering by combining the structured knowledge of biomedical databases with the reasoning capabilities of Large Language Models (LLMs). However, the effectiveness of these systems heavily depends on how context is presented to the LLM and how well the model can leverage its pre-existing domain knowledge.

### Motivation

The baseline KG-RAG system retrieves relevant context from knowledge graphs and presents it to the LLM in an unstructured natural language format. While functional, this approach has several limitations:

- Natural language context can be verbose and difficult for LLMs to parse efficiently, especially when dealing with multiple entities and relationships.

- With RAG, the LLM is not explicitly guided to leverage its pre-existing biomedical knowledge, potentially missing opportunities for better reasoning.

This assignment addresses these limitations through 3 improvement strategies, all of which are lightweight and require no model retraining or architectural modifications, making them readily deployable to existing KG-RAG systems. I implement and evaluate these strategies on a biomedical multiple-choice question answering benchmark, systematically analyzing their individual and combined contributions to performance improvement.

## 2. Method

### Mode 0: Baseline System

The baseline KG-RAG system follows a standard retrieval-augmented generation pipeline:

1. Extract disease entities from the input question using Gemini-2.0-flash
2. Match extracted entities to disease nodes in a vector database
3. Retrieve neighborhood context from the SPOKE knowledge graph
4. Use cosine similarity to filter relevant context sentences
5. Present unstructured context and question to the LLM for answer generation

The baseline presents context in natural language format:
```
Disease psoriasis associates Gene HLA-B. 
Disease psoriasis associates Gene HLA-DQB1. 
Variant rs13203895 associates Disease psoriasis.
```

### Mode 1: Structured JSON Context

Mode 1 turns unstructured context (as seen in mode 0) into a structured JSON representation to improve LLM parsing efficiency and reduce information redundancy. JSON provides multiple advantages: there is a clear hierarchical organization of entities and relationships, and reduced token usage by eliminating redundant natural language connectors.

### Mode 2: Prior Domain Knowledge Integration

Mode 2 augments retrieved context with explicit domain knowledge statements to guide LLM reasoning without requiring additional retrieval. LLMs already possess extensive pre-existing knowledge from their training data; by explicitly priming the model with relevant information, it can provide reasoning heuristics for ambiguous cases and reduce reliance on potentially incomplete retrieved context.

### Mode 3: Combined Approach

Mode 3 integrates both structured JSON context (mode 1) and prior domain knowledge (mode 2).

## 3. Implementation

### Mode 1

The `jsonlize_context()` function in `utility.py` transforms unstructured knowledge graph context into structured JSON format through 6 regex patterns. I aim to keep the JSON as compact as possible, ensuring there are no duplicate gene/variant entries per disease and only includes non-empty associations. I also handle edge cases: stripping trailing periods from gene names and parsing both ". " and ".." sentence separators.

### Example Transformation

#### Input (Unstructured)
```
Disease psoriasis 13 associates Gene TRAF3IP2.. Disease psoriasis associates Gene HLA-B. 
Variant rs13203895 x rs4349859 associates Disease psoriasis.
```

#### Output (Structured JSON)
```json
{
  "Diseases": {
    "psoriasis 13": {
      "Genetic Associations": ["TRAF3IP2"]
    },
    "psoriasis": {
      "Genetic Associations": ["HLA-B"],
      "Variant Associations": ["rs13203895 x rs4349859"]
    }
  }
}
```

### Mode 2

Mode 2 uses few-shot learning to elicit domain knowledge from a different LLM (Claude Sonnet 4.5). I provide 2 seed examples to Claude, which generated these statements:

```
Prior Knowledge:
- Provenance & Symptoms information is useless (seed example)
- Similar diseases tend to have similar gene associations (seed example)
- HLA genes are commonly associated with autoimmune diseases
- Genes in the same pathway often have similar disease associations
- Mendelian diseases have strong, specific gene associations
- Complex diseases involve multiple genes with smaller effect sizes
- Genes expressed in the same tissue are more likely associated with diseases of that organ
- Loss-of-function variants cause recessive diseases, gain-of-function cause dominant diseases
- DNA repair genes are associated with cancer predisposition
- Metabolic genes are associated with inborn errors of metabolism
- When diseases share a gene, consider the gene's biological function
```

This text block is appended after the retrieved context and the enhanced prompt is then feeded to the LLM.

### Mode 3

Mode 3 integrates both strategies by first structuring context with `jsonlize_context()`, then appending prior knowledge.

## 4. Experimental Results

| Mode | Description | Correct Answer Rate | Improvement vs. Baseline |
|------|-------------|---------------------|-------------------------|
| **Mode 0** | Baseline (unstructured) | 72.88% | - |
| **Mode 1** | Structured JSON | 76.14% | +3.26% |
| **Mode 2** | Prior knowledge | 76.80% | +3.92% |
| **Mode 3** | Combined | **79.41%** | **+6.53%** |

### Analysis of Results

As expected, mode 3 achieves the highest accuracy (79.41%). The near-additive improvement suggests orthogonal improvements from both modes: mode 1 provides clean, efficient data representation, while mode 2 provides reasoning guidance and domain expertise.

Both modes 1 and 2 outperform the baseline, which shows that LLMs benefit from structured data formats and explicitly priming domain knowledge. The success of mode 2's few-shot approach also suggests that minimal seed examples can effectively activate extensive pre-existing knowledge in LLMs.

Interestingly, mode 2 achieves slightly higher accuracy than mode 1. There are 2 plausible reasons for this:

1. While mode 1 presents the retrieved context in a clearer format, it doesn't add new information. If the context is incomplete or noisy, mode 1 still suffers from missing information. On the other hand, mode 2's additional prior knowledge provides fallback reasoning principles that can compensate for incomplete retrieval.

<!-- 2. Currently, mode 1 only parses 6 different regex patterns in the retrieved context. There may be errors from context structuring (e.g., incorrect parsing). However, mode 2's domain knowledge provides additional constraints that can catch and correct implausible answers. -->

2. Some questions rely more on reasoning than straightforward information retrieval, which plays to mode 2's advantage.

<!-- ### Limitations

All modes depend on retrieval quality from the SPOKE knowledge graph. Poor retrieval still limits performance even with the proposed enhancement.

### Future Work

**Dynamic Knowledge Selection**
- Adaptively select relevant prior knowledge based on question 
- Reduce noise from irrelevant knowledge statements

**Explanation Generation**
- Leverage structured context for better answer explanations
- Generate reasoning traces that cite specific knowledge principles -->

