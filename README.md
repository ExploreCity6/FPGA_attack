# Attack Chain

## Overview

The **Attack Chain** project consists of multiple modules aimed at constructing a complete attack. This project uses various techniques such as DRAM fingerprinting, communication side-channel analysis, and model training to identify potential attacks and prevent the use of unknown or unauthorized accelerators.

## Modules

### 1. **hardware_fingerprint**
This module is responsible for generating hardware fingerprints. The main functions are:
- **scripts_run**: Generates DRAM fingerprints and saves them for further processing.
- **data_process**: Processes the collected data, visualizes it, and computes the Jaccard index for co-location analysis.

### 2. **side-channel**
This module utilizes communication side-channel techniques to capture accelerator communication patterns. The key components are:
- **example**: Includes six types of accelerators, each with its corresponding communication features, and is responsible for capturing and storing traces of accelerator communications.
- **data_analysis**: Processes the collected traces, including training models to classify behaviors.

### 3. **PredictGuard**
This module is designed to detect and reject unknown types of accelerators. It also contains functionality for exploring similarities between unknown accelerators and known types. Key components include:
- **unknown_accelerator_detection**: Detects and rejects accelerators that do not match any known types.
- **similarity_exploration**: Explores the similarity between unknown accelerator types and known types to improve detection accuracy.
