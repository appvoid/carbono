# Changelog Explanation

This document provides insights into the reasoning behind the changes made between versions of the micro-library for creating minimal neural networks in JavaScript.

---

## Version 7: JSON as Default Export with Optional Binary Compression

### Changes:
- **Default Export Mode**: JSON is now the default export format.
- **Binary Compression**: Added optional binary compression for larger neural networks.

### Reasoning:
- **JSON as Default**: JSON is human-readable and widely supported, making it easier for developers to inspect, debug, and share models. This change prioritizes usability and interoperability.
- **Binary Compression**: For larger models, JSON can result in large file sizes. Binary compression is introduced as an optional feature to reduce file size and improve load/save performance for larger neural networks. Compression is now reliable.

---

## Version 6: Dramatic File Size Optimization

### Changes:
- **File Size Reduction**: A model with 18k parameters was reduced from 376kb to 73kb.

### Reasoning:
- **Efficiency**: Smaller file sizes improve load times and reduce memory usage, which is critical for web-based applications and environments with limited resources.
- **Optimization**: We are now using .uai as default exporting method for binary compression.

---

## Version 5: Metadata Updates

### Changes:
- **New Metadata Fields**: Added new metadata fields to provide more information about the model.
- **Removed "Layers" Metadata**: The unnecessary "layers" metadata field was removed.

### Reasoning:
- **Enhanced Metadata**: Additional metadata fields improve the ability to document and understand the model's structure and training process.

---

## Version 4: Optimized Save/Load Handling for `.uai` Extension

### Changes:
- **Optimized Save/Load**: Improved handling for saving and loading models with the `.uai` extension.
- **JSON-Based Extension**: The `.uai` extension is now JSON-based.

### Reasoning:
- **Performance**: Optimized save/load operations enhance the library's usability, especially for larger models or frequent model updates.
- **Standardization**: Using a JSON-based `.uai` extension ensures compatibility and readability, aligning with modern development practices.

---

## Version 3: Code Rewrite and Adam Optimizer

### Changes:
- **Adam Optimizer**: Introduced the Adam optimizer for training.
- **Code Rewrite**: Complete code rewrite for maintenance purposes.

### Reasoning:
- **Adam Optimizer**: Adam is a popular and effective optimization algorithm, offering better convergence and performance compared to simpler optimizers like SGD.
- **Code Rewrite**: A complete rewrite improves code maintainability, readability, and extensibility, ensuring the library can evolve more easily in the future.

---

## Version 2: Softmax, Cross-Entropy, and Classification Labels

### Changes:
- **Softmax Activation**: Added softmax activation for output layers.
- **Cross-Entropy Loss**: Introduced cross-entropy loss for classification tasks.
- **Labels**: Added support for classification labels.

### Reasoning:
- **Classification Support**: These changes enable the library to handle classification tasks, expanding its use cases beyond regression.
- **Standard Techniques**: Softmax and cross-entropy are standard tools for classification, improving the library's alignment with machine learning best practices.

---

## Version 1: Initial Release

### Changes:
- **First Release**: The initial version of the library.

---

This changelog reflects the library's evolution, focusing on usability, performance, and feature expansion while maintaining minimalism and simplicity.
