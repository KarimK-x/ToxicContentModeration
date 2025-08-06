# Toxic Content Moderation Project

This project focuses on developing a robust system for **toxic content moderation** by classifying user queries and associated image descriptions into various "toxic" categories. It leverages deep learning techniques to identify and categorize harmful content, aiming to ensure a safer online environment.

## Key Features & Highlights

*   **Comprehensive Data Preprocessing**:
    *   **Thorough Data Cleaning**: Addressed and removed **973 fully duplicated rows**, ensuring data integrity [1]. Importantly, there were **0 rows with duplicated queries and different Toxic Categories**, allowing for safe removal of duplicates [1]. No null rows were found [1].
    *   **Effective Feature Engineering**: Enhanced model input by **concatenating query and image descriptions columns** [1].
    *   **Advanced Text Normalization**: Implemented standard text cleaning procedures including converting text to **alphabet-only, all lowercase**, applying **lemmatization**, and **removing stopwords** [1]. A flexible `cleanAndLemmatizeInput()` function was also developed, allowing for minimal cleaning (no lemmatization or stopword removal) specifically for Transformer models [1].
    *   **Stratified Data Splitting**: The dataset was strategically split into train, test, and validation sets using a **60%/15%/15% ratio** [1]. This ensures that each target label is adequately represented across all sets (the 60% trainset allows test and validation sets to contain at least 2 instances of each target label), which is crucial for balanced model training and evaluation [1].
*   **Optimal Multi-Class Label Encoding**:
    *   Utilized **One-Hot Encoding** for multi-class labels, which is essential because there's **no inherent ordinal relationship between the target categories** (e.g., "Safe," "Violent Crimes," "Sex-Related Crimes") [2]. Label Encoding would falsely imply such a relationship, which is misleading for the model [2].
*   **Deep Learning Model Architectures**:
    *   **LSTM Classifier Baseline**: An **LSTM (Long Short-Term Memory) classifier** was designed and established as the deep learning baseline model [2].
    *   **Transformer Fine-tuning**: Explored and implemented advanced models by **fine-tuning DistilBERT** [3]. This was done efficiently using **PEFT (Parameter-Efficient Fine-tuning) with LoRA (Low-Rank Adaptation)** [3], demonstrating an approach to leverage large pre-trained models effectively.

## Performance (LSTM Baseline)

The initial Deep Learning Baseline, an LSTM Classifier, achieved strong performance metrics [2]:
*   Training Accuracy: **98.6%**
*   Validation Accuracy: **94.57%**
*   Test Accuracy: **94.51%**

## Technologies Used

*   Deep Learning Frameworks (TensorFlow/Keras and PyTorch - implied by LSTM and Transformer models)
*   Hugging Face Transformers Library (Implied by DistilBERT)
*   PEFT (Parameter-Efficient Fine-tuning)
*   LoRA (Low-Rank Adaptation)
