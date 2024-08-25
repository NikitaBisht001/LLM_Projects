# README

## Overview

This repository contains two projects that utilize Large Language Models (LLMs) with efficient model handling techniques such as model offloading and 4-bit quantization. These techniques are particularly useful for systems with limited RAM and GPU power.

### 1. Simple LLM Pipeline

The simple pipeline project demonstrates how to use a LLaMA (Large Language Model Meta AI) model with offloading and 4-bit quantization. This approach allows the model to run efficiently on machines with low RAM and limited GPU resources by strategically managing model loading and memory usage.

### 2. Medical Question-Answer Chatbot

The medical chatbot project involves fine-tuning a LLaMA model using a healthcare dataset from Kaggle, specifically the **"Healthcare NLP: LLMs, Transformers, Datasets"** dataset. The chatbot is designed to provide answers to medical questions and is optimized to run efficiently on low-resource machines using similar model handling techniques as the simple pipeline.

## Installation

To run these projects, you need to install the following Python packages:

```bash
pip install -U "transformers==4.40.0"
pip install accelerate bitsandbytes
```

## Project 1: Simple LLM Pipeline

### Description

This project demonstrates a basic setup to run a LLaMA model with minimal computational resources. It uses model offloading and 4-bit quantization to ensure the model runs efficiently on systems with limited RAM and GPU power.

### Key Features

- **Model Offloading:** Utilizes CPU for portions of the model to reduce GPU memory usage.
- **4-Bit Quantization:** Loads the model in 4-bit precision to further reduce memory requirements.
- **Optimized for Low-Resource Systems:** Suitable for machines with limited RAM and GPU capabilities.

## Project 2: Medical Question-Answer Chatbot

### Description

This project involves fine-tuning a LLaMA model to create a medical question-answering chatbot. The fine-tuning process uses a dataset from Kaggle: [Healthcare NLP: LLMs, Transformers, Datasets](https://www.kaggle.com/datasets/jpmiller/layoutlm). The chatbot is optimized to run on systems with low resources using model offloading and 4-bit quantization. Here we use Chainlit for chatbot interface.

### Dataset

The dataset for fine-tuning can be downloaded from Kaggle: [Healthcare NLP: LLMs, Transformers, Datasets](https://www.kaggle.com/datasets/jpmiller/layoutlm).

### Key Features

- **Fine-Tuning with Healthcare Data:** Trains the model specifically for medical question-answering.
- **Efficient Model Handling:** Uses offloading and 4-bit precision to run efficiently on low-resource systems.
- **Customizable and Extendable:** Allows customization for different types of datasets or question-answering tasks.

## Usage

1. **Clone the repository.**
2. **Install the required packages** using the commands provided in the "Installation" section.
3. **Download the dataset** for the medical chatbot from Kaggle and place it in the appropriate directory.
4. **Run the scripts** for the respective projects to see them in action.

## Requirements

- **Python 3.8 or higher**
- **Pytorch and compatible CUDA version** (if using GPU)
- **Kaggle account** to download the healthcare dataset
- **Low-RAM and low-GPU machines** are supported due to offloading and 4-bit quantization techniques.

## Conclusion

Both projects leverage the power of LLaMA models while ensuring compatibility with systems that have limited resources. By using model offloading and 4-bit quantization, these projects demonstrate how to efficiently utilize large language models without the need for high-end hardware.
