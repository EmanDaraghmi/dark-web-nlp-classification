# Dark Web Threat Intelligence: NLP-Based Classification of Cybercrime Marketplace Listings

This project compares traditional machine learning and transformer-based NLP models for classifying dark web marketplace listings.

## Models

- TF-IDF + Linear SVM baseline
- BERT sequence classification model

## Dataset

The project uses the DWData dataset. The dataset is not included in this repository for ethical and storage reasons.

Place the extracted dataset inside:

## Run Additional Transformer Models

### DistilBERT

```bash
python src/train_transformer.py \
  --model_name distilbert-base-uncased \
  --output_name distilbert \
  --epochs 3 \
  --max_length 128 \
  --batch_size 8 \
  --learning_rate 2e-5data/raw/
