#!/usr/bin/env python3
"""
FinBERT embedding extraction for news and social sentiment text.

Generates 768-dim ProsusAI/finbert embeddings with optional sentiment labels,
aligning article timestamps to trading days to comply with docs/metrics_and_evaluation.md.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from pandas.tseries.offsets import BDay
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.utils.logger import get_logger
from backend.utils.config import config
from backend.utils.file_operations import (
    create_timestamped_directory,
    load_existing_data,
    save_json,
)


logger = get_logger(__name__)
FINBERT_FALLBACK_LABELS = {0: "positive", 1: "negative", 2: "neutral"}


def sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace(" ", "_").upper()


def default_input_dir(source: str) -> Path:
    return Path("data/raw/news" if source == "news" else "data/raw/social")


def find_latest_source_file(ticker: str, source: str, input_dir: Path) -> Path:
    """Locate latest raw JSON file for ticker/source combo."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    ticker_key = sanitize_ticker(ticker)
    suffix = "news" if source == "news" else "stocktwits"
    candidate_dirs = sorted([p for p in input_dir.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)
    for date_dir in candidate_dirs:
        candidate = date_dir / f"{ticker_key}_{suffix}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No raw {source} file found for {ticker} under {input_dir}")


def load_json_records(path: Path) -> List[Dict]:
    """Load JSON content as list of dicts."""
    data = load_existing_data(path, file_format="json")
    if data is None:
        raise ValueError(f"Could not parse data from {path}")
    if isinstance(data, pd.DataFrame):
        return data.to_dict("records")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON format in {path}")


def align_to_trading_day(timestamp: str) -> str:
    """Align raw timestamp to next valid trading day (Mon-Fri)."""
    ts = pd.to_datetime(timestamp, utc=True, errors="coerce")
    if ts is None or pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {timestamp}")
    if ts.weekday() >= 5:
        ts = (ts + BDay(1)).normalize()
    else:
        ts = ts.normalize()
    return ts.date().isoformat()


def prepare_news_samples(records: List[Dict], ticker: str) -> List[Dict]:
    samples: List[Dict] = []
    for idx, article in enumerate(records):
        title = article.get("title") or ""
        description = article.get("description") or ""
        text = f"{title.strip()} {description.strip()}".strip()
        if not text:
            logger.debug("Skipping article with empty text | idx=%s", idx)
            continue
        timestamp = article.get("published_at") or article.get("publishedAt")
        if not timestamp:
            raise ValueError("Article missing published_at timestamp")
        samples.append(
            {
                "id": article.get("url") or f"{ticker}_news_{idx}",
                "text": text,
                "timestamp": timestamp,
                "aligned_day": align_to_trading_day(timestamp),
                "snippet": text[:100],
                "source_name": article.get("source_name"),
                "raw": article,
            }
        )
    return samples


def prepare_social_samples(records: List[Dict], ticker: str) -> List[Dict]:
    samples: List[Dict] = []
    payloads = records if isinstance(records, list) else [records]
    for payload in payloads:
        messages = payload.get("messages") or []
        for msg in messages:
            text = (msg.get("body") or "").strip()
            if not text:
                continue
            timestamp = msg.get("created_at")
            if not timestamp:
                raise ValueError("Social message missing created_at timestamp")
            samples.append(
                {
                    "id": str(msg.get("id") or f"{ticker}_social_{len(samples)}"),
                    "text": text,
                    "timestamp": timestamp,
                    "aligned_day": align_to_trading_day(timestamp),
                    "snippet": text[:100],
                    "source_name": msg.get("source"),
                    "raw": msg,
                }
            )
    return samples


def load_text_samples(path: Path, source: str, ticker: str) -> List[Dict]:
    records = load_json_records(path)
    if source == "news":
        samples = prepare_news_samples(records, ticker)
    else:
        samples = prepare_social_samples(records, ticker)
    if not samples:
        raise ValueError(f"No text samples found in {path}")
    logger.info("Loaded %s %s samples from %s", len(samples), source, path)
    return samples


def select_device(preferred: Optional[str]) -> torch.device:
    """
    Select compute device for model inference with validation.
    
    Args:
        preferred: Preferred device string (e.g., 'cuda', 'cuda:0', 'cpu').
                   If None, auto-selects CUDA if available, else CPU.
    
    Returns:
        torch.device instance validated for availability.
    
    Notes:
        If a CUDA device is requested but CUDA is unavailable, logs a warning
        and falls back to CPU instead of failing.
    """
    if preferred:
        device = torch.device(preferred)
        # Validate CUDA availability if CUDA device requested
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning(
                "CUDA device requested (%s) but CUDA is not available. Falling back to CPU. "
                "Check torch.cuda.is_available() and your PyTorch installation.",
                preferred
            )
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info("Using device: %s", device.type)
    return device


def load_models(model_name: str, device: torch.device, classify: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embed_model = AutoModel.from_pretrained(model_name)
    embed_model.to(device)
    embed_model.eval()
    classifier = None
    label_map: Dict[int, str] | None = None
    if classify:
        classifier = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier.to(device)
        classifier.eval()
        if getattr(classifier.config, "id2label", None):
            label_map = {int(k): v.lower() for k, v in classifier.config.id2label.items()}
        else:
            label_map = FINBERT_FALLBACK_LABELS
    return tokenizer, embed_model, classifier, label_map


def chunk_items(items: Sequence[Dict], size: int) -> Iterable[Sequence[Dict]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def extract_embeddings_batch(
    batch: Sequence[Dict],
    tokenizer,
    embed_model,
    classifier,
    label_map: Optional[Dict[int, str]],
    device: torch.device,
    max_length: int,
) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[List[float]]]]:
    texts = [item["text"] for item in batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = embed_model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
        sentiment_labels = None
        sentiment_probs = None
        if classifier is not None:
            cls_outputs = classifier(**encoded)
            probs = torch.softmax(cls_outputs.logits, dim=-1).detach().cpu().numpy()
            sentiment_probs = probs.tolist()
            sentiment_labels = [
                (label_map.get(int(np.argmax(row)), str(int(np.argmax(row)))) if label_map else str(int(np.argmax(row))))
                for row in probs
            ]
    return embeddings, sentiment_labels, sentiment_probs


def generate_embeddings(
    samples: List[Dict],
    tokenizer,
    embed_model,
    classifier,
    label_map,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[np.ndarray, List[Optional[str]], List[Optional[List[float]]]]:
    embeddings_accum: List[np.ndarray] = []
    sentiment_labels: List[Optional[str]] = []
    sentiment_probs: List[Optional[List[float]]] = []
    total = len(samples)
    for idx, batch in enumerate(chunk_items(samples, batch_size), start=1):
        batch_embeddings, batch_labels, batch_probs = extract_embeddings_batch(
            batch,
            tokenizer=tokenizer,
            embed_model=embed_model,
            classifier=classifier,
            label_map=label_map,
            device=device,
            max_length=max_length,
        )
        embeddings_accum.append(batch_embeddings)
        if classifier is not None:
            sentiment_labels.extend(batch_labels or [])
            sentiment_probs.extend(batch_probs or [])
        else:
            sentiment_labels.extend([None] * len(batch))
            sentiment_probs.extend([None] * len(batch))
        processed = min(idx * batch_size, total)
        if idx == 1 or idx % 10 == 0 or processed == total:
            logger.info("Processed %s/%s texts for embeddings", processed, total)
    return np.vstack(embeddings_accum), sentiment_labels, sentiment_probs


def handle_cuda_oom(exc: RuntimeError) -> bool:
    return "cuda" in str(exc).lower() and "out of memory" in str(exc).lower()


def save_embeddings(
    ticker: str,
    source: str,
    embeddings: np.ndarray,
    samples: List[Dict],
    sentiment_labels: List[Optional[str]],
    sentiment_probs: List[Optional[List[float]]],
    model_name: str,
    batch_size: int,
    device: torch.device,
    raw_path: Path,
    output_dir: Path,
) -> Path:
    processed_dir = create_timestamped_directory(output_dir / sanitize_ticker(ticker))
    array_path = processed_dir / f"embeddings_{source}.npy"
    metadata_path = processed_dir / f"embeddings_{source}_metadata.json"
    np.save(array_path, embeddings, allow_pickle=False)

    metadata_records = []
    for sample, label, probs in zip(samples, sentiment_labels, sentiment_probs):
        record = {
            "id": sample["id"],
            "ticker": ticker,
            "source": source,
            "original_timestamp": sample["timestamp"],
            "aligned_trading_day": sample["aligned_day"],
            "snippet": sample["snippet"],
            "model": model_name,
            "device": device.type,
            "text_length": len(sample["text"]),
            "sentiment": label,
            "sentiment_probs": probs,
        }
        metadata_records.append(record)

    save_json(
        {
            "ticker": ticker,
            "source": source,
            "model": model_name,
            "batch_size": batch_size,
            "device": device.type,
            "generated_at": datetime.utcnow().isoformat(),
            "raw_file": str(raw_path),
            "embedding_file": str(array_path),
            "records": metadata_records,
        },
        metadata_path,
    )
    logger.info(
        "Saved embeddings and metadata | path=%s samples=%s dims=%s",
        array_path,
        embeddings.shape[0],
        embeddings.shape[1],
    )
    return processed_dir


def process_embeddings(
    ticker: str,
    source: str,
    input_dir: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    device_str: Optional[str],
    classify_sentiment: bool,
) -> Path:
    raw_path = find_latest_source_file(ticker, source, input_dir)
    samples = load_text_samples(raw_path, source, ticker)
    device = select_device(device_str)
    
    # Load models with error handling for download/initialization failures
    try:
        tokenizer, embed_model, classifier, label_map = load_models(model_name, device, classify_sentiment)
    except (OSError, ValueError) as exc:
        logger.error(
            "Failed to load model '%s'. Error: %s. "
            "Hints: (1) Check internet connectivity for model download. "
            "(2) Verify model name is correct on Hugging Face Hub. "
            "(3) Check local cache at ~/.cache/huggingface/hub. "
            "(4) Ensure sufficient disk space for model files.",
            model_name,
            str(exc)
        )
        raise RuntimeError(
            f"Model loading failed for '{model_name}'. "
            f"See logs for details and troubleshooting hints."
        ) from exc

    try:
        embeddings, labels, probs = generate_embeddings(
            samples,
            tokenizer=tokenizer,
            embed_model=embed_model,
            classifier=classifier,
            label_map=label_map,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
    except RuntimeError as exc:
        if device.type == "cuda" and handle_cuda_oom(exc):
            logger.warning("CUDA OOM detected. Retrying on CPU with same batch size.")
            device = torch.device("cpu")
            embed_model.to(device)
            if classifier:
                classifier.to(device)
            embeddings, labels, probs = generate_embeddings(
                samples,
                tokenizer=tokenizer,
                embed_model=embed_model,
                classifier=classifier,
                label_map=label_map,
                device=device,
                batch_size=batch_size,
                max_length=max_length,
            )
        else:
            raise

    return save_embeddings(
        ticker=ticker,
        source=source,
        embeddings=embeddings,
        samples=samples,
        sentiment_labels=labels,
        sentiment_probs=probs,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        raw_path=raw_path,
        output_dir=output_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FinBERT embeddings for news or social sentiment text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ticker", required=True, help="Ticker or company identifier (RELIANCE or RELIANCE.NS)")
    parser.add_argument("--source", choices=["news", "social"], default="news", help="Source dataset to process")
    parser.add_argument("--input-dir", type=Path, help="Raw input directory (defaults based on source)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Processed output directory")
    parser.add_argument("--model-name", type=str, default=config.finbert_model_name, help="Transformers model name")
    parser.add_argument("--batch-size", type=int, default=config.finbert_batch_size, help="Batch size for inference")
    parser.add_argument("--max-length", type=int, default=config.finbert_max_length, help="Max token length")
    parser.add_argument("--device", type=str, help="Device override (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--classify-sentiment", action="store_true", help="Enable FinBERT sentiment classification")

    args = parser.parse_args()
    input_dir = args.input_dir or default_input_dir(args.source)

    process_embeddings(
        ticker=args.ticker,
        source=args.source,
        input_dir=input_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device_str=args.device,
        classify_sentiment=args.classify_sentiment,
    )


if __name__ == "__main__":
    main()

