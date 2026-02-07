"""
AI Scorecard Generator - Tickertape-style stock analysis scorecard.

Pipeline: DB (fundamentals) + RAG (news context) → AI → Scorecard
Produces a comprehensive investment scorecard with category-wise scores.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


# ─── Score Category Definitions ──────────────────────────────────────────────

SCORECARD_CATEGORIES = {
    "valuation": {
        "name": "Valuation",
        "icon": "💰",
        "description": "Is the stock fairly priced?",
        "weight": 0.20,
    },
    "profitability": {
        "name": "Profitability",
        "icon": "📈",
        "description": "How profitable is the business?",
        "weight": 0.20,
    },
    "growth": {
        "name": "Growth",
        "icon": "🚀",
        "description": "Revenue and earnings growth trajectory",
        "weight": 0.20,
    },
    "financial_health": {
        "name": "Financial Health",
        "icon": "🏦",
        "description": "Debt levels and financial stability",
        "weight": 0.15,
    },
    "sentiment": {
        "name": "Market Sentiment",
        "icon": "📰",
        "description": "News and market perception",
        "weight": 0.15,
    },
    "ownership": {
        "name": "Ownership",
        "icon": "👥",
        "description": "Promoter holding and institutional interest",
        "weight": 0.10,
    },
}


def _rate(value: Optional[float], thresholds: List[Tuple[float, int]], default: int = 5) -> int:
    """
    Rate a value on a 1-10 scale given sorted thresholds.
    thresholds is a list of (cutoff, score) from worst to best.
    """
    if value is None:
        return default
    for cutoff, score in thresholds:
        if value <= cutoff:
            return score
    return thresholds[-1][1] if thresholds else default


def _rate_inverse(value: Optional[float], thresholds: List[Tuple[float, int]], default: int = 5) -> int:
    """Rate inversely - lower value = higher score (e.g., debt, PE)."""
    if value is None:
        return default
    for cutoff, score in thresholds:
        if value <= cutoff:
            return score
    return thresholds[-1][1] if thresholds else default


def compute_valuation_score(fundamentals: Dict) -> Dict:
    """Score valuation metrics (PE, PB, dividend yield)."""
    ratios = fundamentals.get("ratios", {})
    
    pe = ratios.get("pe_ratio")
    industry_pe = ratios.get("industry_pe")
    pb = fundamentals.get("price_to_book") or (
        ratios.get("current_price", 0) / ratios.get("book_value", 1)
        if ratios.get("book_value") and ratios.get("current_price") else None
    )
    div_yield = ratios.get("dividend_yield")
    
    # PE Score: lower is better (for profitable companies)
    pe_score = _rate_inverse(pe, [
        (10, 9), (15, 8), (20, 7), (25, 6), (30, 5), (40, 4), (50, 3), (80, 2)
    ], default=5)
    
    # If PE below industry PE, bonus
    if pe and industry_pe and pe < industry_pe:
        pe_score = min(10, pe_score + 1)
    
    # PB Score: lower is better
    pb_score = _rate_inverse(pb, [
        (1, 9), (2, 8), (3, 7), (5, 6), (8, 5), (12, 4), (20, 3)
    ], default=5)
    
    # Dividend yield: higher is better
    dy_score = _rate(div_yield, [
        (0, 3), (0.5, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9)
    ], default=4)
    
    score = round((pe_score * 0.5 + pb_score * 0.3 + dy_score * 0.2), 1)
    
    return {
        "score": score,
        "max_score": 10,
        "components": {
            "pe_score": pe_score,
            "pb_score": pb_score,
            "dividend_yield_score": dy_score,
        },
        "details": {
            "pe_ratio": pe,
            "industry_pe": industry_pe,
            "price_to_book": round(pb, 2) if pb else None,
            "dividend_yield": div_yield,
        },
        "verdict": "Undervalued" if score >= 7 else ("Fairly Valued" if score >= 5 else "Overvalued"),
    }


def compute_profitability_score(fundamentals: Dict) -> Dict:
    """Score profitability metrics (ROCE, ROE, margins)."""
    ratios = fundamentals.get("ratios", {})
    
    roce = ratios.get("roce")
    roe = ratios.get("roe")
    profit_margins = fundamentals.get("profit_margins")
    operating_margins = fundamentals.get("operating_margins")
    
    # If we have percentage margins from yfinance (0-1 range), convert
    if profit_margins and profit_margins < 1:
        profit_margins = profit_margins * 100
    if operating_margins and operating_margins < 1:
        operating_margins = operating_margins * 100
    
    roce_score = _rate(roce, [
        (5, 3), (10, 5), (15, 6), (20, 7), (25, 8), (30, 9), (40, 10)
    ], default=5)
    
    roe_score = _rate(roe, [
        (5, 3), (10, 5), (15, 7), (20, 8), (25, 9), (30, 10)
    ], default=5)
    
    margin_score = _rate(profit_margins, [
        (5, 4), (10, 5), (15, 6), (20, 7), (25, 8), (30, 9)
    ], default=5)
    
    score = round(roce_score * 0.4 + roe_score * 0.4 + margin_score * 0.2, 1)
    
    return {
        "score": score,
        "max_score": 10,
        "components": {
            "roce_score": roce_score,
            "roe_score": roe_score,
            "margin_score": margin_score,
        },
        "details": {
            "roce": roce,
            "roe": roe,
            "profit_margins": profit_margins,
            "operating_margins": operating_margins,
        },
        "verdict": "Highly Profitable" if score >= 7 else ("Moderately Profitable" if score >= 5 else "Low Profitability"),
    }


def compute_growth_score(fundamentals: Dict) -> Dict:
    """Score growth metrics."""
    revenue_growth = fundamentals.get("revenue_growth")
    earnings_growth = fundamentals.get("earnings_growth")
    eps = fundamentals.get("ratios", {}).get("eps")
    
    # Convert from 0-1 range if needed
    if revenue_growth and -1 < revenue_growth < 1:
        revenue_growth = revenue_growth * 100
    if earnings_growth and -1 < earnings_growth < 1:
        earnings_growth = earnings_growth * 100
    
    rev_score = _rate(revenue_growth, [
        (-10, 2), (0, 4), (5, 5), (10, 6), (15, 7), (20, 8), (30, 9), (50, 10)
    ], default=5)
    
    earn_score = _rate(earnings_growth, [
        (-20, 2), (-5, 3), (0, 4), (5, 5), (10, 6), (15, 7), (25, 8), (40, 9)
    ], default=5)
    
    eps_score = _rate(eps, [
        (0, 2), (5, 4), (15, 5), (30, 6), (50, 7), (80, 8), (120, 9)
    ], default=5)
    
    score = round(rev_score * 0.4 + earn_score * 0.4 + eps_score * 0.2, 1)
    
    return {
        "score": score,
        "max_score": 10,
        "components": {
            "revenue_growth_score": rev_score,
            "earnings_growth_score": earn_score,
            "eps_score": eps_score,
        },
        "details": {
            "revenue_growth_pct": revenue_growth,
            "earnings_growth_pct": earnings_growth,
            "eps": eps,
        },
        "verdict": "Strong Growth" if score >= 7 else ("Moderate Growth" if score >= 5 else "Slow Growth"),
    }


def compute_financial_health_score(fundamentals: Dict) -> Dict:
    """Score financial health (debt, current ratio)."""
    ratios = fundamentals.get("ratios", {})
    
    debt_cr = ratios.get("debt_cr")
    mcap = ratios.get("market_cap_cr")
    current_ratio_val = fundamentals.get("current_ratio")
    debt_to_equity = fundamentals.get("debt_to_equity")
    
    # Debt-to-market-cap ratio
    debt_mcap_ratio = None
    if debt_cr and mcap and mcap > 0:
        debt_mcap_ratio = debt_cr / mcap
    
    debt_score = _rate_inverse(debt_mcap_ratio, [
        (0.05, 9), (0.1, 8), (0.2, 7), (0.3, 6), (0.5, 5), (0.7, 4), (1.0, 3), (2.0, 2)
    ], default=5)
    
    de_score = _rate_inverse(debt_to_equity, [
        (0.1, 9), (0.3, 8), (0.5, 7), (0.8, 6), (1.0, 5), (1.5, 4), (2.0, 3)
    ], default=5)
    
    cr_score = _rate(current_ratio_val, [
        (0.5, 3), (0.8, 4), (1.0, 5), (1.3, 6), (1.5, 7), (2.0, 8), (3.0, 9)
    ], default=5)
    
    score = round(debt_score * 0.4 + de_score * 0.3 + cr_score * 0.3, 1)
    
    return {
        "score": score,
        "max_score": 10,
        "components": {
            "debt_score": debt_score,
            "debt_equity_score": de_score,
            "current_ratio_score": cr_score,
        },
        "details": {
            "debt_cr": debt_cr,
            "debt_to_mcap_ratio": round(debt_mcap_ratio, 3) if debt_mcap_ratio else None,
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio_val,
        },
        "verdict": "Strong Balance Sheet" if score >= 7 else ("Adequate" if score >= 5 else "High Leverage"),
    }


def compute_sentiment_score(sentiment_data: Optional[Dict]) -> Dict:
    """Score based on news sentiment analysis."""
    if not sentiment_data:
        return {
            "score": 5.0,
            "max_score": 10,
            "components": {},
            "details": {"note": "No sentiment data available"},
            "verdict": "Neutral",
        }
    
    sent_score = sentiment_data.get("sentiment_score", 0)  # -1 to +1
    confidence = sentiment_data.get("confidence", 0.5)
    risk_level = sentiment_data.get("risk_level", "MEDIUM")
    
    # Map -1..+1 to 1..10
    normalized = round((sent_score + 1) * 4.5 + 1, 1)
    
    # Adjust by confidence
    adjusted = round(5 + (normalized - 5) * confidence, 1)
    
    # Risk penalty
    risk_penalty = {"LOW": 0, "MEDIUM": -0.5, "HIGH": -1.5}.get(risk_level, -0.5)
    final = max(1, min(10, round(adjusted + risk_penalty, 1)))
    
    return {
        "score": final,
        "max_score": 10,
        "components": {
            "raw_sentiment": sent_score,
            "confidence": confidence,
            "risk_level": risk_level,
        },
        "details": {
            "reasoning": sentiment_data.get("reasoning", ""),
            "key_themes": sentiment_data.get("key_themes", ""),
            "article_count": sentiment_data.get("article_count", 0),
        },
        "verdict": "Positive" if final >= 7 else ("Neutral" if final >= 4 else "Negative"),
    }


def compute_ownership_score(fundamentals: Dict) -> Dict:
    """Score promoter holding and ownership quality."""
    ratios = fundamentals.get("ratios", {})
    promoter_holding = ratios.get("promoter_holding")
    
    # Shareholding trend from data
    shareholding = fundamentals.get("shareholding", [])
    
    promo_score = _rate(promoter_holding, [
        (20, 3), (30, 4), (40, 5), (50, 6), (55, 7), (60, 8), (70, 9), (75, 10)
    ], default=5)
    
    # Pledged shares penalty (if available in cons)
    cons = fundamentals.get("cons", [])
    pledged = any("pledg" in c.lower() for c in cons) if cons else False
    if pledged:
        promo_score = max(1, promo_score - 2)
    
    return {
        "score": promo_score,
        "max_score": 10,
        "components": {
            "promoter_holding_score": promo_score,
        },
        "details": {
            "promoter_holding_pct": promoter_holding,
            "pledged_shares": pledged,
        },
        "verdict": "Strong Ownership" if promo_score >= 7 else ("Moderate" if promo_score >= 5 else "Weak Ownership"),
    }


def generate_scorecard(
    fundamentals: Dict,
    sentiment_data: Optional[Dict] = None,
    news_articles: Optional[List[Dict]] = None,
) -> Dict:
    """
    Generate a comprehensive AI-powered stock scorecard (Tickertape-style).
    
    Args:
        fundamentals: Company fundamentals from DB (screener/yfinance data)
        sentiment_data: RAG sentiment analysis results
        news_articles: Recent news articles for context
        
    Returns:
        Complete scorecard dict with category scores and overall rating
    """
    symbol = fundamentals.get("symbol", "UNKNOWN")
    company_name = fundamentals.get("company_name", symbol)
    
    # Compute category scores
    categories = {
        "valuation": compute_valuation_score(fundamentals),
        "profitability": compute_profitability_score(fundamentals),
        "growth": compute_growth_score(fundamentals),
        "financial_health": compute_financial_health_score(fundamentals),
        "sentiment": compute_sentiment_score(sentiment_data),
        "ownership": compute_ownership_score(fundamentals),
    }
    
    # Compute weighted overall score
    total_score = 0
    total_weight = 0
    for cat_key, cat_data in categories.items():
        weight = SCORECARD_CATEGORIES[cat_key]["weight"]
        total_score += cat_data["score"] * weight
        total_weight += weight
    
    overall_score = round(total_score / total_weight, 1) if total_weight > 0 else 5.0
    
    # Determine overall verdict
    if overall_score >= 8:
        overall_verdict = "Strong Buy"
        overall_badge = "excellent"
    elif overall_score >= 6.5:
        overall_verdict = "Buy"
        overall_badge = "good"
    elif overall_score >= 5:
        overall_verdict = "Hold"
        overall_badge = "average"
    elif overall_score >= 3.5:
        overall_verdict = "Underperform"
        overall_badge = "below_average"
    else:
        overall_verdict = "Sell"
        overall_badge = "poor"
    
    # Build pros/cons summary
    pros = fundamentals.get("pros", [])
    cons = fundamentals.get("cons", [])
    
    # Identify strongest and weakest categories
    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["score"], reverse=True)
    strengths = [
        {"category": SCORECARD_CATEGORIES[k]["name"], "score": v["score"], "verdict": v["verdict"]}
        for k, v in sorted_cats[:2]
    ]
    weaknesses = [
        {"category": SCORECARD_CATEGORIES[k]["name"], "score": v["score"], "verdict": v["verdict"]}
        for k, v in sorted_cats[-2:]
    ]
    
    scorecard = {
        "symbol": symbol,
        "company_name": company_name,
        "sector": fundamentals.get("sector"),
        "industry": fundamentals.get("industry"),
        "source": fundamentals.get("source"),
        "generated_at": datetime.now().isoformat(),
        
        # Overall
        "overall_score": overall_score,
        "overall_max": 10,
        "overall_verdict": overall_verdict,
        "overall_badge": overall_badge,
        
        # Category breakdown
        "categories": {},
        
        # Key stats
        "key_stats": {
            "market_cap_cr": fundamentals.get("ratios", {}).get("market_cap_cr"),
            "current_price": fundamentals.get("ratios", {}).get("current_price"),
            "pe_ratio": fundamentals.get("ratios", {}).get("pe_ratio"),
            "roce": fundamentals.get("ratios", {}).get("roce"),
            "roe": fundamentals.get("ratios", {}).get("roe"),
            "dividend_yield": fundamentals.get("ratios", {}).get("dividend_yield"),
            "eps": fundamentals.get("ratios", {}).get("eps"),
            "debt_cr": fundamentals.get("ratios", {}).get("debt_cr"),
            "promoter_holding": fundamentals.get("ratios", {}).get("promoter_holding"),
            "high_low": fundamentals.get("ratios", {}).get("high_low"),
        },
        
        # Qualitative
        "pros": pros[:5],
        "cons": cons[:5],
        "strengths": strengths,
        "weaknesses": weaknesses,
        
        # Peer comparison
        "peers": fundamentals.get("peers", [])[:5],
        
        # News summary
        "news_summary": {
            "sentiment": sentiment_data.get("reasoning") if sentiment_data else None,
            "key_themes": sentiment_data.get("key_themes") if sentiment_data else None,
            "article_count": sentiment_data.get("article_count", 0) if sentiment_data else 0,
        },
    }
    
    # Add enriched category data with metadata from SCORECARD_CATEGORIES
    for cat_key, cat_scores in categories.items():
        meta = SCORECARD_CATEGORIES[cat_key]
        scorecard["categories"][cat_key] = {
            "name": meta["name"],
            "icon": meta["icon"],
            "description": meta["description"],
            "weight": meta["weight"],
            **cat_scores,
        }
    
    logger.info(f"Generated scorecard for {symbol}: {overall_score}/10 ({overall_verdict})")
    return scorecard


def generate_ai_summary(scorecard: Dict, model_type: str = None) -> str:
    """
    Generate an AI-written investment summary using LLM.
    
    Args:
        scorecard: Generated scorecard dict
        model_type: AI model type override
        
    Returns:
        AI-generated text summary
    """
    from backend.services.rag_sentiment_analyzer import create_sentiment_analyzer
    
    prompt = f"""You are an expert Indian stock market analyst. Write a concise 4-5 sentence investment summary for:

**{scorecard['company_name']}** ({scorecard['symbol']}) — {scorecard.get('sector', 'Unknown Sector')}

Overall Score: {scorecard['overall_score']}/10 — {scorecard['overall_verdict']}

Category Scores:
"""
    for cat_key, cat_data in scorecard["categories"].items():
        prompt += f"- {cat_data['name']}: {cat_data['score']}/10 ({cat_data['verdict']})\n"
    
    stats = scorecard.get("key_stats", {})
    prompt += f"""
Key Stats: PE={stats.get('pe_ratio')}, ROCE={stats.get('roce')}%, ROE={stats.get('roe')}%, Div Yield={stats.get('dividend_yield')}%
Market Cap: ₹{stats.get('market_cap_cr')} Cr

Pros: {', '.join(scorecard.get('pros', ['N/A'])[:3])}
Cons: {', '.join(scorecard.get('cons', ['N/A'])[:3])}

Write a balanced, professional summary covering strengths, risks, and an investment view. Be specific to this company. Keep it under 100 words."""

    try:
        analyzer = create_sentiment_analyzer(model_type=model_type)
        
        if analyzer.model_type == "gemini":
            response = analyzer.client.models.generate_content(
                model=analyzer.model_name,
                contents=prompt,
            )
            return getattr(response, "text", "Summary unavailable.")
        
        elif analyzer.model_type == "openai":
            response = analyzer.client.chat.completions.create(
                model=analyzer.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=200,
            )
            return response.choices[0].message.content
        
        elif analyzer.model_type == "anthropic":
            response = analyzer.client.messages.create(
                model=analyzer.model_name,
                max_tokens=200,
                temperature=0.4,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        
        elif analyzer.model_type == "local" and analyzer.client:
            response = analyzer.client.chat(
                model=analyzer.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.4},
            )
            return response["message"]["content"]
        
        return "AI summary unavailable — no model configured."
    except Exception as e:
        logger.error(f"Failed to generate AI summary: {e}")
        return f"AI summary generation failed. Score: {scorecard['overall_score']}/10 — {scorecard['overall_verdict']}."
