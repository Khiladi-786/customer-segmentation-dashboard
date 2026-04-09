"""
LLM-powered Business Recommendation Engine.
Uses Google Gemini API (free tier) to generate personalized marketing strategies.
Falls back to template-based recommendations if no API key is set.
"""

from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Cache (in-memory) ────────────────────────────────────────────────────────
_CACHE: dict[str, dict] = {}


# ── Template Fallbacks ───────────────────────────────────────────────────────

_TEMPLATES = {
    "Champions": {
        "campaign_name": "VIP Loyalty Accelerator",
        "strategy": "Reward top customers with exclusive early access to new products, personalized thank-you gifts, and an invite-only premium membership tier.",
        "primary_channel": "Email + Direct Mail",
        "offer_type": "Premium Membership & Exclusive Perks",
        "budget_priority": "High",
        "expected_roi": "3.5x–5x",
        "actions": [
            "Launch VIP loyalty program with tiered rewards",
            "Send personalized anniversary offers",
            "Offer co-creation opportunities (beta testing, surveys)",
            "Provide dedicated customer success manager",
        ],
    },
    "Loyal Customers": {
        "campaign_name": "Loyalty Maximizer",
        "strategy": "Deepen engagement with a points-based loyalty scheme, bundle discounts, and referral incentives to grow wallet share and brand advocacy.",
        "primary_channel": "Email + Mobile App",
        "offer_type": "Loyalty Points & Bundle Deals",
        "budget_priority": "High",
        "expected_roi": "2.5x–4x",
        "actions": [
            "Introduce double-points weekend campaigns",
            "Create 'Refer a Friend' bonus program",
            "Personalize product recommendations by purchase history",
            "Offer subscription / auto-replenishment discounts",
        ],
    },
    "Potential Loyalists": {
        "campaign_name": "Growth Nurture Campaign",
        "strategy": "Convert engaged but inconsistent buyers into loyal advocates through behavioral triggers, personalized nudges, and first-purchase milestone rewards.",
        "primary_channel": "Email + Social Retargeting",
        "offer_type": "Milestone Discounts & Personalized Offers",
        "budget_priority": "Medium",
        "expected_roi": "2x–3x",
        "actions": [
            "Trigger 'You might also like' recommendations",
            "Offer 20% off on second purchase",
            "Use browse-abandonment email sequences",
            "Enroll in welcome loyalty program",
        ],
    },
    "At Risk": {
        "campaign_name": "Win-Back Retention Drive",
        "strategy": "Re-engage customers showing reduced activity via time-sensitive win-back offers, survey outreach to understand churn drivers, and service recovery gestures.",
        "primary_channel": "Email + SMS",
        "offer_type": "Win-Back Discount (25–35%)",
        "budget_priority": "Medium",
        "expected_roi": "1.5x–2.5x",
        "actions": [
            "Send 'We miss you' email with 30% discount",
            "Run NPS survey to understand dissatisfaction",
            "Offer free shipping on next purchase",
            "Retarget with display ads featuring bestsellers",
        ],
    },
    "Lost / Inactive": {
        "campaign_name": "Reactivation Blitz",
        "strategy": "Deploy aggressive re-engagement tactics for dormant customers including deep discounts, mystery offers, and final-chance communications.",
        "primary_channel": "Email + Paid Social",
        "offer_type": "Deep Discount (40–50%)",
        "budget_priority": "Low",
        "expected_roi": "1x–1.5x",
        "actions": [
            "Send final 'Last chance' win-back email",
            "Offer 50% discount with 7-day expiry",
            "Use lookalike audience targeting for re-acquisition",
            "Suppress from future high-cost campaigns if no response",
        ],
    },
}

_GENERIC_TEMPLATE = {
    "campaign_name": "Engagement Boost Campaign",
    "strategy": "Drive engagement with personalized content and targeted offers.",
    "primary_channel": "Email + Social Media",
    "offer_type": "Personalized Discount",
    "budget_priority": "Medium",
    "expected_roi": "2x",
    "actions": ["Analyze segment behavior", "Design targeted campaign", "A/B test messaging"],
}


# ── Recommender ──────────────────────────────────────────────────────────────

class LLMRecommender:
    """
    Generates marketing campaign strategies using Gemini API.
    Falls back to curated templates when no API key is available.
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.use_llm = bool(self.api_key and self.api_key != "your_gemini_api_key_here")

    def recommend(
        self,
        cluster_id: int,
        rfm_segment: str,
        cluster_stats: dict,
        force_template: bool = False,
    ) -> dict:
        """
        Generate recommendation for a cluster.
        Args:
            cluster_id:    Cluster number
            rfm_segment:  RFM segment name (e.g., 'Champions')
            cluster_stats: Dict with avg Monetary, Frequency, Recency, CLV_Score, Age, Income
            force_template: Skip LLM and use template
        Returns:
            dict with campaign_name, strategy, channel, offer, actions, etc.
        """
        cache_key = f"{cluster_id}_{rfm_segment}"
        if cache_key in _CACHE:
            return _CACHE[cache_key]

        if not force_template and self.use_llm:
            result = self._call_gemini(cluster_id, rfm_segment, cluster_stats)
        else:
            result = self._template_recommendation(rfm_segment, cluster_stats)

        result["cluster_id"]   = cluster_id
        result["rfm_segment"]  = rfm_segment
        _CACHE[cache_key] = result
        return result

    def recommend_all(
        self, df: pd.DataFrame, cluster_col: str = "Cluster"
    ) -> list[dict]:
        """Generate recommendations for every cluster in df."""
        import pandas as pd
        recommendations = []
        clusters = sorted(df[cluster_col].unique())
        for c in clusters:
            mask = df[cluster_col] == c
            seg = df[mask]
            rfm_seg = (
                seg["RFM_Segment"].mode()[0]
                if "RFM_Segment" in seg.columns and len(seg) > 0
                else "Loyal Customers"
            )
            stats = {
                "avg_monetary":  round(seg.get("Monetary", pd.Series([0])).mean(), 0),
                "avg_recency":   round(seg.get("Recency",  pd.Series([30])).mean(), 1),
                "avg_frequency": round(seg.get("Frequency",pd.Series([1])).mean(), 1),
                "avg_clv":       round(seg.get("CLV_Score",pd.Series([0])).mean(), 0),
                "avg_age":       round(seg.get("Age",      pd.Series([40])).mean(), 0),
                "avg_income":    round(seg.get("Income",   pd.Series([50000])).mean(), 0),
                "size":          int(len(seg)),
            }
            rec = self.recommend(int(c), rfm_seg, stats)
            recommendations.append(rec)
        return recommendations

    # ── Gemini API ───────────────────────────────────────────────────────────

    def _call_gemini(
        self, cluster_id: int, rfm_segment: str, stats: dict
    ) -> dict:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")

            prompt = f"""You are a senior marketing strategist. Generate a data-driven marketing campaign for the following customer segment.

Segment: Cluster {cluster_id} — {rfm_segment}
Key Statistics:
- Average Spend: ${stats.get('avg_monetary', 0):,.0f}
- Average Recency: {stats.get('avg_recency', 30):.0f} days
- Average Frequency: {stats.get('avg_frequency', 1):.1f} purchases
- Average CLV Score: {stats.get('avg_clv', 0):.0f}/1000
- Average Age: {stats.get('avg_age', 40):.0f} years
- Average Income: ${stats.get('avg_income', 50000):,.0f}
- Segment Size: {stats.get('size', 0)} customers

Return ONLY a valid JSON object with these exact keys:
{{
  "campaign_name": "string",
  "strategy": "2-3 sentence strategic rationale",
  "primary_channel": "string (e.g., Email + SMS)",
  "offer_type": "string",
  "budget_priority": "High|Medium|Low",
  "expected_roi": "string (e.g., 3x-5x)",
  "actions": ["action 1", "action 2", "action 3", "action 4"]
}}"""

            response = model.generate_content(prompt)
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except Exception as e:
            logger.warning(f"Gemini API failed: {e} — using template fallback")
            return self._template_recommendation(rfm_segment, stats)

    # ── Template ─────────────────────────────────────────────────────────────

    def _template_recommendation(self, rfm_segment: str, stats: dict) -> dict:
        template = _TEMPLATES.get(rfm_segment, _GENERIC_TEMPLATE).copy()
        # Personalize budget priority based on CLV
        clv = stats.get("avg_clv", 0)
        if clv > 750:
            template["budget_priority"] = "High"
        elif clv > 400:
            template["budget_priority"] = "Medium"
        else:
            template["budget_priority"] = "Low"
        return template
