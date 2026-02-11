# mvp_simulation.py
# Streamlit demo-MVP: симуляция Supercard (FMCG + Аптека) с ускоренным временем
#
# Запуск:
#   pip install streamlit pandas numpy plotly
#   streamlit run mvp_simulation.py
#
# FIXED:
# - Счётчики "транзакций/TPV/вкладов" теперь кумулятивные и НЕ зависят от окна ledger.
# - Ledger остаётся "витриной" последних N строк для FPS.
#
# "КОМИТЕТНЫЙ АПГРЕЙД":
# - Введено агрегированное хранилище time-series (TS) по сим-времени
#   (minute/hour buckets), чтобы графики и KPI не ломались из-за усечения ledger.
# - Графики динамики теперь строятся из TS (стабильно, быстро, масштабируемо).
#
# NOTE:
# - TX строки уже учитывают weight (масштабирование) по суммам.
# - Казначейство (TREASURY) начисляется по dt и активной базе.

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# -----------------------------
# Page config (call once!)
# -----------------------------

st.set_page_config(page_title="Supercard MVP Simulator", layout="wide", page_icon="🟦")


# -----------------------------
# Helpers
# -----------------------------

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0.0 else 0.0


def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    if hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    st.stop()


# -----------------------------
# Domain models
# -----------------------------

@dataclass(frozen=True)
class Merchant:
    merchant_id: str
    name: str
    category: str  # "FMCG" | "PHARMA"
    is_partner: bool
    retailer_discount_rate: float
    retailer_gross_margin: float


@dataclass(frozen=True)
class Segment:
    name: str
    weight: float
    base_partner_affinity: float
    churn_risk: float
    basket_mu: float
    basket_sigma: float


@dataclass
class BrandBudget:
    brand_id: str
    name: str
    category: str
    daily_budget: float
    remaining: float
    avg_brand_discount_rate: float
    promo_sku_share: float

    def reset(self) -> None:
        self.remaining = float(self.daily_budget)


@dataclass(frozen=True)
class PricingDecision:
    discount_bank: float
    discount_retailer: float
    discount_brand: float
    promo_spend: float
    discount_total: float
    explanation: str


# -----------------------------
# Discount engine
# -----------------------------

class DiscountEngine:
    def __init__(self, bank_discount_rate: float):
        self.bank_discount_rate = float(bank_discount_rate)

    def decide(
        self,
        *,
        amount: float,
        merchant: Merchant,
        segment: Segment,
        brand: BrandBudget | None,
        enable_brand: bool,
        churn_boost: float,
        weight: float,
    ) -> PricingDecision:
        if not merchant.is_partner:
            return PricingDecision(0.0, 0.0, 0.0, 0.0, 0.0, "Not a partner merchant")

        b = _clamp01(self.bank_discount_rate + float(churn_boost) * float(segment.churn_risk))
        discount_bank = amount * b

        r = _clamp01(merchant.retailer_discount_rate)
        discount_retailer = amount * r

        discount_brand = 0.0
        promo_spend = 0.0
        expl = [f"Partner; bank={b:.3%}, retailer={r:.3%}"]

        if enable_brand and brand is not None and brand.category == merchant.category:
            promo_spend = amount * _clamp01(brand.promo_sku_share)
            desired_per_txn = promo_spend * _clamp01(brand.avg_brand_discount_rate)
            desired_total = desired_per_txn * weight

            if brand.remaining > 0.0 and desired_total > 0.0:
                spent_total = min(desired_total, brand.remaining)
                brand.remaining -= spent_total
                discount_brand = spent_total / weight
                expl.append(
                    f"brand: s={brand.promo_sku_share:.0%}, m={brand.avg_brand_discount_rate:.0%}, spent={spent_total:,.0f}₽ (w={weight:.1f})"
                )
            else:
                expl.append("brand: budget exhausted")

        total = discount_bank + discount_retailer + discount_brand
        return PricingDecision(
            discount_bank=discount_bank,
            discount_retailer=discount_retailer,
            discount_brand=discount_brand,
            promo_spend=promo_spend,
            discount_total=total,
            explanation="; ".join(expl),
        )


# -----------------------------
# P&L calculators
# -----------------------------

@dataclass(frozen=True)
class PaymentsPnLParams:
    interchange_rate: float
    partner_cpa_rate: float
    processing_cost_rate: float
    brand_fee_rate: float


@dataclass(frozen=True)
class TreasuryPnLParams:
    net_interest_rate_annual: float
    sub_price_monthly: float
    sub_penetration: float


class PnLCalculator:
    def __init__(self, pay: PaymentsPnLParams, tre: TreasuryPnLParams):
        self.pay = pay
        self.tre = tre

    def payments_components(
        self,
        *,
        amount: float,
        promo_spend: float,
        is_partner: bool,
        discount_bank: float,
        weight: float,
    ) -> Dict[str, float]:
        rev_interchange = amount * self.pay.interchange_rate
        rev_cpa = amount * self.pay.partner_cpa_rate if is_partner else 0.0
        rev_brand_fee = promo_spend * self.pay.brand_fee_rate if is_partner else 0.0

        cost_processing = amount * self.pay.processing_cost_rate
        cost_bank_discount = discount_bank

        # weight applies at the end
        rev_interchange *= weight
        rev_cpa *= weight
        rev_brand_fee *= weight
        cost_processing *= weight
        cost_bank_discount *= weight

        rev_pay = rev_interchange + rev_cpa + rev_brand_fee
        cost_pay = cost_processing + cost_bank_discount
        contrib_pay = rev_pay - cost_pay

        return {
            "bank_rev_interchange": rev_interchange,
            "bank_rev_cpa": rev_cpa,
            "bank_rev_brand_fee": rev_brand_fee,
            "bank_cost_processing": cost_processing,
            "bank_cost_bank_discount": cost_bank_discount,
            "bank_payments_rev": rev_pay,
            "bank_payments_cost": cost_pay,
            "bank_payments_contrib": contrib_pay,
        }

    def treasury_accrual(
        self,
        *,
        incremental_balance_total: float,
        activated_users: float,
        virtual_dt_seconds: float,
    ) -> Dict[str, float]:
        seconds_in_month = 30.0 * 24.0 * 3600.0
        frac_month = float(virtual_dt_seconds) / seconds_in_month

        rev_nii = incremental_balance_total * (self.tre.net_interest_rate_annual / 12.0) * frac_month
        rev_sub = activated_users * self.tre.sub_penetration * self.tre.sub_price_monthly * frac_month

        return {
            "bank_rev_nii": float(rev_nii),
            "bank_rev_sub": float(rev_sub),
            "bank_treasury_contrib": float(rev_nii + rev_sub),
        }


# -----------------------------
# World setup
# -----------------------------

def build_world() -> Tuple[List[Merchant], List[Segment], Dict[str, BrandBudget]]:
    merchants = [
        Merchant("m_fmcg_1", "FMCG Chain A", "FMCG", True, retailer_discount_rate=0.030, retailer_gross_margin=0.23),
        Merchant("m_fmcg_2", "FMCG Chain B", "FMCG", True, retailer_discount_rate=0.025, retailer_gross_margin=0.22),
        Merchant("m_ph_1", "Pharmacy Chain A", "PHARMA", True, retailer_discount_rate=0.020, retailer_gross_margin=0.30),
        Merchant("m_ph_2", "Pharmacy Chain B", "PHARMA", True, retailer_discount_rate=0.015, retailer_gross_margin=0.28),
        Merchant("m_np_1", "Non-partner FMCG", "FMCG", False, retailer_discount_rate=0.0, retailer_gross_margin=0.22),
        Merchant("m_np_2", "Non-partner Pharmacy", "PHARMA", False, retailer_discount_rate=0.0, retailer_gross_margin=0.28),
    ]

    segments = [
        Segment("price_sensitive",      0.35, 0.55, 0.35, 7.40, 0.35),
        Segment("family_shopper",       0.25, 0.60, 0.20, 7.65, 0.30),
        Segment("marketplace_addicted", 0.20, 0.40, 0.45, 7.25, 0.40),
        Segment("steady_local",         0.20, 0.70, 0.10, 7.35, 0.28),
    ]
    w = np.array([s.weight for s in segments], dtype=np.float64)
    w /= w.sum()
    segments = [
        Segment(
            segments[i].name,
            float(w[i]),
            segments[i].base_partner_affinity,
            segments[i].churn_risk,
            segments[i].basket_mu,
            segments[i].basket_sigma,
        )
        for i in range(len(segments))
    ]

    brands = {
        "b_fmcg": BrandBudget("b_fmcg", "Brand FMCG Pool", "FMCG", 3_000_000.0, 3_000_000.0, 0.20, 0.10),
        "b_ph":   BrandBudget("b_ph",   "Brand Pharma Pool", "PHARMA", 2_000_000.0, 2_000_000.0, 0.20, 0.10),
    }
    return merchants, segments, brands


# -----------------------------
# Simulation params
# -----------------------------

@dataclass(frozen=True)
class SimParams:
    scenario_name: str
    wall_tick_seconds: float
    virtual_dt_seconds: float
    tx_per_virtual_min: float
    max_tx_per_tick: int

    partner_coverage: float
    discount_sensitivity: float

    base_incremental_rate: float
    incremental_uplift_per_pct: float

    bank_discount_rate: float
    churn_boost: float
    enable_brand: bool

    total_eligible_users: float
    tx_per_user_per_month: float
    activation_base: float
    activation_uplift_per_pct: float
    activation_uplift_cov: float
    churn_monthly: float

    balance_lift_max: float
    lift_disc_at: float
    lift_cov_at: float

    cap_amount: float
    cap_tx_per_virtual_min: float

    # committee TS settings
    ts_bucket_seconds: int
    ts_max_buckets: int


# -----------------------------
# Random picks
# -----------------------------

def pick_segment(rng: np.random.Generator, segments: List[Segment]) -> Segment:
    probs = np.fromiter((s.weight for s in segments), dtype=np.float64)
    idx = int(rng.choice(len(segments), p=probs))
    return segments[idx]


def pick_category(rng: np.random.Generator) -> str:
    return "FMCG" if rng.random() < 0.72 else "PHARMA"


def pick_merchant(rng: np.random.Generator, *, merchants: List[Merchant], category: str, want_partner: bool) -> Merchant:
    pool = [m for m in merchants if m.category == category and m.is_partner == want_partner]
    if not pool:
        pool = [m for m in merchants if m.category == category]
    return pool[int(rng.integers(0, len(pool)))]


def sample_amount(rng: np.random.Generator, segment: Segment, category: str, cap_amount: float) -> float:
    base = float(rng.lognormal(mean=segment.basket_mu, sigma=segment.basket_sigma))
    if category == "PHARMA":
        base *= 1.10
    base = float(np.clip(base, 150.0, cap_amount))
    return base


def is_incremental_txn(
    rng: np.random.Generator,
    *,
    base_incremental_rate: float,
    incremental_uplift_per_pct: float,
    discount_rate_on_txn: float,
) -> bool:
    p = float(base_incremental_rate) + float(discount_rate_on_txn) * 100.0 * float(incremental_uplift_per_pct)
    return rng.random() < _clamp01(p)


# -----------------------------
# Activation + churn + balance lift
# -----------------------------

def update_user_base(
    *,
    params: SimParams,
    activated_users: float,
    tick_total_tx: float,
    avg_discount_rate: float,
    virtual_dt_seconds: float,
) -> Tuple[float, float, float]:
    total = float(params.total_eligible_users)
    act = float(activated_users)

    month_seconds = 30.0 * 24.0 * 3600.0
    churn_m = _clamp01(float(params.churn_monthly))
    hazard = 0.0 if churn_m <= 0 else (-np.log(max(1e-12, 1.0 - churn_m)) / month_seconds)
    churned = act * (1.0 - np.exp(-hazard * float(virtual_dt_seconds)))
    act_after_churn = max(0.0, act - churned)

    tx_per_user_per_sec = float(params.tx_per_user_per_month) / month_seconds
    denom = max(1e-9, tx_per_user_per_sec * float(virtual_dt_seconds))
    approx_users_seen = min(total, float(tick_total_tx) / denom)

    not_act = max(0.0, total - act_after_churn)
    not_act_frac = 0.0 if total <= 0 else (not_act / total)
    nonactivated_seen = approx_users_seen * not_act_frac

    p = float(params.activation_base)
    p += float(params.activation_uplift_per_pct) * (float(avg_discount_rate) * 100.0)
    p += float(params.activation_uplift_cov) * float(params.partner_coverage)
    p = _clamp01(p)

    newly_activated = min(not_act, nonactivated_seen * p)
    act_new = min(total, act_after_churn + newly_activated)

    return act_new, newly_activated, churned


def calc_balance_lift_per_user(params: SimParams, avg_discount_rate: float) -> float:
    disc_factor = _clamp01(float(avg_discount_rate) / max(1e-9, float(params.lift_disc_at)))
    cov_factor = _clamp01(float(params.partner_coverage) / max(1e-9, float(params.lift_cov_at)))
    return float(params.balance_lift_max) * disc_factor * cov_factor


# -----------------------------
# Committee TS (aggregated time-series buckets)
# -----------------------------

def _ts_init():
    if "ts" not in st.session_state:
        st.session_state.ts = {}  # bucket_id -> dict metrics


def _ts_bucket_id(sim_epoch: float, sim_ts: float, bucket_seconds: int) -> int:
    return int((float(sim_ts) - float(sim_epoch)) // float(bucket_seconds))


def _ts_touch_bucket(bucket_id: int):
    ts = st.session_state.ts
    if bucket_id not in ts:
        ts[bucket_id] = {
            "bucket": int(bucket_id),
            "tx": 0.0,
            "tpv": 0.0,
            "disc": 0.0,
            "pay": 0.0,
            "tre": 0.0,
            "tpv_partner": 0.0,
            "tpv_nonpartner": 0.0,
            "tpv_fmcg": 0.0,
            "tpv_pharma": 0.0,
            "ret_disc": 0.0,
            "ret_inc_gp": 0.0,
            "activated": float(st.session_state.activated_users),
            "d_balance_per_user": float(st.session_state.last_balance_lift),
        }


def _ts_update(bucket_id: int, *, add: Dict[str, float]):
    _ts_touch_bucket(bucket_id)
    b = st.session_state.ts[bucket_id]
    for k, v in add.items():
        b[k] = float(b.get(k, 0.0)) + float(v)


def _ts_set_state(bucket_id: int, *, activated: float, d_balance_per_user: float):
    _ts_touch_bucket(bucket_id)
    b = st.session_state.ts[bucket_id]
    b["activated"] = float(activated)
    b["d_balance_per_user"] = float(d_balance_per_user)


def _ts_prune(max_buckets: int):
    ts = st.session_state.ts
    if len(ts) <= max_buckets:
        return
    keys = sorted(ts.keys())
    drop = len(keys) - max_buckets
    for k in keys[:drop]:
        del ts[k]


def ts_dataframe(sim_params: SimParams) -> pd.DataFrame:
    if "ts" not in st.session_state or not st.session_state.ts:
        return pd.DataFrame()
    df_ts = pd.DataFrame(list(st.session_state.ts.values())).sort_values("bucket")
    # convert bucket -> sim-time label (minutes or hours depending on bucket_seconds)
    step = int(sim_params.ts_bucket_seconds)
    df_ts["t"] = df_ts["bucket"] * step
    return df_ts


# -----------------------------
# simulate_tick
# -----------------------------

def simulate_tick(
    rng: np.random.Generator,
    *,
    params: SimParams,
    merchants: List[Merchant],
    segments: List[Segment],
    brands: Dict[str, BrandBudget],
    engine: DiscountEngine,
    pnl: PnLCalculator,
    sim_now_ts: float,
) -> Tuple[List[Dict[str, float | str]], Dict[str, float]]:
    tx_per_vmin = min(float(params.tx_per_virtual_min), float(params.cap_tx_per_virtual_min))
    lam = tx_per_vmin * (float(params.virtual_dt_seconds) / 60.0)
    k_full = int(rng.poisson(lam))
    if k_full <= 0:
        return [], {"tick_total_tx": 0.0, "avg_discount_rate": 0.0, "k_full": 0.0}

    k = min(k_full, int(params.max_tx_per_tick))
    weight = float(k_full) / float(k)

    partner_merchants = [m for m in merchants if m.is_partner]
    avg_r = float(np.mean([m.retailer_discount_rate for m in partner_merchants])) if partner_merchants else 0.0
    exp_brand = 0.0
    if params.enable_brand:
        exp_brand = float(brands["b_fmcg"].promo_sku_share) * float(brands["b_fmcg"].avg_brand_discount_rate)
    expected_partner_disc = _clamp01(float(params.bank_discount_rate) + avg_r + exp_brand)

    rows: List[Dict[str, float | str]] = []

    tick_amount_sum = 0.0
    tick_disc_sum = 0.0
    tick_total_tx = float(k_full)

    for _ in range(k):
        seg = pick_segment(rng, segments)
        category = pick_category(rng)

        base = float(seg.base_partner_affinity)
        cov = float(params.partner_coverage)
        disc_pull = float(params.discount_sensitivity) * expected_partner_disc
        p_partner = _clamp01(0.10 + 0.55 * cov + 0.25 * base + 0.60 * disc_pull)
        want_partner = rng.random() < p_partner

        merchant = pick_merchant(rng, merchants=merchants, category=category, want_partner=want_partner)
        amount = sample_amount(rng, seg, category, cap_amount=float(params.cap_amount))

        brand = brands["b_fmcg"] if category == "FMCG" else brands["b_ph"]

        decision = engine.decide(
            amount=amount,
            merchant=merchant,
            segment=seg,
            brand=brand,
            enable_brand=bool(params.enable_brand),
            churn_boost=float(params.churn_boost),
            weight=weight,
        )

        amount_w = amount * weight
        disc_w = decision.discount_total * weight
        final_amount_w = max(0.0, (amount - decision.discount_total)) * weight

        disc_rate = _safe_div(decision.discount_total, max(1.0, amount))
        inc = is_incremental_txn(
            rng,
            base_incremental_rate=float(params.base_incremental_rate),
            incremental_uplift_per_pct=float(params.incremental_uplift_per_pct),
            discount_rate_on_txn=float(disc_rate),
        )

        pay_parts = pnl.payments_components(
            amount=amount,
            promo_spend=decision.promo_spend,
            is_partner=merchant.is_partner,
            discount_bank=decision.discount_bank,
            weight=weight,
        )

        ret_inc_gp = (amount * merchant.retailer_gross_margin) if inc else 0.0
        ret_inc_gp *= weight
        ret_disc = (decision.discount_retailer * weight) if merchant.is_partner else 0.0

        tick_amount_sum += amount_w
        tick_disc_sum += disc_w

        row = {
            "sim_ts": float(sim_now_ts),
            "row_type": "TX",
            "tx_id": str(uuid.uuid4())[:8],
            "scenario": params.scenario_name,
            "segment": seg.name,
            "category": category,
            "merchant": merchant.name,
            "is_partner": float(merchant.is_partner),
            "weight": float(weight),
            "amount": float(amount_w),
            "final_amount": float(final_amount_w),
            "disc_total": float(disc_w),
            "disc_bank": float(decision.discount_bank * weight),
            "disc_retailer": float(decision.discount_retailer * weight),
            "disc_brand": float(decision.discount_brand * weight),
            "promo_spend": float(decision.promo_spend * weight),
            "disc_rate": float(disc_rate),
            "incremental": float(inc),
            "explain": decision.explanation + ("" if weight <= 1.01 else f" | weight≈{weight:.1f}x"),
            **pay_parts,
            "bank_rev_nii": 0.0,
            "bank_rev_sub": 0.0,
            "bank_treasury_contrib": 0.0,
            "ret_discount_cost": float(ret_disc),
            "ret_inc_gross_profit": float(ret_inc_gp),
        }
        rows.append(row)

    avg_disc_rate_tick = _safe_div(tick_disc_sum, max(1.0, tick_amount_sum))
    return rows, {"tick_total_tx": float(tick_total_tx), "avg_discount_rate": float(avg_disc_rate_tick), "k_full": float(k_full)}


# -----------------------------
# Streamlit state init
# -----------------------------

def _init_state() -> None:
    if "rng_seed" not in st.session_state:
        st.session_state.rng_seed = 42
    if "running" not in st.session_state:
        st.session_state.running = False
    if "ledger" not in st.session_state:
        st.session_state.ledger = []
    if "world" not in st.session_state:
        st.session_state.world = build_world()

    if "sim_epoch" not in st.session_state:
        st.session_state.sim_epoch = time.time()
    if "sim_elapsed" not in st.session_state:
        st.session_state.sim_elapsed = 0.0
    if "sim_day_index" not in st.session_state:
        st.session_state.sim_day_index = 0

    if "activated_users" not in st.session_state:
        st.session_state.activated_users = 0.0
    if "last_new_activated" not in st.session_state:
        st.session_state.last_new_activated = 0.0
    if "last_churned" not in st.session_state:
        st.session_state.last_churned = 0.0
    if "last_balance_lift" not in st.session_state:
        st.session_state.last_balance_lift = 0.0

    if "_do_step" not in st.session_state:
        st.session_state._do_step = False

    # burn tracking
    if "burn" not in st.session_state:
        st.session_state.burn = []

    # cumulative totals (independent of ledger window)
    if "total_tx_est" not in st.session_state:
        st.session_state.total_tx_est = 0.0
    if "total_tpv" not in st.session_state:
        st.session_state.total_tpv = 0.0
    if "total_disc" not in st.session_state:
        st.session_state.total_disc = 0.0
    if "total_bank_pay_contrib" not in st.session_state:
        st.session_state.total_bank_pay_contrib = 0.0
    if "total_bank_tre_contrib" not in st.session_state:
        st.session_state.total_bank_tre_contrib = 0.0
    if "total_ret_disc" not in st.session_state:
        st.session_state.total_ret_disc = 0.0
    if "total_ret_inc_gp" not in st.session_state:
        st.session_state.total_ret_inc_gp = 0.0

    _ts_init()


def reset_sim() -> None:
    st.session_state.running = False
    st.session_state.ledger = []
    st.session_state.sim_elapsed = 0.0
    st.session_state.sim_day_index = 0

    st.session_state.activated_users = 0.0
    st.session_state.last_new_activated = 0.0
    st.session_state.last_churned = 0.0
    st.session_state.last_balance_lift = 0.0

    st.session_state.burn = []

    # reset totals
    st.session_state.total_tx_est = 0.0
    st.session_state.total_tpv = 0.0
    st.session_state.total_disc = 0.0
    st.session_state.total_bank_pay_contrib = 0.0
    st.session_state.total_bank_tre_contrib = 0.0
    st.session_state.total_ret_disc = 0.0
    st.session_state.total_ret_inc_gp = 0.0

    # reset committee TS
    st.session_state.ts = {}

    _, _, brands = st.session_state.world
    for b in brands.values():
        b.reset()


_init_state()


# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.header("⚙️ Симуляция")

scenario = st.sidebar.selectbox("Сценарий", ["Base", "Stress", "Worst"], index=0)

if scenario == "Base":
    tx_per_vmin_default = 80.0
    partner_cov_default = 0.60
    bank_b_default = 0.007
    churn_boost_default = 0.0015
    brand_s_default = 0.12
    brand_m_default = 0.20
    fmcg_budget = 3_000_000.0
    ph_budget = 2_000_000.0
elif scenario == "Stress":
    tx_per_vmin_default = 55.0
    partner_cov_default = 0.50
    bank_b_default = 0.006
    churn_boost_default = 0.0010
    brand_s_default = 0.10
    brand_m_default = 0.16
    fmcg_budget = 2_000_000.0
    ph_budget = 1_500_000.0
else:
    tx_per_vmin_default = 35.0
    partner_cov_default = 0.40
    bank_b_default = 0.005
    churn_boost_default = 0.0008
    brand_s_default = 0.08
    brand_m_default = 0.12
    fmcg_budget = 1_200_000.0
    ph_budget = 900_000.0

st.sidebar.subheader("✅ Validation mode")
validation_mode = st.sidebar.selectbox("Режим", ["Realistic", "Aggressive"], index=0)
if validation_mode == "Realistic":
    cap_amount = 12_000.0
    cap_tx_per_virtual_min = 120.0
else:
    cap_amount = 25_000.0
    cap_tx_per_virtual_min = 400.0
st.sidebar.caption(f"Кэп чека: {cap_amount:,.0f}₽; кэп потока: {cap_tx_per_virtual_min:,.0f} tx/вирт-мин")

st.sidebar.subheader("⏩ Масштаб времени")
enable_fast_time = st.sidebar.toggle("Ускорять время", value=True)
wall_tick_seconds = st.sidebar.slider("Шаг обновления UI (сек)", 0.2, 2.0, 0.6, 0.1)

MONTH_SECONDS = 30.0 * 24.0 * 3600.0
target_month_seconds = st.sidebar.slider("1 месяц (30д) за … сек", 5.0, 120.0, 30.0, 1.0)
speedup = (MONTH_SECONDS / target_month_seconds) if enable_fast_time else 1.0
virtual_dt_seconds = float(wall_tick_seconds) * float(speedup)
st.sidebar.caption(f"Ускорение: ~{speedup:,.0f}×, виртуальный dt: {virtual_dt_seconds:,.0f} сек/тик")

st.sidebar.subheader("Поток")
tx_per_virtual_min = st.sidebar.slider("Транзакций / виртуальную минуту", 1.0, 400.0, float(tx_per_vmin_default), 1.0)
max_tx_per_tick = st.sidebar.slider("Лимит транзакций на тик (для FPS)", 200, 8000, 2500, 100)

st.sidebar.subheader("Комитетный time-series")
ts_bucket_seconds = st.sidebar.selectbox("TS bucket", [60, 300, 900, 3600], index=1)  # 5 min default
ts_max_buckets = st.sidebar.slider("TS max buckets", 200, 2000, 800, 50)
st.sidebar.caption("TS хранит агрегаты по bucket и не зависит от ledger.")

st.sidebar.subheader("Поведение")
partner_coverage = st.sidebar.slider("Доля трат в партнёрах (p)", 0.10, 0.95, float(partner_cov_default), 0.05)
discount_sensitivity = st.sidebar.slider("Чувствительность выбора к скидке", 0.0, 3.0, 1.0, 0.1)

st.sidebar.subheader("Слои скидки")
bank_discount_rate = st.sidebar.slider("Банк (b)", 0.002, 0.015, float(bank_b_default), 0.0005, format="%.4f")
churn_boost = st.sidebar.slider("Churn-boost к b (макс добавка)", 0.0, 0.005, float(churn_boost_default), 0.0001, format="%.4f")
enable_brand = st.sidebar.toggle("Включить брендовый слой", value=True)
promo_sku_share = st.sidebar.slider("Проникновение promo-SKU (s)", 0.00, 0.60, float(brand_s_default), 0.01)
avg_brand_discount_rate = st.sidebar.slider("Скидка бренда на promo-SKU (m)", 0.00, 0.25, float(brand_m_default), 0.01)

st.sidebar.subheader("Инкрементальность ритейла (proxy)")
base_inc = st.sidebar.slider("Базовая вероятность", 0.00, 0.40, 0.08, 0.01)
uplift_per_pct = st.sidebar.slider("Добавка за 1% скидки", 0.00, 0.08, 0.020, 0.002, format="%.3f")

st.sidebar.subheader("Банк Payments P&L (toy)")
interchange_rate = st.sidebar.slider("Interchange", 0.002, 0.020, 0.011, 0.0005, format="%.4f")
partner_cpa_rate = st.sidebar.slider("CPA партнёра", 0.000, 0.020, 0.012, 0.0005, format="%.4f")
processing_cost_rate = st.sidebar.slider("Процессинг (rate)", 0.0005, 0.010, 0.003, 0.0005, format="%.4f")
brand_fee_rate = st.sidebar.slider("Комиссия банка с promo-spend", 0.0, 0.020, 0.005, 0.001, format="%.3f")

st.sidebar.subheader("Treasury/Value (инкрементальные остатки)")
key_rate = st.sidebar.slider("Ключевая ставка (proxy)", 0.05, 0.30, 0.165, 0.005, format="%.3f")
transfer_spread = st.sidebar.slider("Transfer spread", 0.00, 0.10, 0.02, 0.005, format="%.3f")
net_interest_rate_annual = max(0.0, float(key_rate) - float(transfer_spread))

sub_price = st.sidebar.number_input("Подписка (₽/мес)", min_value=0, max_value=2_000, value=299, step=10)
sub_pen = st.sidebar.slider("Проникновение подписки (на активных)", 0.0, 0.6, 0.15, 0.01)

st.sidebar.subheader("Активация пользователей (toy)")
total_eligible_users = st.sidebar.number_input("Потенциал (eligible users)", min_value=50_000, max_value=100_000_000, value=5_000_000, step=50_000)
tx_per_user_per_month = st.sidebar.slider("Транзакций/польз/мес (в категории)", 1.0, 60.0, 20.0, 1.0)

activation_base = st.sidebar.slider("База активации (вероятность)", 0.0, 0.30, 0.03, 0.01)
activation_uplift_per_pct = st.sidebar.slider("Добавка активации за 1% скидки", 0.0, 0.03, 0.006, 0.001, format="%.3f")
activation_uplift_cov = st.sidebar.slider("Добавка активации за покрытие (p)", 0.0, 0.30, 0.06, 0.01)
churn_monthly = st.sidebar.slider("Отток активных / мес", 0.0, 0.25, 0.03, 0.01)

st.sidebar.subheader("Инкрементальный остаток (Δbalance) на активного")
balance_lift_max = st.sidebar.number_input("Макс Δbalance (₽)", min_value=0, max_value=200_000, value=12_000, step=1_000)
lift_disc_at = st.sidebar.slider("Скидка, где lift насыщается", 0.005, 0.10, 0.040, 0.005, format="%.3f")
lift_cov_at = st.sidebar.slider("Покрытие, где lift насыщается", 0.10, 0.95, 0.60, 0.05)

st.sidebar.subheader("Бюджеты брендов (в день, ₽)")
fmcg_budget_in = st.sidebar.number_input("FMCG бюджет", min_value=0.0, max_value=50_000_000.0, value=float(fmcg_budget), step=100_000.0)
ph_budget_in = st.sidebar.number_input("Аптека бюджет", min_value=0.0, max_value=50_000_000.0, value=float(ph_budget), step=100_000.0)

st.sidebar.divider()
colA, colB, colC, colD = st.sidebar.columns(4)
with colA:
    if st.button("▶", use_container_width=True):
        st.session_state.running = True
with colB:
    if st.button("⏸", use_container_width=True):
        st.session_state.running = False
with colC:
    if st.button("⏭", use_container_width=True):
        st.session_state.running = False
        st.session_state._do_step = True
with colD:
    if st.button("↺", use_container_width=True):
        reset_sim()


# -----------------------------
# Build world + apply brand params/budgets
# -----------------------------

merchants, segments, brands = st.session_state.world

for bid in ("b_fmcg", "b_ph"):
    brands[bid].promo_sku_share = float(promo_sku_share)
    brands[bid].avg_brand_discount_rate = float(avg_brand_discount_rate)

for bid, new_budget in [("b_fmcg", float(fmcg_budget_in)), ("b_ph", float(ph_budget_in))]:
    b = brands[bid]
    if b.daily_budget != new_budget:
        spent = max(0.0, b.daily_budget - b.remaining)
        b.daily_budget = new_budget
        b.remaining = max(0.0, new_budget - spent)

engine = DiscountEngine(bank_discount_rate=float(bank_discount_rate))
pnl = PnLCalculator(
    pay=PaymentsPnLParams(float(interchange_rate), float(partner_cpa_rate), float(processing_cost_rate), float(brand_fee_rate)),
    tre=TreasuryPnLParams(float(net_interest_rate_annual), float(sub_price), float(sub_pen)),
)

sim_params = SimParams(
    scenario_name=f"{scenario}/{validation_mode}",
    wall_tick_seconds=float(wall_tick_seconds),
    virtual_dt_seconds=float(virtual_dt_seconds),
    tx_per_virtual_min=float(tx_per_virtual_min),
    max_tx_per_tick=int(max_tx_per_tick),
    partner_coverage=float(partner_coverage),
    discount_sensitivity=float(discount_sensitivity),
    base_incremental_rate=float(base_inc),
    incremental_uplift_per_pct=float(uplift_per_pct),
    bank_discount_rate=float(bank_discount_rate),
    churn_boost=float(churn_boost),
    enable_brand=bool(enable_brand),
    total_eligible_users=float(total_eligible_users),
    tx_per_user_per_month=float(tx_per_user_per_month),
    activation_base=float(activation_base),
    activation_uplift_per_pct=float(activation_uplift_per_pct),
    activation_uplift_cov=float(activation_uplift_cov),
    churn_monthly=float(churn_monthly),
    balance_lift_max=float(balance_lift_max),
    lift_disc_at=float(lift_disc_at),
    lift_cov_at=float(lift_cov_at),
    cap_amount=float(cap_amount),
    cap_tx_per_virtual_min=float(cap_tx_per_virtual_min),
    ts_bucket_seconds=int(ts_bucket_seconds),
    ts_max_buckets=int(ts_max_buckets),
)


# -----------------------------
# Sim time + daily budget roll + burn tracking
# -----------------------------

def _advance_virtual_time(virtual_dt: float) -> float:
    st.session_state.sim_elapsed = float(st.session_state.sim_elapsed) + float(virtual_dt)
    return float(st.session_state.sim_epoch) + float(st.session_state.sim_elapsed)


def _record_burn(day_index: int, fmcg_spent: float, ph_spent: float) -> None:
    st.session_state.burn.append({
        "day": int(day_index),
        "fmcg_spent": float(fmcg_spent),
        "pharma_spent": float(ph_spent),
        "fmcg_remaining": float(brands["b_fmcg"].remaining),
        "pharma_remaining": float(brands["b_ph"].remaining),
    })


def _maybe_roll_day(sim_ts: float) -> None:
    day_index = int((sim_ts - st.session_state.sim_epoch) // (24.0 * 3600.0))
    if day_index != st.session_state.sim_day_index:
        prev_fmcg_spent = max(0.0, brands["b_fmcg"].daily_budget - brands["b_fmcg"].remaining)
        prev_ph_spent = max(0.0, brands["b_ph"].daily_budget - brands["b_ph"].remaining)
        _record_burn(st.session_state.sim_day_index, prev_fmcg_spent, prev_ph_spent)

        st.session_state.sim_day_index = day_index
        for b in brands.values():
            b.reset()


# -----------------------------
# Title
# -----------------------------

st.title("🟦 Supercard MVP: симуляция FMCG + Аптека")
st.caption(
    "Поток транзакций → скидка слоями (банк+ритейл+бренд) → ledger (витрина) + committee TS (агрегаты) → KPI. "
    "NII/подписка начисляются только на активированных пользователей и только на Δbalance."
)


# -----------------------------
# Tick execution
# -----------------------------

ledger_len_before = len(st.session_state.ledger)
rng = np.random.default_rng(int(st.session_state.rng_seed) + ledger_len_before)

do_tick = bool(st.session_state.running) or bool(st.session_state._do_step)
tick_stats = {"tick_total_tx": 0.0, "avg_discount_rate": 0.0, "k_full": 0.0}

if do_tick:
    sim_ts = _advance_virtual_time(sim_params.virtual_dt_seconds)
    _maybe_roll_day(sim_ts)

    tx_rows, tick_stats = simulate_tick(
        rng,
        params=sim_params,
        merchants=merchants,
        segments=segments,
        brands=brands,
        engine=engine,
        pnl=pnl,
        sim_now_ts=sim_ts,
    )

    act_prev = float(st.session_state.activated_users)
    act_new, newly_activated, churned = update_user_base(
        params=sim_params,
        activated_users=act_prev,
        tick_total_tx=float(tick_stats["tick_total_tx"]),
        avg_discount_rate=float(tick_stats["avg_discount_rate"]),
        virtual_dt_seconds=float(sim_params.virtual_dt_seconds),
    )
    st.session_state.activated_users = float(act_new)
    st.session_state.last_new_activated = float(newly_activated)
    st.session_state.last_churned = float(churned)

    lift_per_user = calc_balance_lift_per_user(sim_params, float(tick_stats["avg_discount_rate"]))
    st.session_state.last_balance_lift = float(lift_per_user)
    incremental_balance_total = float(st.session_state.activated_users) * float(lift_per_user)

    tre = pnl.treasury_accrual(
        incremental_balance_total=incremental_balance_total,
        activated_users=float(st.session_state.activated_users),
        virtual_dt_seconds=float(sim_params.virtual_dt_seconds),
    )

    treasury_row = {
        "sim_ts": float(sim_ts),
        "row_type": "TREASURY",
        "tx_id": "TREASURY",
        "scenario": sim_params.scenario_name,
        "segment": "—",
        "category": "TREASURY",
        "merchant": "—",
        "is_partner": 0.0,
        "weight": 0.0,
        "amount": 0.0,
        "final_amount": 0.0,
        "disc_total": 0.0,
        "disc_bank": 0.0,
        "disc_retailer": 0.0,
        "disc_brand": 0.0,
        "promo_spend": 0.0,
        "disc_rate": 0.0,
        "incremental": 0.0,
        "explain": f"Activated={st.session_state.activated_users:,.0f}, Δbal/user={lift_per_user:,.0f}₽",
        "bank_rev_interchange": 0.0,
        "bank_rev_cpa": 0.0,
        "bank_rev_brand_fee": 0.0,
        "bank_cost_processing": 0.0,
        "bank_cost_bank_discount": 0.0,
        "bank_payments_rev": 0.0,
        "bank_payments_cost": 0.0,
        "bank_payments_contrib": 0.0,
        "bank_rev_nii": float(tre["bank_rev_nii"]),
        "bank_rev_sub": float(tre["bank_rev_sub"]),
        "bank_treasury_contrib": float(tre["bank_treasury_contrib"]),
        "ret_discount_cost": 0.0,
        "ret_inc_gross_profit": 0.0,
    }

    # --- update cumulative totals (independent of ledger cap) ---
    st.session_state.total_tx_est += float(tick_stats.get("tick_total_tx", 0.0))

    if tx_rows:
        st.session_state.total_tpv += float(sum(r["amount"] for r in tx_rows))
        st.session_state.total_disc += float(sum(r["disc_total"] for r in tx_rows))
        st.session_state.total_bank_pay_contrib += float(sum(r["bank_payments_contrib"] for r in tx_rows))
        st.session_state.total_ret_disc += float(sum(r["ret_discount_cost"] for r in tx_rows))
        st.session_state.total_ret_inc_gp += float(sum(r["ret_inc_gross_profit"] for r in tx_rows))

    st.session_state.total_bank_tre_contrib += float(tre["bank_treasury_contrib"])

    # --- committee TS update (bucket aggregates) ---
    bucket_id = _ts_bucket_id(float(st.session_state.sim_epoch), float(sim_ts), int(sim_params.ts_bucket_seconds))

    # partner/nonpartner/category splits per tick (use tx_rows sums: already weighted)
    if tx_rows:
        tpv_tick = float(sum(r["amount"] for r in tx_rows))
        disc_tick = float(sum(r["disc_total"] for r in tx_rows))
        pay_tick = float(sum(r["bank_payments_contrib"] for r in tx_rows))
        ret_disc_tick = float(sum(r["ret_discount_cost"] for r in tx_rows))
        ret_inc_tick = float(sum(r["ret_inc_gross_profit"] for r in tx_rows))
        tpv_partner_tick = float(sum(r["amount"] for r in tx_rows if r["is_partner"] > 0.5))
        tpv_fmcg_tick = float(sum(r["amount"] for r in tx_rows if r["category"] == "FMCG"))
        tpv_ph_tick = float(sum(r["amount"] for r in tx_rows if r["category"] == "PHARMA"))
    else:
        tpv_tick = disc_tick = pay_tick = ret_disc_tick = ret_inc_tick = 0.0
        tpv_partner_tick = tpv_fmcg_tick = tpv_ph_tick = 0.0

    _ts_update(bucket_id, add={
        "tx": float(tick_stats.get("tick_total_tx", 0.0)),
        "tpv": tpv_tick,
        "disc": disc_tick,
        "pay": pay_tick,
        "tre": float(tre["bank_treasury_contrib"]),
        "tpv_partner": tpv_partner_tick,
        "tpv_nonpartner": (tpv_tick - tpv_partner_tick),
        "tpv_fmcg": tpv_fmcg_tick,
        "tpv_pharma": tpv_ph_tick,
        "ret_disc": ret_disc_tick,
        "ret_inc_gp": ret_inc_tick,
    })
    _ts_set_state(bucket_id, activated=float(st.session_state.activated_users), d_balance_per_user=float(lift_per_user))
    _ts_prune(int(sim_params.ts_max_buckets))

    # --- ledger update (window) ---
    if tx_rows:
        st.session_state.ledger.extend(tx_rows)
    st.session_state.ledger.append(treasury_row)

    st.session_state._do_step = False


# -----------------------------
# Ledger cap (vitirna)
# -----------------------------

MAX_ROWS = 60_000
if len(st.session_state.ledger) > MAX_ROWS:
    st.session_state.ledger = st.session_state.ledger[-MAX_ROWS:]

df = pd.DataFrame(st.session_state.ledger) if st.session_state.ledger else pd.DataFrame()


# -----------------------------
# Top headline
# -----------------------------

sim_now = float(st.session_state.sim_epoch) + float(st.session_state.sim_elapsed)
sim_day = int((sim_now - float(st.session_state.sim_epoch)) // (24.0 * 3600.0)) + 1
sim_time_in_day = (sim_now - float(st.session_state.sim_epoch)) % (24.0 * 3600.0)
hh = int(sim_time_in_day // 3600)
mm = int((sim_time_in_day % 3600) // 60)
ss = int(sim_time_in_day % 60)

# window tx (for reference only)
tx_window = float(df.loc[df["row_type"] == "TX", "weight"].sum()) if not df.empty else 0.0
tx_total = float(st.session_state.total_tx_est)

st.divider()
t1, t2, t3, t4, t5, t6 = st.columns(6)
with t1:
    st.metric("Транзакций (total)", f"{tx_total:,.0f}", delta=f"окно: {tx_window:,.0f}")
with t2:
    st.metric("Строк в ledger", f"{len(st.session_state.ledger):,}")
with t3:
    st.metric("Статус", "RUNNING ✅" if st.session_state.running else "PAUSED ⏸️")
with t4:
    st.metric("Сим. время", f"Day {sim_day} {hh:02d}:{mm:02d}:{ss:02d}")
with t5:
    st.metric("Активировано (users)", f"{st.session_state.activated_users:,.0f}",
              delta=f"+{st.session_state.last_new_activated:,.0f} / −{st.session_state.last_churned:,.0f}")
with t6:
    st.metric("Δbalance / активного", f"{st.session_state.last_balance_lift:,.0f} ₽")

b1, b2 = st.columns(2)
with b1:
    st.metric("Бюджет FMCG (остаток)", f"{brands['b_fmcg'].remaining:,.0f} ₽")
with b2:
    st.metric("Бюджет Аптека (остаток)", f"{brands['b_ph'].remaining:,.0f} ₽")

st.divider()

if df.empty and not do_tick:
    st.info("Нажми Start, чтобы начать симуляцию.")
    if st.session_state.running:
        time.sleep(float(sim_params.wall_tick_seconds))
        _rerun()
    st.stop()


# -----------------------------
# KPI computations (use cumulative totals)
# -----------------------------

avg_disc_rate_total = _safe_div(float(st.session_state.total_disc), max(1.0, float(st.session_state.total_tpv)))
bank_total_contrib = float(st.session_state.total_bank_pay_contrib) + float(st.session_state.total_bank_tre_contrib)
tre_share_total = _safe_div(float(st.session_state.total_bank_tre_contrib), max(1.0, bank_total_contrib))
ret_roi_total = _safe_div(float(st.session_state.total_ret_inc_gp), max(1.0, float(st.session_state.total_ret_disc)))

st.subheader("🏦 Банк | 🛒 Ритейлер (итоги, кумулятивно)")
left, right = st.columns([1, 1])
with left:
    a, b, c = st.columns(3)
    with a:
        st.metric("TPV (total)", f"{st.session_state.total_tpv:,.0f} ₽")
        st.metric("Средняя скидка (total)", f"{avg_disc_rate_total*100:.2f}%")
    with b:
        st.metric("Payments contrib (total)", f"{st.session_state.total_bank_pay_contrib:,.0f} ₽")
        st.metric("Treasury contrib (total)", f"{st.session_state.total_bank_tre_contrib:,.0f} ₽")
    with c:
        st.metric("Total contrib (total)", f"{bank_total_contrib:,.0f} ₽")
        st.metric("Treasury share (total)", f"{tre_share_total*100:.1f}%")
with right:
    a, b, c = st.columns(3)
    with a:
        st.metric("Скидка ритейлера (total)", f"{st.session_state.total_ret_disc:,.0f} ₽")
    with b:
        st.metric("Инкрем. GP (total)", f"{st.session_state.total_ret_inc_gp:,.0f} ₽")
    with c:
        st.metric("ROI (proxy, total)", f"{ret_roi_total:.2f}x")

st.caption("Итоги не зависят от ledger. Ledger — только витрина последних строк. Графики ниже строятся из committee TS.")

st.divider()


# -----------------------------
# Committee TS charts (stable & fast)
# -----------------------------

df_ts = ts_dataframe(sim_params)
if df_ts.empty:
    st.info("TS пока пустой. Нажми Start и дай симуляции сделать хотя бы пару тиков.")
else:
    st.subheader("📈 Комитетные графики (TS агрегаты)")

    # Build time axis label
    step = int(sim_params.ts_bucket_seconds)
    if step >= 3600:
        df_ts["t_label"] = (df_ts["t"] / 3600.0).round(2).astype(str) + "h"
    elif step >= 60:
        df_ts["t_label"] = (df_ts["t"] / 60.0).round(1).astype(str) + "m"
    else:
        df_ts["t_label"] = df_ts["t"].astype(int).astype(str) + "s"

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(df_ts, x="t_label", y=["tpv", "disc"], markers=False)
        fig.update_layout(xaxis_title=f"time ({sim_params.ts_bucket_seconds}s buckets)", yaxis_title="₽ per bucket")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(df_ts, x="t_label", y=["pay", "tre"], markers=False)
        fig.update_layout(xaxis_title=f"time ({sim_params.ts_bucket_seconds}s buckets)", yaxis_title="₽ per bucket")
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.line(df_ts, x="t_label", y=["tx"], markers=False)
        fig.update_layout(xaxis_title=f"time ({sim_params.ts_bucket_seconds}s buckets)", yaxis_title="TX per bucket")
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.line(df_ts, x="t_label", y=["activated"], markers=False)
        fig.update_layout(xaxis_title=f"time ({sim_params.ts_bucket_seconds}s buckets)", yaxis_title="Activated users")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Partner vs Non-partner (TS)")
    fig = px.line(df_ts, x="t_label", y=["tpv_partner", "tpv_nonpartner"], markers=False)
    fig.update_layout(xaxis_title=f"time ({sim_params.ts_bucket_seconds}s buckets)", yaxis_title="₽ per bucket")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧺 FMCG vs PHARMA (TS)")
    fig = px.line(df_ts, x="t_label", y=["tpv_fmcg", "tpv_pharma"], markers=False)
    fig.update_layout(xaxis_title=f"time ({sim_params.ts_bucket_seconds}s buckets)", yaxis_title="₽ per bucket")
    st.plotly_chart(fig, use_container_width=True)

st.divider()


# -----------------------------
# Burn-rate brands by day
# -----------------------------

st.subheader("🔥 Burn-rate брендов по дням")
burn_df = pd.DataFrame(st.session_state.burn)
# append current day partial
cur_fmcg_spent = max(0.0, brands["b_fmcg"].daily_budget - brands["b_fmcg"].remaining)
cur_ph_spent = max(0.0, brands["b_ph"].daily_budget - brands["b_ph"].remaining)
burn_df = pd.concat([burn_df, pd.DataFrame([{
    "day": int(st.session_state.sim_day_index),
    "fmcg_spent": float(cur_fmcg_spent),
    "pharma_spent": float(cur_ph_spent),
    "fmcg_remaining": float(brands["b_fmcg"].remaining),
    "pharma_remaining": float(brands["b_ph"].remaining),
}])], ignore_index=True)

if len(burn_df) > 0:
    c1, c2 = st.columns(2)
    with c1:
        fig_burn = px.line(burn_df.sort_values("day"), x="day", y=["fmcg_spent", "pharma_spent"], markers=True)
        fig_burn.update_layout(xaxis_title="Day", yaxis_title="₽ spent (per day)")
        st.plotly_chart(fig_burn, use_container_width=True)
    with c2:
        fig_rem = px.line(burn_df.sort_values("day"), x="day", y=["fmcg_remaining", "pharma_remaining"], markers=True)
        fig_rem.update_layout(xaxis_title="Day", yaxis_title="₽ remaining (end of day / current)")
        st.plotly_chart(fig_rem, use_container_width=True)
else:
    st.caption("Пока нет дневных точек burn-rate (пройдут сутки сим-времени — появятся).")

st.divider()


# -----------------------------
# Live feed (window)
# -----------------------------

c1, c2 = st.columns([1, 2])
with c1:
    st.subheader("🧾 Live feed (последние 25 TX)")
    if not df.empty:
        tx = df[df["row_type"] == "TX"]
        if not tx.empty:
            tail = tx.tail(25).copy()
            tail["line"] = tail.apply(
                lambda r: (
                    f"{r['tx_id']} | {r['category']} | {'P' if r['is_partner']>0.5 else 'NP'} | {r['merchant']} | "
                    f"{r['amount']:.0f}→{r['final_amount']:.0f} ₽ (−{r['disc_total']:.0f}) "
                    f"| bank {r['disc_bank']:.0f}, ret {r['disc_retailer']:.0f}, brand {r['disc_brand']:.0f}"
                ),
                axis=1,
            )
            for s in reversed(tail["line"].tolist()):
                st.write(s)
        else:
            st.caption("Пока нет TX строк.")
    else:
        st.caption("Ledger пуст.")

with c2:
    st.subheader("🧪 Диагностика окна ledger")
    st.caption("Окно усечено до 60k строк. Счётчики и графики не зависят от этого.")
    # show window-only bar decomposition for intuition
    if not df.empty:
        tx = df[df["row_type"] == "TX"]
        if not tx.empty:
            parts = {
                "Interchange": float(tx["bank_rev_interchange"].sum()),
                "CPA партнёров": float(tx["bank_rev_cpa"].sum()),
                "Brand fee": float(tx["bank_rev_brand_fee"].sum()),
                "Processing (cost)": -float(tx["bank_cost_processing"].sum()),
                "Bank discount (cost)": -float(tx["bank_cost_bank_discount"].sum()),
            }
            p_df = pd.DataFrame({"Item": list(parts.keys()), "Value": list(parts.values())})
            fig = px.bar(p_df, x="Item", y="Value", text_auto=".2s")
            fig.update_layout(xaxis_title="", yaxis_title="₽ in window")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Нет TX в окне.")

st.divider()

with st.expander("🔍 Ledger (витрина)"):
    st.caption("Показываем последние 2 000 строк. TX суммы уже учитывают weight.")
    if not df.empty:
        st.dataframe(df.tail(2000), use_container_width=True, height=420)
    else:
        st.write("—")


# -----------------------------
# Auto-refresh
# -----------------------------

if st.session_state.running:
    time.sleep(float(sim_params.wall_tick_seconds))
    _rerun()
