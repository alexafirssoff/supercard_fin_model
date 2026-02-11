import copy
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px

# --- 1) CONFIG ---
st.set_page_config(layout="wide", page_title="Финмодель 'Синий Ценник' PRO", page_icon="🏦")


# --- 2) MODEL ---
class FinancialModel:
    def __init__(self, params: dict):
        # params: {key: {"value": ..., "desc": ...}}
        self.p = {k: float(v["value"]) for k, v in params.items()}

    def calculate_unit_economics(self) -> dict:
        """Экономика на 1 клиента в месяц."""
        p = self.p

        # --- FEE & COMMISSION REVENUE ---
        rev_interchange = p["tpv_per_user"] * p["interchange_rate"]

        partner_vol = p["tpv_per_user"] * p["partner_spend_share"]
        rev_partner_cpa = partner_vol * p["partner_cpa_rate"]

        sku_vol = partner_vol * p["sku_share_in_check"]
        rev_sku = sku_vol * p["sku_funding_rate"]

        rev_sub = p["sub_price"] * p["sub_penetration"]

        # --- NII (Net Interest Income) ---
        # net_interest_rate = max(0, key_rate - ftp_spread)
        net_interest_rate = max(0.0, p["key_rate"] - p["transfer_price_spread"])
        rev_float_monthly = (p["avg_balance"] * net_interest_rate) / 12.0

        total_revenue = rev_interchange + rev_partner_cpa + rev_sku + rev_sub + rev_float_monthly

        # --- COSTS ---
        cost_processing = p["tpv_per_user"] * p["processing_cost_rate"]

        contribution_margin = total_revenue - cost_processing
        margin_percent = (contribution_margin / total_revenue) * 100.0 if total_revenue > 0 else 0.0

        return {
            "rev_interchange": rev_interchange,
            "rev_partner_cpa": rev_partner_cpa,
            "rev_sku": rev_sku,
            "rev_sub": rev_sub,
            "rev_float": rev_float_monthly,
            "total_revenue": total_revenue,
            "cost_processing": cost_processing,
            "contribution_margin": contribution_margin,
            "margin_percent": margin_percent,
        }

    def calculate_pl_year(self, unit_eco: dict) -> dict:
        """P&L (Year 1) + LTV NPV (36m) with nominal discounting and inflation growth."""
        p = self.p
        users = p["active_users_year1"]

        gross_revenue = unit_eco["total_revenue"] * users * 12.0
        operating_contribution = unit_eco["contribution_margin"] * users * 12.0

        ebitda = operating_contribution - p["opex_year"]

        # CAC amortisation (straight-line 3y)
        cac_amortization = (users * p["cac"]) / 3.0
        ebt = ebitda - cac_amortization

        tax = ebt * p["tax_rate"] if ebt > 0 else 0.0
        net_income = ebt - tax

        # --- LTV NPV (36 months) ---
        # monthly_discount_rate = (1 + key_rate)^(1/12) - 1
        monthly_discount_rate = (1.0 + p["key_rate"]) ** (1.0 / 12.0) - 1.0
        # monthly_growth_rate = (1 + inflation_rate)^(1/12) - 1
        monthly_growth_rate = (1.0 + p["inflation_rate"]) ** (1.0 / 12.0) - 1.0

        base_margin = unit_eco["contribution_margin"]

        cash_flows = np.empty(36, dtype=np.float64)
        # survival_rate_i = (1 - churn)^i
        churn = min(max(p["churn_rate"], 0.0), 0.999999)

        surv = 1.0
        growth = 1.0
        for i in range(36):
            if i > 0:
                surv *= (1.0 - churn)
                growth *= (1.0 + monthly_growth_rate)
            cash_flows[i] = base_margin * growth * surv

        # NPV expects CF0 at t=0. We use CF0=0, CF1..CF36
        ltv_npv = float(npf.npv(monthly_discount_rate, np.concatenate(([0.0], cash_flows))))
        ltv_cac = (ltv_npv / p["cac"]) if p["cac"] > 0 else 0.0

        # Simple payback (months): CAC / monthly contribution (guarded)
        payback_months = (p["cac"] / base_margin) if base_margin > 0 else float("inf")

        return {
            "gross_revenue": gross_revenue,
            "operating_contribution": operating_contribution,
            "opex": p["opex_year"],
            "ebitda": ebitda,
            "ebt": ebt,
            "tax": tax,
            "net_income": net_income,
            "ltv_npv": ltv_npv,
            "ltv_cac": ltv_cac,
            "payback_months": payback_months,
        }


# --- 3) DEFAULT PARAMS ---
DEFAULT_PARAMS = {
    "active_users_year1": {"desc": "Активная база (MAU)", "value": 1_000_000},
    "tpv_per_user": {"desc": "Оборот (TPV) руб/мес", "value": 35_000},
    "avg_balance": {"desc": "Средний остаток на карте (руб)", "value": 15_000},

    "partner_spend_share": {"desc": "Доля партнерских трат", "value": 0.45},
    "sku_share_in_check": {"desc": "Доля SKU-промо в чеке (в партнерском обороте)", "value": 0.05},

    "sub_penetration": {"desc": "Проникновение подписки", "value": 0.15},
    "sub_price": {"desc": "Цена подписки руб/мес", "value": 299},

    "interchange_rate": {"desc": "Interchange rate", "value": 0.011},
    "partner_cpa_rate": {"desc": "Комиссия партнера (CPA)", "value": 0.012},
    "sku_funding_rate": {"desc": "Комиссия брендов (SKU funding)", "value": 0.10},

    "processing_cost_rate": {"desc": "Кост процессинга (в % от TPV)", "value": 0.003},

    "key_rate": {"desc": "Ключевая ставка ЦБ", "value": 0.165},
    "transfer_price_spread": {"desc": "Трансфертный спред (FTP spread)", "value": 0.02},

    "cac": {"desc": "CAC (привлечение, ₽)", "value": 2500},
    "churn_rate": {"desc": "Отток (Churn, мес)", "value": 0.01},

    "tax_rate": {"desc": "Налог на прибыль", "value": 0.25},
    "opex_year": {"desc": "Годовой OPEX (включая все расходы), ₽", "value": 2_800_000_000},

    "inflation_rate": {"desc": "Инфляция (рост чека/потока, год)", "value": 0.08},
}

# --- 4) SCENARIOS ---
SCENARIOS = {
    "base": {
        "label": "Base case",
        "overrides": {}
    },
    "worst": {
        "label": "Worst case",
        "overrides": {
            "active_users_year1": 500_000,
            "partner_spend_share": 0.30,
            "sku_share_in_check": 0.03,
            "sku_funding_rate": 0.05,
            "cac": 4000,
            "churn_rate": 0.025,
            "key_rate": 0.10,
            "inflation_rate": 0.04,
            # OPEX intentionally unchanged (жёстко)
        }
    }
}


def apply_scenario(base_params: dict, scenario_key: str) -> dict:
    p = copy.deepcopy(base_params)
    overrides = SCENARIOS.get(scenario_key, SCENARIOS["base"])["overrides"]
    for k, v in overrides.items():
        if k in p:
            p[k]["value"] = float(v)
    return p


# --- 5) UI ---
st.title("Финмодель Суперкарты")
st.caption("Scenario + Unit Economics + P&L + LTV(NPV).")

st.sidebar.header("📉 Сценарии")
scenario_key = st.sidebar.radio(
    "Выберите сценарий",
    options=list(SCENARIOS.keys()),
    format_func=lambda k: SCENARIOS[k]["label"],
    index=0
)

lock_scenario = st.sidebar.toggle("🔒 Зафиксировать параметры сценария (без ручных правок)", value=False)

params = apply_scenario(DEFAULT_PARAMS, scenario_key)

st.sidebar.header("⚙️ Параметры")
updated_params = {}

def ui_disabled() -> bool:
    return bool(lock_scenario)

for key, item in params.items():
    desc = item["desc"]
    default_val = float(item["value"])

    is_share = ("share" in key) or ("penetration" in key)
    is_rate = ("rate" in key)

    # Configure UI controls with sane bounds
    if key == "active_users_year1":
        val = st.sidebar.number_input(desc, min_value=0.0, value=default_val, step=10_000.0, disabled=ui_disabled())
    elif key in ("tpv_per_user", "avg_balance"):
        val = st.sidebar.number_input(desc, min_value=0.0, value=default_val, step=500.0, disabled=ui_disabled())
    elif key == "opex_year":
        val = st.sidebar.number_input(desc, min_value=0.0, value=default_val, step=50_000_000.0, disabled=ui_disabled())
    elif key == "cac":
        val = st.sidebar.number_input(desc, min_value=0.0, value=default_val, step=100.0, disabled=ui_disabled())
    elif key == "key_rate":
        val = st.sidebar.slider(f"🏦 {desc}", min_value=0.05, max_value=0.30, value=default_val, step=0.005, format="%.3f", disabled=ui_disabled())
    elif key == "inflation_rate":
        val = st.sidebar.slider(f"📈 {desc}", min_value=0.00, max_value=0.30, value=default_val, step=0.005, format="%.3f", disabled=ui_disabled())
    elif is_share:
        val = st.sidebar.slider(desc, min_value=0.0, max_value=1.0, value=min(max(default_val, 0.0), 1.0), step=0.01, format="%.2f", disabled=ui_disabled())
    elif is_rate:
        val = st.sidebar.number_input(desc, min_value=0.0, value=default_val, step=0.001, format="%.4f", disabled=ui_disabled())
    elif key == "churn_rate":
        val = st.sidebar.slider(desc, min_value=0.0, max_value=0.20, value=min(max(default_val, 0.0), 0.20), step=0.001, format="%.3f", disabled=ui_disabled())
    elif key == "tax_rate":
        val = st.sidebar.slider(desc, min_value=0.0, max_value=0.50, value=min(max(default_val, 0.0), 0.50), step=0.01, format="%.2f", disabled=ui_disabled())
    else:
        val = st.sidebar.number_input(desc, value=default_val, disabled=ui_disabled())

    updated_params[key] = {"value": float(val), "desc": desc}

# --- 6) CALC ---
model = FinancialModel(updated_params)
unit = model.calculate_unit_economics()
pl = model.calculate_pl_year(unit)

# --- 7) DASHBOARD ---
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric("EBITDA (Год 1)", f"{pl['ebitda'] / 1e9:,.2f} млрд ₽")

with k2:
    nii_year = unit["rev_float"] * updated_params["active_users_year1"]["value"] * 12.0
    nii_share = (unit["rev_float"] / unit["total_revenue"]) * 100.0 if unit["total_revenue"] > 0 else 0.0
    st.metric("NII (остатки)", f"{nii_year / 1e9:,.2f} млрд ₽", delta=f"{nii_share:.1f}% выручки")

with k3:
    st.metric("Net income (Год 1)", f"{pl['net_income'] / 1e9:,.2f} млрд ₽")

with k4:
    st.metric("LTV (NPV 3 года)", f"{pl['ltv_npv']:,.0f} ₽", delta=f"LTV/CAC = {pl['ltv_cac']:.2f}x")

with k5:
    pb = pl["payback_months"]
    pb_txt = "∞" if not np.isfinite(pb) else f"{pb:.1f} мес"
    st.metric("Payback (грубо)", pb_txt, delta="CAC / contrib.margin")

st.markdown("---")

c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("💰 Декомпозиция дохода (₽/клиент/мес)")
    df_unit = pd.DataFrame(
        [
            {"Item": "Interchange", "Value": unit["rev_interchange"]},
            {"Item": "CPA партнёра", "Value": unit["rev_partner_cpa"]},
            {"Item": "SKU funding", "Value": unit["rev_sku"]},
            {"Item": "Подписка", "Value": unit["rev_sub"]},
            {"Item": "NII (остатки)", "Value": unit["rev_float"]},
            {"Item": "Процессинг (кост)", "Value": -unit["cost_processing"]},
        ]
    )

    fig_bar = px.bar(df_unit, x="Item", y="Value", text_auto=".0f")
    fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="₽ на клиента")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Quick NII intuition box
    key_rate = updated_params["key_rate"]["value"]
    ftp_spread = updated_params["transfer_price_spread"]["value"]
    eff = max(0.0, key_rate - ftp_spread)
    st.info(
        f"**Интуиция по NII:** при остатке 10 000 ₽ и net-rate {(eff*100):.1f}% годовых → "
        f"≈ **{(10_000*eff/12):.0f} ₽/мес** маржи на клиента."
    )

with c2:
    st.subheader("📈 EBITDA vs Key Rate (sensitivity)")
    rates = np.linspace(0.05, 0.30, 26, dtype=np.float64)

    base_params_for_sens = copy.deepcopy(updated_params)
    ebitda_sensitivity = np.empty_like(rates)

    for i, r in enumerate(rates):
        temp_params = copy.deepcopy(base_params_for_sens)
        temp_params["key_rate"]["value"] = float(r)

        temp_model = FinancialModel(temp_params)
        temp_unit = temp_model.calculate_unit_economics()
        temp_pl = temp_model.calculate_pl_year(temp_unit)
        ebitda_sensitivity[i] = temp_pl["ebitda"]

    df_sens = pd.DataFrame({"Key Rate": rates, "EBITDA": ebitda_sensitivity})
    fig_sens = px.line(df_sens, x="Key Rate", y="EBITDA", markers=True)
    fig_sens.update_layout(xaxis_tickformat=".0%")
    st.plotly_chart(fig_sens, use_container_width=True)

st.markdown("---")

# Base vs Worst comparison (even when user is in one scenario)
st.subheader("🧯 Base vs Worst (сводка)")
base_params = apply_scenario(DEFAULT_PARAMS, "base")
worst_params = apply_scenario(DEFAULT_PARAMS, "worst")

base_model = FinancialModel(base_params)
worst_model = FinancialModel(worst_params)

base_unit = base_model.calculate_unit_economics()
worst_unit = worst_model.calculate_unit_economics()

base_pl = base_model.calculate_pl_year(base_unit)
worst_pl = worst_model.calculate_pl_year(worst_unit)

cmp = pd.DataFrame(
    [
        {"Metric": "MAU (Year 1)", "Base": base_params["active_users_year1"]["value"], "Worst": worst_params["active_users_year1"]["value"]},
        {"Metric": "Unit revenue (₽/мес)", "Base": base_unit["total_revenue"], "Worst": worst_unit["total_revenue"]},
        {"Metric": "Contrib.margin (₽/мес)", "Base": base_unit["contribution_margin"], "Worst": worst_unit["contribution_margin"]},
        {"Metric": "EBITDA (₽/год)", "Base": base_pl["ebitda"], "Worst": worst_pl["ebitda"]},
        {"Metric": "Net income (₽/год)", "Base": base_pl["net_income"], "Worst": worst_pl["net_income"]},
        {"Metric": "LTV NPV (₽)", "Base": base_pl["ltv_npv"], "Worst": worst_pl["ltv_npv"]},
        {"Metric": "LTV/CAC (x)", "Base": base_pl["ltv_cac"], "Worst": worst_pl["ltv_cac"]},
        {"Metric": "Payback (мес)", "Base": base_pl["payback_months"], "Worst": worst_pl["payback_months"]},
    ]
)

# Pretty formatting
cmp_display = cmp.copy()
for col in ("Base", "Worst"):
    cmp_display[col] = cmp_display.apply(
        lambda row: f"{row[col]:,.0f}" if row["Metric"] not in ("LTV/CAC (x)", "Payback (мес)") else f"{row[col]:.2f}",
        axis=1
    )
st.dataframe(cmp_display, use_container_width=True)

with st.expander("🔍 Полные данные (текущий сценарий)"):
    st.write("Unit economics:", unit)
    st.write("P&L:", pl)
