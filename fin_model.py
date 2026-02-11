import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import numpy_financial as npf  # Для профессионального NPV

# --- 1. CONFIG & SETUP ---
st.set_page_config(layout="wide", page_title="Финмодель 'Синий Ценник' PRO", page_icon="🏦")


# --- 2. LOGIC CLASS ---
class FinancialModel:
    def __init__(self, params):
        self.p = {k: v['value'] for k, v in params.items()}

    def calculate_unit_economics(self):
        """Расчет экономики на 1 клиента в месяц"""
        p = self.p

        # --- КОМИССИОННЫЕ ДОХОДЫ (F&C) ---
        rev_interchange = p['tpv_per_user'] * p['interchange_rate']

        partner_vol = p['tpv_per_user'] * p['partner_spend_share']
        rev_partner_cpa = partner_vol * p['partner_cpa_rate']

        sku_vol = partner_vol * p['sku_share_in_check']
        rev_sku = sku_vol * p['sku_funding_rate']

        rev_sub = p['sub_price'] * p['sub_penetration']

        # --- ПРОЦЕНТНЫЕ ДОХОДЫ (NII - Net Interest Income) ---
        # Банк размещает остатки клиентов под (Key Rate - Spread)
        # Spread - это внутренняя стоимость ликвидности/операционки
        net_interest_rate = max(0, p['key_rate'] - p['transfer_price_spread'])
        rev_float_annual = p['avg_balance'] * net_interest_rate
        rev_float_monthly = rev_float_annual / 12

        total_revenue = rev_interchange + rev_partner_cpa + rev_sku + rev_sub + rev_float_monthly

        # --- РАСХОДЫ ---
        cost_processing = p['tpv_per_user'] * p['processing_cost_rate']

        contribution_margin = total_revenue - cost_processing
        margin_percent = (contribution_margin / total_revenue) * 100 if total_revenue > 0 else 0

        return {
            "rev_interchange": rev_interchange,
            "rev_partner_cpa": rev_partner_cpa,
            "rev_sku": rev_sku,
            "rev_sub": rev_sub,
            "rev_float": rev_float_monthly,  # Доход от остатков
            "total_revenue": total_revenue,
            "cost_processing": cost_processing,
            "contribution_margin": contribution_margin,
            "margin_percent": margin_percent
        }

    def calculate_pl_year(self, unit_eco):
        """Расчет P&L и LTV с учетом дисконтирования И ИНФЛЯЦИИ"""
        p = self.p
        users = p['active_users_year1']

        # Операционные показатели (Run-rate year 1)
        gross_revenue = unit_eco['total_revenue'] * users * 12
        operating_contribution = unit_eco['contribution_margin'] * users * 12

        ebitda = operating_contribution - p['opex_year']

        # Амортизация
        cac_amortization = (users * p['cac']) / 3
        ebt = ebitda - cac_amortization

        tax = ebt * p['tax_rate'] if ebt > 0 else 0
        net_income = ebt - tax

        # --- LTV CALCULATOR (DCF Model with Growth) ---
        # Ставка дисконтирования (месячная)
        monthly_discount_rate = (1 + p['key_rate']) ** (1 / 12) - 1

        # Ставка роста чека из-за инфляции (месячная)
        monthly_growth_rate = (1 + p['inflation_rate']) ** (1 / 12) - 1

        # Базовая маржа
        base_margin = unit_eco['contribution_margin']

        # Генерируем потоки на 36 месяцев
        cash_flows = []
        for i in range(36):
            # Маржа растет вместе с инфляцией (номинально)
            inflated_margin = base_margin * ((1 + monthly_growth_rate) ** i)
            # Учитываем вероятность, что клиент останется (Survival Rate)
            survival_rate = (1 - p['churn_rate']) ** i

            cash_flows.append(inflated_margin * survival_rate)

        # Считаем NPV от растущего потока
        ltv_npv = npf.npv(monthly_discount_rate, [0] + cash_flows)

        ltv_cac = ltv_npv / p['cac'] if p['cac'] > 0 else 0

        return {
            "gross_revenue": gross_revenue,
            "operating_contribution": operating_contribution,
            "opex": p['opex_year'],
            "ebitda": ebitda,
            "net_income": net_income,
            "ltv_npv": ltv_npv,
            "ltv_cac": ltv_cac,
            "ebt": ebt,
            "tax": tax
        }


# --- 3. UI LAYOUT ---

st.title("Финмодель Суперкарты")
st.markdown(f"### Key Rate Impact Analysis")

# Defaults
default_params = {
    "active_users_year1": {"desc": "Активная база (MAU)", "value": 1000000},
    "tpv_per_user": {"desc": "Оборот (TPV) руб/мес", "value": 35000},
    "avg_balance": {"desc": "Средний остаток на карте (руб)", "value": 15000},
    "partner_spend_share": {"desc": "Доля партнерских трат", "value": 0.45},
    "sub_penetration": {"desc": "Проникновение подписки", "value": 0.15},
    "sub_price": {"desc": "Цена подписки руб/мес", "value": 299},
    "interchange_rate": {"desc": "Interchange Rate", "value": 0.011},
    "partner_cpa_rate": {"desc": "Комиссия Партнера (CPA)", "value": 0.012},
    "sku_funding_rate": {"desc": "Комиссия Брендов (SKU)", "value": 0.10},
    "sku_share_in_check": {"desc": "Доля SKU-промо в чеке", "value": 0.05},
    "processing_cost_rate": {"desc": "Кост процессинга", "value": 0.003},
    "key_rate": {"desc": "Ключевая ставка ЦБ", "value": 0.165},
    "transfer_price_spread": {"desc": "Маржа трансфертная (расход)", "value": 0.02},
    "cac": {"desc": "CAC (Привлечение)", "value": 2500},
    "churn_rate": {"desc": "Отток (Churn)", "value": 0.01},
    "tax_rate": {"desc": "Налог на прибыль", "value": 0.25},
    "opex_year": {"desc": "Годовой OPEX", "value": 2800000000},
    "inflation_rate": {"desc": "Инфляция (рост чека)", "value": 0.08},
}

# Sidebar
st.sidebar.header("⚙️ Управление моделью")
updated_params = {}
for key, item in default_params.items():
    if key == 'key_rate':
        val = st.sidebar.slider(f"🏦 {item['desc']}", 0.05, 0.30, float(item['value']), 0.005, format="%.3f")
    elif 'rate' in key or 'share' in key or 'penetration' in key:
        val = st.sidebar.number_input(f"{item['desc']}", value=float(item['value']), format="%.4f")
    else:
        val = st.sidebar.number_input(f"{item['desc']}", value=float(item['value']))
    updated_params[key] = {"value": val, "desc": item['desc']}

# Calculation
model = FinancialModel(updated_params)
unit = model.calculate_unit_economics()
pl = model.calculate_pl_year(unit)

# --- DASHBOARD ---

# Top Metrics
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("EBITDA (Год 1)", f"{pl['ebitda'] / 1e9:,.2f} млрд ₽", delta="Операционная прибыль")
with k2:
    nii_share = (unit['rev_float'] / unit['total_revenue']) * 100
    st.metric("NII (Доход от остатков)",
              f"{(unit['rev_float'] * updated_params['active_users_year1']['value'] * 12) / 1e9:,.2f} млрд ₽",
              delta=f"{nii_share:.1f}% от выручки")
with k3:
    st.metric("Чистая прибыль (Net)", f"{pl['net_income'] / 1e9:,.2f} млрд ₽", delta=f"Налог 25% вычтен")
with k4:
    # LTV Logic with Discounting
    st.metric("LTV (NPV 3 года)", f"{pl['ltv_npv']:,.0f} ₽", delta="Дисконтировано по ставке ЦБ")

st.markdown("---")

c1, c2 = st.columns([1, 2])

with c1:
    st.subheader("💰 Декомпозиция дохода (Руб/мес)")
    df_unit = pd.DataFrame([
        {"Item": "Interchange", "Value": unit['rev_interchange']},
        {"Item": "Комиссия Партнера", "Value": unit['rev_partner_cpa']},
        {"Item": "SKU Бренды", "Value": unit['rev_sku']},
        {"Item": "Подписка", "Value": unit['rev_sub']},
        {"Item": "Процентный доход (NII)", "Value": unit['rev_float']},
        {"Item": "Процессинг (Кост)", "Value": -unit['cost_processing']},
    ])

    fig_bar = px.bar(df_unit, x="Item", y="Value", color="Value",
                     color_continuous_scale=["red", "green"], text_auto='.0f')
    fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Рублей на клиента")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.info(
        f"**Влияние Ставки ЦБ:** Каждые 10к остатков на счетах приносят банку **+{10000 * (updated_params['key_rate']['value'] - 0.02) / 12:.0f} руб/мес** маржи.")

with c2:
    st.subheader("📈 Зависимость EBITDA от Ставки ЦБ")

    rates = np.linspace(0.05, 0.30, 20)
    ebitda_sensitivity = []

    current_key_rate = updated_params['key_rate']['value']

    for r in rates:
        # Clone params
        temp_params = updated_params.copy()
        temp_params['key_rate'] = {'value': r}
        temp_model = FinancialModel(temp_params)
        temp_unit = temp_model.calculate_unit_economics()
        temp_pl = temp_model.calculate_pl_year(temp_unit)
        ebitda_sensitivity.append(temp_pl['ebitda'])

    df_sens = pd.DataFrame({"Key Rate": rates, "EBITDA": ebitda_sensitivity})

    fig_sens = px.line(df_sens, x="Key Rate", y="EBITDA", markers=True)
    fig_sens.add_vline(x=current_key_rate, line_dash="dash", line_color="red", annotation_text="Текущая ставка")

    # Format axis as percentage
    fig_sens.layout.xaxis.tickformat = '.0%'

    st.plotly_chart(fig_sens, use_container_width=True)
    st.caption(
        "График показывает, что наша модель зарабатывает БОЛЬШЕ при высокой ставке (за счет доходов от остатков), в отличие от кредитных продуктов.")

st.markdown("---")
with st.expander("🔍 Полные данные"):
    st.write(pl)