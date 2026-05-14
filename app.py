from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Длительная прочность от номера зерна",
    layout="wide",
)

SIGMA_VALUES = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
GRAIN_VALUES = list(range(3, 11))
Y_MIN = 20
Y_MAX = 100


@dataclass
class FitResult:
    name: str
    r2: float
    predict: Callable[[np.ndarray], np.ndarray]
    equation: str


def calc_k1(grain: int) -> float:
    return 0.002 * grain**3 - 0.0237 * grain**2 + 0.1007 * grain + 0.3419


def calc_k2(k1: float) -> float:
    return 220.5 / math.exp(-k1 * 14.698)


def calc_p_from_sigma(sigma: float, k1: float, k2: float) -> float:
    base = k2 / 1.25
    ratio = sigma / base
    if ratio <= 0 or ratio >= 1:
        raise ValueError("Значение σ выходит за допустимый диапазон для расчета P.")
    return -math.log(ratio) / k1


def calc_p_from_user_input(temperature_c: float, tau_hours: float) -> float:
    temperature_k = temperature_c + 273.15
    if temperature_k <= 0:
        raise ValueError("Температура в Кельвинах должна быть положительной.")
    if tau_hours <= 0:
        raise ValueError("τост должна быть больше нуля.")
    return temperature_k * (math.log10(tau_hours) - 2 * math.log10(temperature_k) + 19.52) * 1e-3


def calc_sigma_from_p(p_value: float, k1: float, k2: float) -> float:
    return (k2 / 1.25) * math.exp(-k1 * p_value)


def fit_linear(x: np.ndarray, y: np.ndarray) -> FitResult:
    a, b = np.polyfit(x, y, 1)
    pred = lambda values: a * values + b
    return FitResult("Линейная", r2_score(y, pred(x)), pred, f"σ = {a:.4f}·P + {b:.4f}")


def fit_exponential(x: np.ndarray, y: np.ndarray) -> FitResult | None:
    if np.any(y <= 0):
        return None
    b, ln_a = np.polyfit(x, np.log(y), 1)
    a = float(np.exp(ln_a))
    pred = lambda values: a * np.exp(b * values)
    return FitResult("Экспоненциальная", r2_score(y, pred(x)), pred, f"σ = {a:.4f}·exp({b:.4f}·P)")


def fit_logarithmic(x: np.ndarray, y: np.ndarray) -> FitResult | None:
    if np.any(x <= 0):
        return None
    a, b = np.polyfit(np.log(x), y, 1)
    pred = lambda values: a * np.log(values) + b
    return FitResult("Логарифмическая", r2_score(y, pred(x)), pred, f"σ = {a:.4f}·ln(P) + {b:.4f}")


def fit_power(x: np.ndarray, y: np.ndarray) -> FitResult | None:
    if np.any(x <= 0) or np.any(y <= 0):
        return None
    b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
    a = float(np.exp(ln_a))
    pred = lambda values: a * np.power(values, b)
    return FitResult("Степенная", r2_score(y, pred(x)), pred, f"σ = {a:.4f}·P^{b:.4f}")


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> FitResult:
    coeffs = np.polyfit(x, y, degree)
    poly = np.poly1d(coeffs)
    pred = lambda values: poly(values)
    pieces = []
    power = degree
    for coef in coeffs:
        if power > 1:
            pieces.append(f"{coef:.4f}·P^{power}")
        elif power == 1:
            pieces.append(f"{coef:.4f}·P")
        else:
            pieces.append(f"{coef:.4f}")
        power -= 1
    equation = "σ = " + " + ".join(pieces).replace("+ -", "- ")
    return FitResult(f"Полиномиальная {degree} ст.", r2_score(y, pred(x)), pred, equation)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0:
        return 1.0
    return 1 - ss_res / ss_tot


def get_best_fit(x: np.ndarray, y: np.ndarray) -> tuple[FitResult, list[FitResult]]:
    candidates = [
        fit_linear(x, y),
        fit_exponential(x, y),
        fit_logarithmic(x, y),
        fit_power(x, y),
        fit_polynomial(x, y, 2),
        fit_polynomial(x, y, 3),
    ]
    valid = [item for item in candidates if item is not None]
    best = max(valid, key=lambda item: item.r2)
    return best, valid


def build_grain_dataframe(grain: int) -> tuple[pd.DataFrame, float, float, FitResult, list[FitResult]]:
    k1 = calc_k1(grain)
    k2 = calc_k2(k1)
    p_values = np.array([calc_p_from_sigma(sigma, k1, k2) for sigma in SIGMA_VALUES], dtype=float)
    df = pd.DataFrame(
        {
            "Номер зерна G": grain,
            "σ, МПа": SIGMA_VALUES,
            "P": np.round(p_values, 5),
            "k1": round(k1, 6),
            "k2": round(k2, 6),
        }
    )
    best_fit, all_fits = get_best_fit(p_values, SIGMA_VALUES)
    return df, k1, k2, best_fit, all_fits


def make_chart(
    grain: int,
    df: pd.DataFrame,
    best_fit: FitResult,
    user_point: tuple[float, float] | None = None,
) -> go.Figure:
    x = df["P"].to_numpy(dtype=float)
    y = df["σ, МПа"].to_numpy(dtype=float)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    y_line = best_fit.predict(x_line)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            name=f"Расчётные точки G={grain}",
            marker=dict(size=9),
            line=dict(width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"Тренд: {best_fit.name}",
            line=dict(width=3, dash="dash"),
        )
    )

    if user_point is not None:
        p_user, sigma_user = user_point
        fig.add_trace(
            go.Scatter(
                x=[p_user],
                y=[sigma_user],
                mode="markers",
                name="Точка пользователя",
                marker=dict(size=12, color="red", symbol="diamond"),
            )
        )

    fig.update_layout(
        title=f"Номер зерна G = {grain}",
        xaxis_title="Параметр P",
        yaxis_title="Напряжение σ, МПа",
        yaxis=dict(range=[Y_MIN, Y_MAX]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def main() -> None:
    st.title("Длительная прочность от номера зерна")
    st.write(
        "Программа строит 8 графиков зависимости напряжения σ от параметра P, "
        "подбирает лучшую линию тренда и показывает таблицу расчётных точек."
    )

    st.sidebar.header("Пользовательская точка")
    selected_grain = st.sidebar.selectbox(
        "Номер зерна G",
        options=GRAIN_VALUES,
        index=0,
    )
    temperature_c = st.sidebar.number_input(
        "Температура, °C",
        min_value=1.0,
        value=600.0,
        step=1.0,
    )
    tau_hours = st.sidebar.number_input(
        "τост, ч",
        min_value=1.0,
        value=100000.0,
        step=1000.0,
    )

    user_point = None
    user_point_message = None
    k1_selected = calc_k1(selected_grain)
    k2_selected = calc_k2(k1_selected)
    try:
        p_user = calc_p_from_user_input(temperature_c, tau_hours)
        sigma_user = calc_sigma_from_p(p_user, k1_selected, k2_selected)
        user_point = (p_user, sigma_user)
        user_point_message = {
            "G": selected_grain,
            "P": p_user,
            "σ": sigma_user,
            "k1": k1_selected,
            "k2": k2_selected,
        }
    except ValueError as error:
        st.sidebar.error(str(error))

    if user_point_message:
        st.sidebar.success("Пользовательская точка рассчитана")
        st.sidebar.write(f"P = {user_point_message['P']:.5f}")
        st.sidebar.write(f"σ = {user_point_message['σ']:.5f} МПа")
        if not (Y_MIN <= user_point_message["σ"] <= Y_MAX):
            st.sidebar.warning(
                "Полученная пользовательская точка по σ выходит за диапазон 20–100 МПа."
            )

    all_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    tabs = st.tabs([f"G = {grain}" for grain in GRAIN_VALUES])

    for tab, grain in zip(tabs, GRAIN_VALUES):
        df, k1, k2, best_fit, all_fits = build_grain_dataframe(grain)
        all_tables.append(df)
        summary_rows.append(
            {
                "Номер зерна G": grain,
                "k1": round(k1, 6),
                "k2": round(k2, 6),
                "Лучшая аппроксимация": best_fit.name,
                "R²": round(best_fit.r2, 6),
                "Уравнение": best_fit.equation,
            }
        )

        point_for_chart = user_point if grain == selected_grain else None

        with tab:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(
                    make_chart(grain, df, best_fit, point_for_chart),
                    use_container_width=True,
                )
            with col2:
                st.subheader("Лучшая аппроксимация")
                st.write(f"**Тип:** {best_fit.name}")
                st.write(f"**R²:** {best_fit.r2:.6f}")
                st.code(best_fit.equation)
                st.write("**Все проверенные варианты:**")
                fits_df = pd.DataFrame(
                    [
                        {
                            "Аппроксимация": item.name,
                            "R²": round(item.r2, 6),
                            "Уравнение": item.equation,
                        }
                        for item in sorted(all_fits, key=lambda fit: fit.r2, reverse=True)
                    ]
                )
                st.dataframe(fits_df, use_container_width=True, hide_index=True)

            st.subheader("Таблица расчётных значений")
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.header("Сводка по всем номерам зерна")
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.header("Общая таблица расчётных точек")
    combined_df = pd.concat(all_tables, ignore_index=True)
    st.dataframe(combined_df, use_container_width=True, hide_index=True)

    if user_point_message:
        st.header("Пользовательская точка")
        user_df = pd.DataFrame(
            [
                {
                    "Номер зерна G": user_point_message["G"],
                    "Температура, °C": temperature_c,
                    "Температура, K": round(temperature_c + 273.15, 2),
                    "τост, ч": tau_hours,
                    "k1": round(user_point_message["k1"], 6),
                    "k2": round(user_point_message["k2"], 6),
                    "P": round(user_point_message["P"], 5),
                    "σ, МПа": round(user_point_message["σ"], 5),
                }
            ]
        )
        st.dataframe(user_df, use_container_width=True, hide_index=True)

        st.caption(
            "Пользовательская точка считается по формуле P = T·(log(τост) − 2·log(T) + 19,52)·10⁻³, "
            "где T переведена из °C в K."
        )


if __name__ == "__main__":
    main()
