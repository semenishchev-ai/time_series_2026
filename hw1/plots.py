import matplotlib.pyplot as plt
import pandas as pd

from analysis import SALES


def plot_aggregations(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """Показывает исходные продажи на трёх календарных частотах."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for ax, table, title in zip(
        axes, (daily, weekly, monthly), ("По дням", "По неделям", "По месяцам")
    ):
        ax.plot(table.index, table[SALES], color="#176380", marker=".", lw=1)
        if "full_period" in table:
            edge = table.loc[~table["full_period"], SALES]
            ax.scatter(edge.index, edge, marker="x", s=45, color="#b0522a", label="Неполный период")
            ax.legend(loc="upper left")
        ax.set(title=title, ylabel="Продажи, шт.")
    axes[-1].set_xlabel("Дата периода")
    fig.suptitle("Зафиксированные продажи выбранного товара")


def plot_quality(daily: pd.DataFrame, spikes: pd.DataFrame) -> None:
    """Отмечает ненаблюдаемые даты и кандидатов в выбросы."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout="constrained")
    known = daily[SALES]
    unknown = known.index[known.isna()]
    axes[0].plot(known, color="#176380", lw=1, label="Исходные продажи")
    axes[0].scatter(
        unknown, [0.04] * len(unknown), transform=axes[0].get_xaxis_transform(),
        marker="|", s=85, color="#b0522a", label="Нет наблюдения",
    )
    axes[0].set(title="Дневные продажи и календарные разрывы", ylabel="Шт. за день")
    axes[0].legend(loc="upper left")
    axes[1].plot(spikes["value"], color="#176380", label="Продажи")
    axes[1].plot(spikes["median"], linestyle="--", color="#346650", label="Медиана соседей")
    flagged = spikes.loc[spikes["flag"], "value"]
    if not flagged.empty:
        axes[1].scatter(flagged.index, flagged, color="#b0522a", label="Кандидаты в выбросы")
    axes[1].set(title="Диагностика всплесков на полных неделях", ylabel="Шт. за неделю", xlabel="Дата")
    axes[1].legend(loc="upper left")


def plot_rolling(series: pd.Series, window: int = 8) -> None:
    """Отдельно показывает изменение уровня и стандартного отклонения."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, layout="constrained")
    axes[0].plot(series, color="#9fadbb", label="Недельные продажи")
    axes[0].plot(mean, color="#176380", label=f"Среднее за {window} недель")
    axes[0].legend(loc="upper left")
    axes[0].set(title="Локальный уровень продаж", ylabel="Шт. за неделю")
    axes[1].plot(std, color="#346650")
    axes[1].set(title=f"Стандартное отклонение за {window} недель", ylabel="Шт.", xlabel="Конец недели")


def plot_decomposition(components: pd.DataFrame) -> None:
    """Показывает входной ряд и три компоненты STL."""
    names = {
        "diagnostic": "Диагностический ряд",
        "trend": "Тренд",
        "seasonal": "Недельная компонента",
        "residual": "Остатки",
    }
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, layout="constrained")
    for ax, (column, name) in zip(axes, names.items()):
        ax.plot(components[column], color="#176380", linewidth=1)
        ax.set(title=name, ylabel="Шт.")
        if column in {"seasonal", "residual"}:
            ax.axhline(0, color="#949ea8", linewidth=0.8)
    axes[-1].set_xlabel("Дата")
    fig.suptitle("Устойчивая STL-декомпозиция с периодом 7 дней")


def plot_weekdays(profiles: pd.DataFrame, halves: pd.DataFrame) -> None:
    """Сравнивает уровни наблюдаемых продаж и устойчивость профиля по времени."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), layout="constrained")
    profiles[["mean", "median"]].rename(
        columns={"mean": "Среднее", "median": "Медиана"}
    ).plot.bar(ax=axes[0], color=["#176380", "#a5bfad"], rot=0)
    axes[0].set(title="Продажи по наблюдаемым дням", xlabel="День недели", ylabel="Шт. за день")
    halves.plot.bar(ax=axes[1], color=["#176380", "#b0522a"], rot=0)
    axes[1].axhline(0, color="#949ea8", linewidth=0.8)
    axes[1].set(
        title="Медиана с поправкой на тренд",
        xlabel="День недели",
        ylabel="Отклонение от тренда, шт.",
    )


def plot_difference(series: pd.Series, difference: pd.Series) -> None:
    """Сопоставляет недельные уровни и первые разности."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.8), sharex=True, layout="constrained")
    axes[0].plot(series, color="#176380")
    axes[0].set(title="Исходный недельный ряд", ylabel="Шт. за неделю")
    axes[1].plot(difference, color="#346650")
    axes[1].axhline(0, color="#949ea8", linewidth=0.8)
    axes[1].set(title="Первая разность", ylabel="Изменение, шт.", xlabel="Конец недели")
