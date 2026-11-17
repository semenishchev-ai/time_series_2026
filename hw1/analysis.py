from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from warnings import catch_warnings, simplefilter
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss


SOURCE_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
SOURCE_NAME = "Online Retail.xlsx"
WEEKDAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
SALES = "sales"


def load_retail(data_dir: Path) -> pd.DataFrame:
    """Загружает исходный файл UCI в локальный кэш и читает таблицу."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / SOURCE_NAME
    if not path.exists():
        with NamedTemporaryFile(dir=data_dir) as download:
            with urlopen(SOURCE_URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    download.write(chunk)
            download.flush()
            with ZipFile(download.name) as archive:
                if SOURCE_NAME not in archive.namelist():
                    raise ValueError("В архиве UCI отсутствует таблица Online Retail.")
                archive.extract(SOURCE_NAME, data_dir)
    with path.open("rb") as source:
        return pd.read_excel(
            source,
            engine="openpyxl",
            dtype={"InvoiceNo": "string", "StockCode": "string"},
        )


def audit_fields(raw: pd.DataFrame) -> pd.DataFrame:
    """Показывает типы полей и фактическое число пустых ячеек."""
    meanings = [
        "Номер счёта",
        "Код позиции",
        "Наименование",
        "Количество единиц в строке",
        "Дата и время счёта",
        "Цена за единицу, фунты",
        "Код покупателя",
        "Страна покупателя",
    ]
    return pd.DataFrame(
        {
            "Содержание": dict(zip(raw.columns, meanings)),
            "Тип": raw.dtypes.astype(str),
            "Пропуски": raw.isna().sum(),
        }
    )


def prepare_retail(raw: pd.DataFrame) -> pd.DataFrame:
    """Приводит ключевые поля к нужным типам и классифицирует строки."""
    retail = raw.assign(
        InvoiceNo=raw["InvoiceNo"].str.strip().str.upper(),
        StockCode=raw["StockCode"].str.strip().str.upper(),
        InvoiceDate=pd.to_datetime(raw["InvoiceDate"], errors="coerce"),
        Quantity=pd.to_numeric(raw["Quantity"], errors="coerce"),
        UnitPrice=pd.to_numeric(raw["UnitPrice"], errors="coerce"),
    )
    retail["day"] = retail["InvoiceDate"].dt.normalize()
    retail["is_product"] = retail["StockCode"].str.fullmatch(r"\d{5}[A-Z]?", na=False)
    retail["is_cancelled"] = retail["InvoiceNo"].str.startswith("C", na=False)
    retail["is_sale"] = (
        retail["is_product"]
        & retail["InvoiceNo"].notna()
        & ~retail["is_cancelled"]
        & retail["Quantity"].gt(0)
        & retail["UnitPrice"].gt(0)
        & retail["day"].notna()
    )
    return retail


def rank_products(retail: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает частотность товаров по продажам, счетам и количеству единиц."""
    sales = retail.loc[retail["is_sale"]]
    ranked = sales.groupby("StockCode").agg(
        rows=("InvoiceNo", "size"),
        orders=("InvoiceNo", "nunique"),
        units=("Quantity", "sum"),
    )
    names = sales.groupby("StockCode")["Description"].agg(
        lambda values: values.value_counts().index[0] if values.notna().any() else "Без названия"
    )
    ranked["name"] = names
    ranked["raw_rows"] = retail.loc[retail["is_product"], "StockCode"].value_counts()
    for metric in ("rows", "orders", "units", "raw_rows"):
        ranked[f"rank_{metric}"] = ranked[metric].rank(ascending=False, method="min").astype(int)
    return ranked.sort_values(["rows", "orders"], ascending=False, kind="stable")


def choose_product(ranked: pd.DataFrame) -> str:
    """Берёт самый частый товар среди тех, кто не лидирует ни по одному счётчику."""
    places = ["rank_rows", "rank_orders", "rank_units", "rank_raw_rows"]
    alternatives = ranked.loc[ranked[places].gt(1).all(axis=1)]
    if alternatives.empty:
        raise ValueError("В рейтинге не найден товар, отличный от лидеров.")
    return str(alternatives.index[0])


def build_calendar(retail: pd.DataFrame, stock_code: str) -> pd.DataFrame:
    """Отделяет наблюдаемые нули от дней без подтверждённой активности магазина."""
    if not (retail["StockCode"].eq(stock_code) & retail["is_sale"]).any():
        raise ValueError(f"Для кода {stock_code} нет подтверждённых продаж.")
    dates = pd.date_range(retail["day"].min(), retail["day"].max(), freq="D", name="date")
    daily = pd.DataFrame(index=dates)
    daily["store_rows"] = retail.groupby("day").size().reindex(dates, fill_value=0)
    sales = retail.loc[retail["is_sale"]]
    daily["store_active"] = sales.groupby("day").size().reindex(dates, fill_value=0).gt(0)
    product = sales.loc[sales["StockCode"].eq(stock_code)]
    daily[SALES] = product.groupby("day")["Quantity"].sum().reindex(dates)
    zero = daily["store_active"] & daily[SALES].isna()
    daily.loc[zero, SALES] = 0
    daily["status"] = np.select(
        [daily[SALES].gt(0), zero, daily["store_rows"].gt(0)],
        ["Есть продажи", "Нет продаж товара", "Есть записи, нет продаж магазина"],
        default="Нет записей магазина",
    )
    return daily


def aggregate_calendar(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Суммирует зафиксированные продажи и показывает покрытие календаря."""
    groups = daily.resample(frequency)
    result = groups.agg(
        calendar_days=(SALES, "size"),
        observed_days=(SALES, "count"),
        sale_days=(SALES, lambda values: int(values.gt(0).sum())),
    )
    result.insert(0, SALES, groups[SALES].sum(min_count=1))
    result["unknown_days"] = result["calendar_days"] - result["observed_days"]
    expected = 7 if frequency == "W-SUN" else result.index.days_in_month
    result["full_period"] = result["calendar_days"].eq(expected)
    return result


def describe_series(series: pd.Series) -> pd.Series:
    """Считает типичный уровень, разброс и долю наблюдаемых нулей."""
    known = series.dropna()
    mean, std = known.mean(), known.std()
    return pd.Series(
        {
            "Периодов": len(series),
            "Без наблюдения": int(series.isna().sum()),
            "Доля нулей, %": 100 * known.eq(0).mean(),
            "Среднее": mean,
            "Медиана": known.median(),
            "Стандартное отклонение": std,
            "Коэффициент вариации": std / mean if mean > 0 else np.nan,
            "Минимум": known.min(),
            "Максимум": known.max(),
            "Асимметрия": known.skew(),
        }
    )


def detect_spikes(series: pd.Series, window: int = 13, cutoff: float = 3.5) -> pd.DataFrame:
    """Ищет необычные точки по медиане и MAD соседей, исключая саму точку."""
    result = pd.DataFrame({"value": series, "median": np.nan, "score": np.nan})
    radius = window // 2
    for index, value in enumerate(series):
        if pd.isna(value):
            continue
        neighbors = pd.concat(
            [series.iloc[max(0, index - radius):index], series.iloc[index + 1:index + radius + 1]]
        ).dropna()
        if len(neighbors) < 6:
            continue
        center = neighbors.median()
        mad = (neighbors - center).abs().median()
        result.iloc[index, 1] = center
        if mad > 0:
            result.iloc[index, 2] = (value - center) / (1.4826 * mad)
    result["flag"] = result["score"].abs().gt(cutoff)
    return result


def decompose_daily(daily: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Применяет устойчивую STL с недельным периодом к ряду для диагностики."""
    observed = daily[SALES]
    weekday = observed.index.dayofweek
    always_inactive = daily.groupby(weekday)["store_active"].sum().eq(0)
    structural = observed.isna() & weekday.isin(always_inactive[always_inactive].index)
    recorded = observed.mask(structural, 0)
    estimated = pd.Series(index=observed.index, dtype=float)
    for day in range(7):
        same_day = observed.loc[weekday == day]
        if same_day.notna().sum() == 0:
            continue
        typical = same_day.rolling(9, center=True, min_periods=2).median()
        estimated.loc[same_day.index] = typical.fillna(same_day.median())

    diagnostic = recorded.fillna(estimated)
    if diagnostic.isna().any():
        raise ValueError("Для STL не удалось восстановить часть диагностического ряда.")
    decomposition = STL(diagnostic, period=7, robust=True).fit()
    result = pd.DataFrame(
        {
            "diagnostic": diagnostic,
            "trend": decomposition.trend,
            "seasonal": decomposition.seasonal,
            "residual": decomposition.resid,
            "structural": structural,
            "estimated": recorded.isna(),
        }
    )
    total_variance = (result["seasonal"] + result["residual"]).var()
    strength = max(0.0, 1 - result["residual"].var() / total_variance) if total_variance else 0.0
    return result, strength


def weekday_diagnostics(
    daily: pd.DataFrame, components: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, float, float, int, float]:
    """Проверяет недельный ритм на наблюдаемых днях, сравнивая их внутри недель."""
    known = daily[SALES].dropna()
    typical_days = sorted(known.index.dayofweek.unique())
    profile = known.groupby(known.index.dayofweek).agg(["size", "mean", "median"])
    profile.index = pd.Index([WEEKDAYS[i] for i in profile.index], name="День недели")
    detrended = daily[SALES] - components["trend"]
    frame = pd.DataFrame(
        {
            "week": detrended.index.to_period("W-SUN"),
            "weekday": detrended.index.dayofweek,
            "value": detrended,
        }
    )
    matched = frame.pivot(index="week", columns="weekday", values="value")[typical_days].dropna()
    if len(typical_days) < 3 or len(matched) < 2:
        raise ValueError("Недостаточно полных недель для сравнения дней по Фридману.")
    statistic, pvalue = friedmanchisquare(*(matched[day] for day in typical_days))
    size = len(matched)
    half_profiles = pd.DataFrame(
        {
            "Первая половина": matched.iloc[: size // 2].median(),
            "Вторая половина": matched.iloc[size // 2 :].median(),
        }
    )
    half_profiles.index = pd.Index([WEEKDAYS[i] for i in half_profiles.index], name="День недели")
    paired = pd.concat([detrended, detrended.shift(7)], axis=1).dropna()
    lag_correlation = paired.iloc[:, 0].corr(paired.iloc[:, 1])
    kendall_w = statistic / (size * (len(typical_days) - 1))
    return profile, half_profiles, pvalue, kendall_w, size, lag_correlation


def longest_observed_run(series: pd.Series) -> pd.Series:
    """Выбирает самый длинный отрезок между пропусками, не склеивая даты."""
    observed = series.notna()
    if not observed.any():
        raise ValueError("В ряду нет наблюдений.")
    spans = (~observed).cumsum()
    longest = spans[observed].value_counts().idxmax()
    return series.loc[observed & spans.eq(longest)]


def stationarity_tests(series: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Сравнивает ADF и KPSS с константой, не смешивая их нулевые гипотезы."""
    if series.isna().any() or not series.index.to_series().diff().iloc[1:].eq(pd.Timedelta(days=7)).all():
        raise ValueError("Для тестов передайте регулярный ряд без внутренних пропусков.")
    adf_stat, adf_p, adf_lags, *_ = adfuller(
        series, regression="c", autolag="AIC", result_object=False,
    )
    with catch_warnings(record=True) as captured:
        simplefilter("always", InterpolationWarning)
        kpss_stat, kpss_p, kpss_lags, _ = kpss(series, regression="c", nlags="auto")
    bound = ""
    if any(issubclass(item.category, InterpolationWarning) for item in captured):
        bound = "≤" if kpss_p <= 0.01 else "≥"
    return pd.DataFrame(
        {
            "Нулевая гипотеза": ["Единичный корень", "Стационарность вокруг константы"],
            "Статистика": [adf_stat, kpss_stat],
            "p": [adf_p, kpss_p],
            "Граница p": ["", bound],
            "Лагов": [adf_lags, kpss_lags],
            "H₀ отклонена": [adf_p < alpha, kpss_p < alpha],
        },
        index=pd.Index(["ADF", "KPSS"], name="Тест"),
    )


def stationarity_status(tests: pd.DataFrame) -> str:
    """Различает согласованные и противоречивые результаты двух тестов."""
    adf, kpss_reject = tests["H₀ отклонена"]
    if adf and not kpss_reject:
        return "Тесты согласуются со стационарностью вокруг константы."
    if not adf and kpss_reject:
        return "Тесты согласуются с нестационарностью уровня."
    if adf and kpss_reject:
        return "Оба теста отклоняют H₀; простой вывод о стационарности невозможен."
    return "Оба теста не отклоняют H₀; диагностических данных недостаточно."


def quarter_profile(series: pd.Series) -> pd.DataFrame:
    """Показывает изменение среднего и разброса по календарным кварталам."""
    return series.groupby(series.index.to_period("Q")).agg(["size", "mean", "median", "std"]).rename(
        columns={"size": "Недель", "mean": "Среднее", "median": "Медиана", "std": "Ст. отклонение"}
    )


def export_series(daily: pd.DataFrame, weekly: pd.DataFrame, monthly: pd.DataFrame) -> None:
    """Сохраняет ряды с признаками покрытия и отдельными датами периодов."""
    directory = Path("results")
    directory.mkdir(exist_ok=True)
    for name, table in {"daily": daily, "weekly": weekly, "monthly": monthly}.items():
        table.to_csv(directory / f"{name}.csv", index_label="date", date_format="%Y-%m-%d")
