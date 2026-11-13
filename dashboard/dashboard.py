from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import streamlit as st
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


st.set_page_config(
    page_title="Dashboard E-Commerce Olist",
    page_icon="🛒",
    layout="wide",
)

sn.set_theme(style="white")

ACCENT = "#2563EB"
GRAY = "#B8BFCC"
TEXT = "#374151"
MUTED = "#6B7280"
SEGMENT_COLORS = {
    "Bintang (volume & harga tinggi)": "#2563EB",
    "Volume tinggi, harga rendah": "#0D9488",
    "Premium, volume rendah": "#D97706",
    "Potensi rendah": "#94A3B8",
}
PAYMENT_COLORS = {
    "credit_card": ACCENT,
    "boleto": "#D97706",
    "voucher": "#0D9488",
    "debit_card": "#7C3AED",
}
INSTALLMENT_LABELS = [
    "Lunas (1x)",
    "Cicilan pendek (2-6x)",
    "Cicilan menengah (7-12x)",
    "Cicilan panjang (>12x)",
]
SEGMENT_ORDER = list(SEGMENT_COLORS)
METHOD_LABELS = {
    "credit_card": "Credit Card",
    "boleto": "Boleto",
    "voucher": "Voucher",
    "debit_card": "Debit Card",
    "not_defined": "Not Defined",
}
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    .block-container { max-width: 1280px; padding-top: 2rem; }
    h1, h2, h3, h4, p, li, label,
    div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    h1, h2, h3, h4, p, li, label { color: var(--text-color, inherit); }
    [data-testid="stMetric"] {
        border-top: 3px solid #2563EB;
        padding-top: 0.7rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: var(--text-color, inherit); opacity: 0.78; }
    blockquote {
        border-left: 3px solid #2563EB !important;
        color: var(--text-color, inherit);
        margin: 0.5rem 0 1rem;
        padding: 0.25rem 0 0.25rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_num(value, decimals=0):
    """Format angka dengan pemisah ribuan titik dan desimal koma."""
    if pd.isna(value):
        return "—"
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "\0").replace(".", ",").replace("\0", ".")


def fmt_rs(value, decimals=2):
    return f"R$ {fmt_num(value, decimals)}"


def method_name(value):
    return METHOD_LABELS.get(value, str(value).replace("_", " ").title())


def category_name(value):
    return str(value).replace("_", " ").title()


def show_dataframe(frame, column_config=None, height=None):
    """Render tabel selebar kontainer, dengan fallback untuk Streamlit lama."""
    options = {
        "column_config": column_config or {},
        "hide_index": True,
    }
    if height is not None:
        options["height"] = height
    try:
        st.dataframe(frame, width="stretch", **options)
    except TypeError:
        st.dataframe(frame, use_container_width=True, **options)


def number_columns(frame, formats):
    """Buat konfigurasi kolom angka dengan format lokal browser."""
    configs = {}
    for column, label in formats.items():
        configs[column] = st.column_config.NumberColumn(
            label=label,
            format="localized",
        )
    return configs


def insight(text):
    st.markdown(f"> {text}")


def set_figure_style(fig, ax):
    fig.set_facecolor("white")
    ax.set_facecolor("white")
    ax.tick_params(colors=TEXT, labelcolor=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    sn.despine(ax=ax, top=True, right=True)


def render_figure(fig):
    st.pyplot(fig)
    plt.close(fig)


def format_axis_number(decimals=0, currency=False):
    def formatter(value, _position):
        if currency:
            return fmt_rs(value, decimals)
        return fmt_num(value, decimals)

    return FuncFormatter(formatter)


def barh_figure(series, title, xlabel, value_formatter, highlight_label):
    ordered = series.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.2, max(4.2, 0.48 * len(ordered) + 1.2)))
    colors = [ACCENT if label == highlight_label else GRAY for label in ordered.index]
    bars = ax.barh(ordered.index, ordered.values, color=colors, height=0.68)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_xlim(left=0)
    ax.xaxis.set_major_formatter(format_axis_number(0))
    maximum = float(ordered.max()) if len(ordered) else 0.0
    ax.set_xlim(0, max(maximum * 1.23, 1.0))
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, ordered.values):
        ax.text(
            bar.get_width() + maximum * 0.018,
            bar.get_y() + bar.get_height() / 2,
            value_formatter(value),
            va="center",
            ha="left",
            fontsize=8.5,
            color=TEXT,
        )
    set_figure_style(fig, ax)
    fig.tight_layout()
    return fig


def clip_date(value, minimum, maximum):
    return min(max(value, minimum), maximum)


@st.cache_data
def load_data():
    data_dir = Path(__file__).resolve().parent
    payment_path = data_dir / "metode_pembayaran_main.csv"
    category_path = data_dir / "kategori_produk_main.csv"

    for path in (payment_path, category_path):
        if not path.is_file():
            raise FileNotFoundError(path.name)

    payment = pd.read_csv(
        payment_path,
        usecols=[
            "order_id",
            "order_purchase_timestamp",
            "payment_type",
            "payment_installments",
            "payment_value",
        ],
        parse_dates=["order_purchase_timestamp"],
    )
    categories = pd.read_csv(
        category_path,
        usecols=[
            "order_id",
            "order_purchase_timestamp",
            "product_category_name_english",
            "price",
        ],
        parse_dates=["order_purchase_timestamp"],
    )
    payment = payment.loc[payment["payment_type"].ne("not_defined")].copy()
    return payment, categories


try:
    payment_data, category_data = load_data()
except FileNotFoundError as error:
    st.error(
        f"Berkas `{error.args[0]}` tidak ditemukan di folder `dashboard/`. "
        "Pastikan kedua CSV tersedia sejajar dengan `dashboard.py`."
    )
    st.stop()

payment_min = payment_data["order_purchase_timestamp"].min().date()
payment_max = payment_data["order_purchase_timestamp"].max().date()
category_min = category_data["order_purchase_timestamp"].min().date()
category_max = category_data["order_purchase_timestamp"].max().date()


st.title("Dashboard E-Commerce Olist")
st.write(
    "Ringkasan metode pembayaran dan performa kategori produk untuk membantu "
    "menentukan prioritas layanan, stok, dan promosi."
)
st.caption(
    "Data mencakup 4 Sep 2016–17 Okt 2018 untuk pembayaran dan "
    "4 Sep 2016–3 Sep 2018 untuk kategori produk. Seluruh nilai uang dalam Real Brasil (R$)."
)

with st.sidebar:
    st.header("Filter analisis")
    st.subheader("Metode pembayaran")
    payment_default = (
        clip_date(date(2017, 1, 1), payment_min, payment_max),
        clip_date(date(2018, 12, 31), payment_min, payment_max),
    )
    payment_range = st.date_input(
        "Rentang tanggal pembayaran",
        value=payment_default,
        min_value=payment_min,
        max_value=payment_max,
        key="rentang_tanggal_pembayaran",
    )
    all_methods = sorted(payment_data["payment_type"].dropna().unique().tolist())
    selected_methods = st.multiselect(
        "Metode pembayaran",
        options=all_methods,
        default=all_methods,
        format_func=method_name,
        key="metode_pembayaran",
    )

    st.divider()
    st.subheader("Kategori produk")
    category_default = (
        clip_date(date(2018, 1, 1), category_min, category_max),
        clip_date(date(2018, 12, 31), category_min, category_max),
    )
    category_range = st.date_input(
        "Rentang tanggal kategori produk",
        value=category_default,
        min_value=category_min,
        max_value=category_max,
        key="rentang_tanggal_kategori",
    )
    top_n = st.slider(
        "Jumlah kategori teratas",
        min_value=5,
        max_value=15,
        value=10,
        step=1,
        key="jumlah_kategori_teratas",
    )
    st.divider()
    st.caption("Sumber: dataset E-Commerce Public (Olist). Nilai uang dalam R$.")


if not isinstance(payment_range, (tuple, list)) or len(payment_range) != 2:
    st.info("Pilih tanggal akhir untuk melengkapi rentang tanggal pembayaran.")
    st.stop()
if not isinstance(category_range, (tuple, list)) or len(category_range) != 2:
    st.info("Pilih tanggal akhir untuk melengkapi rentang tanggal kategori produk.")
    st.stop()
if not selected_methods:
    st.warning("Pilih minimal satu metode pembayaran di sidebar untuk menampilkan analisis.")
    st.stop()

payment_start, payment_end = payment_range
category_start, category_end = category_range
payment_timestamps = payment_data["order_purchase_timestamp"]
category_timestamps = category_data["order_purchase_timestamp"]
payment_filter = (payment_timestamps >= pd.Timestamp(payment_start)) & (
    payment_timestamps < pd.Timestamp(payment_end) + pd.Timedelta(days=1)
)
category_filter = (category_timestamps >= pd.Timestamp(category_start)) & (
    category_timestamps < pd.Timestamp(category_end) + pd.Timedelta(days=1)
)
payment_filtered = payment_data.loc[payment_filter & payment_data["payment_type"].isin(selected_methods)].copy()
category_filtered = category_data.loc[category_filter].copy()

if payment_filtered.empty:
    st.warning(
        "Tidak ada pembayaran pada rentang dan metode yang dipilih. Perluas rentang tanggal "
        "atau pilih metode lain."
    )
    st.stop()
if category_filtered.empty:
    st.warning(
        "Tidak ada item kategori pada rentang tanggal yang dipilih. Perluas rentang tanggal kategori."
    )
    st.stop()

payment_counts = payment_filtered["payment_type"].value_counts()
payment_total = len(payment_filtered)
payment_value_total = payment_filtered["payment_value"].sum()
payment_share = payment_counts / payment_total * 100
top_method = payment_counts.index[0]

category_summary = (
    category_filtered.groupby("product_category_name_english", as_index=True)
    .agg(
        jumlah=("price", "size"),
        revenue=("price", "sum"),
        rata_harga=("price", "mean"),
    )
    .sort_values("revenue", ascending=False)
)
category_summary["pangsa_revenue"] = (
    category_summary["revenue"] / category_summary["revenue"].sum() * 100
)
top_volume_category = category_summary["jumlah"].idxmax()
top_revenue_category = category_summary["revenue"].idxmax()

metric_columns = st.columns(4)
with metric_columns[0]:
    st.metric("Total transaksi pembayaran", fmt_num(payment_total))
    st.caption(f"Total nilai: {fmt_rs(payment_value_total, 0)}")
with metric_columns[1]:
    st.metric("Metode paling banyak dipakai", method_name(top_method))
    st.caption(f"{fmt_num(payment_share.loc[top_method], 1)}% dari transaksi terfilter")
with metric_columns[2]:
    st.metric("Kategori terlaris (volume)", category_name(top_volume_category))
    st.caption(f"{fmt_num(category_summary.loc[top_volume_category, 'jumlah'])} item")
with metric_columns[3]:
    st.metric("Kategori revenue tertinggi", category_name(top_revenue_category))
    st.caption(f"Revenue: {fmt_rs(category_summary.loc[top_revenue_category, 'revenue'], 0)}")

st.caption(
    "Tiga baris `not_defined` dengan nilai nol dikeluarkan dari analisis pembayaran. "
    "Rentang tanggal tiap bagian mengikuti filter masing-masing di sidebar."
)

tab_payment, tab_category, tab_advanced, tab_conclusion = st.tabs(
    ["Metode pembayaran", "Kategori produk", "Analisis lanjutan", "Kesimpulan dan rekomendasi"]
)


with tab_payment:
    st.subheader("Preferensi metode pembayaran")
    payment_share_display = pd.Series(
        {method_name(method): share for method, share in payment_share.items()}
    ).sort_values(ascending=False)
    top_method_label = method_name(top_method)
    left, right = st.columns(2, gap="large")
    with left:
        fig = barh_figure(
            payment_share_display,
            "Proporsi transaksi menurut metode pembayaran",
            "Proporsi transaksi (%)",
            lambda value: f"{fmt_num(value, 1)}%",
            top_method_label,
        )
        render_figure(fig)
        second_method = payment_counts.index[1] if len(payment_counts) > 1 else top_method
        second_share = payment_share.get(second_method, 0.0)
        insight(
            f"{method_name(top_method)} mencakup {fmt_num(payment_share.loc[top_method], 1)}% "
            f"dari {fmt_num(payment_total)} baris pembayaran; metode berikutnya, "
            f"{method_name(second_method)}, berada di {fmt_num(second_share, 1)}%. "
            "Jaga metode utama tetap mudah ditemukan di tahap pembayaran dan pertahankan pilihan alternatif."
        )

    payment_years = sorted(payment_filtered["order_purchase_timestamp"].dt.year.unique().tolist())
    with right:
        if len(payment_years) >= 2:
            annual = pd.crosstab(
                payment_filtered["order_purchase_timestamp"].dt.year,
                payment_filtered["payment_type"],
            ).reindex(index=payment_years, columns=payment_counts.index, fill_value=0)
            annual_pct = annual.div(annual.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
            methods = payment_counts.index.tolist()
            positions = np.arange(len(methods))
            width = 0.78 / len(payment_years)
            fig, ax = plt.subplots(figsize=(8.2, 5.1))
            gradient = [
                tuple(
                    np.array(plt.matplotlib.colors.to_rgb(GRAY)) * (1 - ratio)
                    + np.array(plt.matplotlib.colors.to_rgb(ACCENT)) * ratio
                )
                for ratio in np.linspace(0, 1, len(payment_years))
            ]
            for index, year in enumerate(payment_years):
                offset = (index - (len(payment_years) - 1) / 2) * width
                bars = ax.bar(
                    positions + offset,
                    annual_pct.loc[year, methods].values,
                    width=width,
                    color=gradient[index],
                    label=str(year),
                )
                for bar in bars:
                    if bar.get_height() >= 1:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8,
                            f"{fmt_num(bar.get_height(), 1)}%",
                            ha="center",
                            va="bottom",
                            fontsize=7.5,
                            color=TEXT,
                        )
            ax.set_xticks(positions, [method_name(method) for method in methods])
            ax.set_ylim(0, max(100, float(annual_pct.max().max()) * 1.18))
            ax.set_ylabel("Proporsi transaksi (%)")
            ax.set_xlabel("Metode pembayaran")
            ax.set_title("Porsi per metode dan tahun dalam rentang terpilih", loc="left", fontsize=12, fontweight="bold", pad=14)
            ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.legend(title="Tahun", frameon=False, ncol=min(len(payment_years), 3))
            set_figure_style(fig, ax)
            fig.tight_layout()
            render_figure(fig)

            first_year, last_year = payment_years[0], payment_years[-1]
            changes = annual_pct.loc[last_year] - annual_pct.loc[first_year]
            largest_method = changes.abs().idxmax()
            largest_change = float(changes.loc[largest_method])
            shift_summary = (
                "pergeseran preferensi masih kecil"
                if abs(largest_change) < 3
                else "pergeseran preferensi mulai terlihat"
            )
            trend_text = (
                f"Dari {first_year} ke {last_year}, perubahan terbesar terjadi pada "
                f"{method_name(largest_method)} sebesar {fmt_num(largest_change, 1)} poin persentase; "
                f"{shift_summary} karena batasnya 3 poin persentase. Perbandingan memakai proporsi "
                "agar adil meskipun jumlah transaksi per tahun berbeda."
            )
            relative_growth = []
            for method in annual_pct.columns:
                earlier = float(annual_pct.loc[first_year, method])
                later = float(annual_pct.loc[last_year, method])
                if 0 < earlier < 5 and later / earlier >= 1.5:
                    relative_growth.append((method, later / earlier, later))
            if relative_growth:
                method, ratio, latest_share = max(relative_growth, key=lambda item: item[1])
                trend_text += (
                    f" {method_name(method)} tumbuh {fmt_num(ratio, 1)} kali secara relatif, "
                    f"meski porsinya masih kecil ({fmt_num(latest_share, 1)}% pada {last_year})."
                )
            insight(trend_text)
        else:
            method_counts = payment_filtered.copy()
            method_counts["bulan"] = method_counts["order_purchase_timestamp"].dt.to_period("M")
            monthly_counts = method_counts.groupby(["bulan", "payment_type"]).size().unstack(fill_value=0)
            monthly_index = pd.period_range(
                pd.Timestamp(payment_start).to_period("M"),
                pd.Timestamp(payment_end).to_period("M"),
                freq="M",
            )
            monthly_counts = monthly_counts.reindex(
                index=monthly_index, columns=payment_counts.index, fill_value=0
            )
            monthly_pct = monthly_counts.div(monthly_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
            fig, ax = plt.subplots(figsize=(8.2, 5.1))
            for method in payment_counts.index:
                ax.plot(
                    range(len(monthly_index)),
                    monthly_pct[method].values,
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label=method_name(method),
                    color=PAYMENT_COLORS.get(method, GRAY),
                )
            ax.set_xticks(range(len(monthly_index)), [MONTH_LABELS[month.month - 1] for month in monthly_index])
            ax.set_ylim(bottom=0)
            ax.set_ylabel("Proporsi transaksi (%)")
            ax.set_xlabel("Bulan")
            ax.set_title("Perubahan porsi metode pembayaran per bulan", loc="left", fontsize=12, fontweight="bold", pad=14)
            ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, ncol=2)
            set_figure_style(fig, ax)
            fig.tight_layout()
            render_figure(fig)
            nonempty_months = monthly_pct.loc[monthly_counts.sum(axis=1).gt(0)]
            if len(nonempty_months) >= 2:
                first_month, last_month = nonempty_months.index[0], nonempty_months.index[-1]
                month_changes = nonempty_months.iloc[-1] - nonempty_months.iloc[0]
                largest_method = month_changes.abs().idxmax()
                trend_text = (
                    f"Antara {MONTH_LABELS[first_month.month - 1]} dan {MONTH_LABELS[last_month.month - 1]}, "
                    f"perubahan terbesar terjadi pada {method_name(largest_method)} sebesar "
                    f"{fmt_num(month_changes.loc[largest_method], 1)} poin persentase. "
                    "Grafik memakai proporsi agar perubahan komposisi dapat dibandingkan meskipun "
                    "jumlah transaksi per bulan berbeda."
                )
            else:
                trend_text = (
                    "Rentang ini hanya memiliki satu bulan dengan transaksi. Grafik memakai proporsi "
                    "agar komposisi metode tetap dapat dibaca tanpa membandingkan jumlah mentah."
                )
            insight(trend_text)

    with st.expander("Lihat ringkasan transaksi per metode"):
        payment_summary = (
            payment_filtered.groupby("payment_type")
            .agg(
                jumlah_transaksi=("payment_value", "size"),
                total_nilai=("payment_value", "sum"),
                rata_rata_nilai=("payment_value", "mean"),
            )
        )
        payment_summary["persentase_transaksi"] = payment_summary["jumlah_transaksi"] / payment_total * 100
        payment_summary = payment_summary.sort_values("jumlah_transaksi", ascending=False).reset_index()
        payment_summary["payment_type"] = payment_summary["payment_type"].map(method_name)
        payment_summary = payment_summary.rename(
            columns={
                "payment_type": "Metode pembayaran",
                "jumlah_transaksi": "Jumlah transaksi",
                "persentase_transaksi": "Transaksi (%)",
                "total_nilai": "Total nilai (R$)",
                "rata_rata_nilai": "Rata-rata nilai (R$)",
            }
        )
        show_dataframe(
            payment_summary,
            number_columns(
                payment_summary,
                {
                    "Jumlah transaksi": "Jumlah transaksi",
                    "Transaksi (%)": "Transaksi (%)",
                    "Total nilai (R$)": "Total nilai (R$)",
                    "Rata-rata nilai (R$)": "Rata-rata nilai (R$)",
                },
            ),
        )
        highest_average = payment_summary.iloc[payment_summary["Rata-rata nilai (R$)"].argmax()]
        insight(
            f"{highest_average['Metode pembayaran']} memiliki rata-rata nilai per baris tertinggi, "
            f"{fmt_rs(highest_average['Rata-rata nilai (R$)'])}; bandingkan nilai ini dengan porsinya "
            "sebelum menentukan insentif khusus per metode."
        )


with tab_category:
    st.subheader("Kategori produk pada rentang terpilih")
    top_by_volume = category_summary.sort_values("jumlah", ascending=False).head(top_n).copy()
    top_by_revenue = category_summary.sort_values("revenue", ascending=False).head(top_n).copy()
    left, right = st.columns(2, gap="large")
    with left:
        volume_series = pd.Series(
            top_by_volume["jumlah"].values,
            index=[category_name(name) for name in top_by_volume.index],
        )
        volume_highlight = category_name(top_by_volume["jumlah"].idxmax())
        render_figure(
            barh_figure(
                volume_series,
                f"{top_n} kategori dengan item terjual terbanyak",
                "Jumlah item",
                lambda value: fmt_num(value),
                volume_highlight,
            )
        )
        volume_first = top_by_volume.iloc[0]
        volume_second = top_by_volume.iloc[1] if len(top_by_volume) > 1 else volume_first
        insight(
            f"{category_name(top_by_volume.index[0])} memimpin volume dengan "
            f"{fmt_num(volume_first['jumlah'])} item, diikuti {category_name(top_by_volume.index[1])} "
            f"({fmt_num(volume_second['jumlah'])}). Kategori teratas layak diprioritaskan untuk "
            "menjaga ketersediaan stok."
        )
    with right:
        revenue_series = pd.Series(
            top_by_revenue["revenue"].values,
            index=[category_name(name) for name in top_by_revenue.index],
        )
        revenue_highlight = category_name(top_by_revenue["revenue"].idxmax())
        render_figure(
            barh_figure(
                revenue_series,
                f"{top_n} kategori dengan total revenue tertinggi",
                "Total revenue (R$)",
                lambda value: fmt_rs(value, 0),
                revenue_highlight,
            )
        )
        revenue_first = top_by_revenue.iloc[0]
        insight(
            f"{category_name(top_by_revenue.index[0])} menghasilkan revenue tertinggi, "
            f"{fmt_rs(revenue_first['revenue'], 0)} atau "
            f"{fmt_num(revenue_first['pangsa_revenue'], 1)}% dari revenue kategori pada rentang ini. "
            "Peringkat revenue membantu membedakan kategori beromzet besar dari kategori beritem banyak."
        )

    avg_price_top_revenue = top_by_revenue.sort_values("rata_harga", ascending=False)
    avg_price_series = pd.Series(
        avg_price_top_revenue["rata_harga"].values,
        index=[category_name(name) for name in avg_price_top_revenue.index],
    )
    average_price_leader = avg_price_top_revenue["rata_harga"].idxmax()
    price_left, price_right = st.columns(2, gap="large")
    with price_left:
        render_figure(
            barh_figure(
                avg_price_series,
                "Rata-rata harga pada kategori dengan revenue tertinggi",
                "Rata-rata harga per item (R$)",
                lambda value: fmt_rs(value, 0),
                category_name(average_price_leader),
            )
        )
        volume_leader_price = category_summary.loc[top_volume_category, "rata_harga"]
        insight(
            f"{category_name(average_price_leader)} memiliki rata-rata harga tertinggi di antara "
            f"{top_n} kategori revenue teratas ({fmt_rs(category_summary.loc[average_price_leader, 'rata_harga'], 0)}), "
            f"sedangkan kategori dengan volume tertinggi, {category_name(top_volume_category)}, rata-rata "
            f"{fmt_rs(volume_leader_price, 0)} per item. Perbedaan harga ini membantu menjelaskan "
            "mengapa peringkat volume dan revenue tidak selalu sama."
        )
    with price_right:
        st.markdown("#### Rincian seluruh kategori")
        category_table = category_summary.copy()
        category_table.index = [category_name(name) for name in category_table.index]
        category_table = category_table.reset_index(names="Kategori").rename(
            columns={
                "jumlah": "Item terjual",
                "revenue": "Revenue (R$)",
                "rata_harga": "Rata-rata harga (R$)",
                "pangsa_revenue": "Pangsa revenue (%)",
            }
        )
        show_dataframe(
            category_table,
            number_columns(
                category_table,
                {
                    "Item terjual": "Item terjual",
                    "Revenue (R$)": "Revenue (R$)",
                    "Rata-rata harga (R$)": "Rata-rata harga (R$)",
                    "Pangsa revenue (%)": "Pangsa revenue (%)",
                },
            ),
            height=420,
        )
        total_items = int(category_summary["jumlah"].sum())
        insight(
            f"Tabel memuat {fmt_num(len(category_summary))} kategori dan {fmt_num(total_items)} item. "
            f"{category_name(top_revenue_category)} menyumbang pangsa revenue terbesar, "
            f"{fmt_num(category_summary.loc[top_revenue_category, 'pangsa_revenue'], 1)}%; gunakan "
            "pangsa ini bersama volume untuk menyusun prioritas kategori."
        )


with tab_advanced:
    st.subheader("Binning cicilan kartu kredit")
    if "credit_card" not in selected_methods:
        st.info("Pilih Credit Card pada filter metode pembayaran untuk melihat analisis cicilan.")
    else:
        credit_card = payment_filtered.loc[payment_filtered["payment_type"].eq("credit_card")].copy()
        if credit_card.empty:
            st.info("Tidak ada transaksi Credit Card pada rentang tanggal pembayaran yang dipilih.")
        else:
            credit_card["kelompok_cicilan"] = pd.cut(
                credit_card["payment_installments"],
                bins=[0, 1, 6, 12, np.inf],
                labels=INSTALLMENT_LABELS,
            )
            installment_summary = (
                credit_card.groupby("kelompok_cicilan", observed=False)
                .agg(
                    transaksi=("payment_value", "size"),
                    rata_rata_nilai=("payment_value", "mean"),
                    total_nilai=("payment_value", "sum"),
                )
                .reindex(INSTALLMENT_LABELS, fill_value=0)
            )
            installment_summary["persentase_transaksi"] = (
                installment_summary["transaksi"] / len(credit_card) * 100
            )
            credit_card_value = credit_card["payment_value"].sum()
            installment_summary["persentase_nilai"] = (
                installment_summary["total_nilai"] / credit_card_value * 100
                if credit_card_value
                else 0
            )
            installment_summary = installment_summary.reset_index(names="Kelompok cicilan")
            installment_left, installment_right = st.columns(2, gap="large")
            with installment_left:
                positions = np.arange(len(installment_summary))
                width = 0.36
                fig, ax = plt.subplots(figsize=(8.2, 5.1))
                bars_count = ax.bar(
                    positions - width / 2,
                    installment_summary["persentase_transaksi"],
                    width,
                    label="% transaksi",
                    color=ACCENT,
                )
                bars_value = ax.bar(
                    positions + width / 2,
                    installment_summary["persentase_nilai"],
                    width,
                    label="% nilai",
                    color=GRAY,
                )
                for bars in (bars_count, bars_value):
                    for bar in bars:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.8,
                            f"{fmt_num(bar.get_height(), 1)}%",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color=TEXT,
                        )
                ax.set_xticks(positions, installment_summary["Kelompok cicilan"], rotation=12, ha="right")
                ax.set_ylim(0, max(100, float(installment_summary[["persentase_transaksi", "persentase_nilai"]].to_numpy().max()) * 1.2))
                ax.set_ylabel("Porsi dari transaksi/nilai kartu kredit (%)")
                ax.set_xlabel("Kelompok cicilan")
                ax.set_title("Porsi jumlah transaksi dan nilai menurut cicilan", loc="left", fontsize=12, fontweight="bold", pad=14)
                ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
                ax.set_axisbelow(True)
                ax.legend(frameon=False)
                set_figure_style(fig, ax)
                fig.tight_layout()
                render_figure(fig)
                medium = installment_summary.loc[
                    installment_summary["Kelompok cicilan"].eq("Cicilan menengah (7-12x)")
                ].iloc[0]
                insight(
                    f"Cicilan menengah (7–12x) mencakup {fmt_num(medium['persentase_transaksi'], 1)}% "
                    f"transaksi tetapi {fmt_num(medium['persentase_nilai'], 1)}% nilai Credit Card. "
                    "Penawaran cicilan menengah saat checkout dapat mendukung pembelian bernilai lebih tinggi."
                )
            with installment_right:
                st.markdown("#### Ringkasan kelompok cicilan")
                installment_table = installment_summary.rename(
                    columns={
                        "transaksi": "Jumlah transaksi",
                        "rata_rata_nilai": "Rata-rata nilai (R$)",
                        "total_nilai": "Total nilai (R$)",
                        "persentase_transaksi": "Transaksi (%)",
                        "persentase_nilai": "Nilai (%)",
                    }
                )
                show_dataframe(
                    installment_table,
                    number_columns(
                        installment_table,
                        {
                            "Jumlah transaksi": "Jumlah transaksi",
                            "Rata-rata nilai (R$)": "Rata-rata nilai (R$)",
                            "Total nilai (R$)": "Total nilai (R$)",
                            "Transaksi (%)": "Transaksi (%)",
                            "Nilai (%)": "Nilai (%)",
                        },
                    ),
                )
                largest_mean = installment_summary.loc[installment_summary["rata_rata_nilai"].idxmax()]
                insight(
                    f"Rata-rata nilai tertinggi ada pada {largest_mean['Kelompok cicilan']}, "
                    f"sebesar {fmt_rs(largest_mean['rata_rata_nilai'])}; bandingkan dengan porsinya "
                    "untuk menilai kontribusi kelompok cicilan terhadap transaksi bernilai besar."
                )

    st.divider()
    st.subheader("Segmentasi kuadran kategori produk")
    top_quadrant = category_summary.sort_values("revenue", ascending=False).head(top_n).copy()
    median_volume = float(top_quadrant["jumlah"].median())
    median_price = float(top_quadrant["rata_harga"].median())

    def segment_row(row):
        if row["jumlah"] >= median_volume and row["rata_harga"] >= median_price:
            return "Bintang (volume & harga tinggi)"
        if row["jumlah"] >= median_volume:
            return "Volume tinggi, harga rendah"
        if row["rata_harga"] >= median_price:
            return "Premium, volume rendah"
        return "Potensi rendah"

    top_quadrant["segmen"] = top_quadrant.apply(segment_row, axis=1)
    strategy = {
        "Bintang (volume & harga tinggi)": "Pertahankan stok dan kualitas",
        "Volume tinggi, harga rendah": "Prioritaskan ketersediaan stok",
        "Premium, volume rendah": "Prioritaskan promosi untuk menaikkan volume",
        "Potensi rendah": "Pantau dan evaluasi",
    }
    quadrant_left, quadrant_right = st.columns(2, gap="large")
    with quadrant_left:
        fig, ax = plt.subplots(figsize=(8.2, 6.4))
        max_revenue = max(float(top_quadrant["revenue"].max()), 1.0)
        for segment in SEGMENT_ORDER:
            selected = top_quadrant.loc[top_quadrant["segmen"].eq(segment)]
            if selected.empty:
                continue
            ax.scatter(
                selected["jumlah"],
                selected["rata_harga"],
                s=selected["revenue"] / max_revenue * 520,
                color=SEGMENT_COLORS[segment],
                alpha=0.78,
                edgecolors="white",
                linewidths=0.8,
                label=segment,
            )
            for category, row in selected.iterrows():
                ax.annotate(
                    category_name(category),
                    (row["jumlah"], row["rata_harga"]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=TEXT,
                )
        ax.axvline(median_volume, color=MUTED, linestyle="--", linewidth=1)
        ax.axhline(median_price, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Jumlah item terjual")
        ax.set_ylabel("Rata-rata harga per item (R$)")
        ax.yaxis.set_major_formatter(format_axis_number(0, currency=True))
        ax.xaxis.set_major_formatter(format_axis_number(0))
        ax.set_title(f"Kuadran {top_n} kategori berdasarkan volume dan harga", loc="left", fontsize=12, fontweight="bold", pad=14)
        ax.grid(color="#E5E7EB", linewidth=0.6)
        ax.set_axisbelow(True)
        handles = [Patch(facecolor=SEGMENT_COLORS[name], label=name) for name in SEGMENT_ORDER]
        ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="best")
        set_figure_style(fig, ax)
        fig.tight_layout()
        render_figure(fig)
        segment_revenue = top_quadrant.groupby("segmen")["revenue"].sum().sort_values(ascending=False)
        leading_segment = segment_revenue.index[0]
        leading_share = float(segment_revenue.iloc[0] / top_quadrant["revenue"].sum() * 100)
        insight(
            f"Median pembatas kuadran adalah {fmt_num(median_volume)} item dan "
            f"{fmt_rs(median_price, 1)} per item. Segmen {leading_segment} menyumbang "
            f"{fmt_num(leading_share, 1)}% revenue dari {fmt_num(top_n)} kategori terpilih; "
            "ukuran gelembung menunjukkan revenue relatif tiap kategori."
        )
    with quadrant_right:
        st.markdown("#### Kategori dan strategi per segmen")
        quadrant_table = top_quadrant.copy()
        quadrant_table.index = [category_name(name) for name in quadrant_table.index]
        quadrant_table = quadrant_table.reset_index(names="Kategori").rename(
            columns={
                "jumlah": "Item terjual",
                "rata_harga": "Rata-rata harga (R$)",
                "revenue": "Revenue (R$)",
                "segmen": "Segmen",
            }
        )
        quadrant_table["Strategi"] = quadrant_table["Segmen"].map(strategy)
        show_dataframe(
            quadrant_table,
            number_columns(
                quadrant_table,
                {
                    "Item terjual": "Item terjual",
                    "Rata-rata harga (R$)": "Rata-rata harga (R$)",
                    "Revenue (R$)": "Revenue (R$)",
                },
            ),
            height=420,
        )
        segment_counts = quadrant_table["Segmen"].value_counts()
        segment_detail = ", ".join(
            f"{segment}: {fmt_num(count)} kategori"
            for segment, count in segment_counts.items()
        )
        insight(
            f"Pembagian pada top-{top_n} saat ini adalah {segment_detail}. Gunakan strategi di tabel "
            "untuk membedakan kebutuhan stok, promosi, dan evaluasi."
        )


with tab_conclusion:
    st.subheader("Kesimpulan analisis notebook")
    st.caption(
        "Ringkasan berikut berasal dari analisis notebook dengan periode pembayaran 2017–2018 "
        "dan kategori produk 2018. Isinya statis dan tidak berubah mengikuti filter dashboard."
    )
    st.markdown(
        """
        **Metode pembayaran**

        Credit Card merupakan metode paling banyak digunakan, dengan porsi 73,9%.  
        Dominasinya stabil dan sedikit menguat dari 72,7% pada 2017 menjadi 74,9% pada 2018; tidak ada pergeseran besar. Debit Card tumbuh paling tajam secara relatif, tetapi porsinya masih kecil.

        **Kategori produk**

        Health Beauty menjadi kategori terlaris dari sisi volume dan revenue. Bed Bath Table menyamai volume penjualannya, tetapi revenuenya lebih rendah karena rata-rata harga lebih rendah. Watches Gifts berada pada posisi premium dengan rata-rata harga tertinggi.

        **Rekomendasi**

        1. Perkuat kemitraan kartu kredit dan jaga opsi Credit Card tetap mudah ditemukan saat checkout.
        2. Pantau pertumbuhan Debit Card dan dorong penggunaannya sambil mengukur perubahan porsinya.
        3. Tonjolkan cicilan 7–12x saat checkout untuk produk bernilai tinggi.
        4. Prioritaskan ketersediaan stok untuk kategori dengan volume tinggi.
        5. Tingkatkan promosi kategori premium dengan volume rendah, terutama Watches Gifts.
        """
    )
