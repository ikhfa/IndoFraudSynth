"""Calibration targets and their public sources.

Every constant used to shape the synthetic population is listed here with either
an OFFICIAL source (Bank Indonesia / OJK / PPATK publications) or an explicit
ASSUMPTION tag. The validation module compares generated aggregates against these
targets; the dataset card reproduces this table. Verified 16 Nov 2026.
"""

CALIBRATION_SOURCES = [
    {
        "target": "QRIS mean ticket ~ Rp48,000",
        "value": 48_000,
        "kind": "official-derived",
        "source": "Bank Indonesia press release No. 28/190/DKom (17 Sep 2026): "
                  "H1-2026 QRIS = 12.55B transactions, ~Rp600.7T => mean ~ Rp47,865.",
        "url": "https://www.bi.go.id/id/publikasi/ruang-media/news-release/Pages/sp_2819026.aspx",
    },
    {
        "target": "BI-FAST mean ticket ~ Rp2.47M",
        "value": 2_470_000,
        "kind": "official-derived (industry summary; cross-check vs BI SPI before publication)",
        "source": "BI Q2-2026: 1.529B BI-FAST transactions, Rp3,777T => mean ~ Rp2.47M.",
        "url": "https://paymentbrief.com/articles/indonesia-payments-operator-guide/",
    },
    {
        "target": "QRIS-dominant rail mix (count share)",
        "value": {"QRIS": 0.45, "EWALLET": 0.25, "BIFAST": 0.15, "VA": 0.12, "CARD": 0.03},
        "kind": "assumption (anchored on BI digital-payment composition)",
        "source": "BI: 6.11B digital txns/month (Aug 2026); QRIS ~2B+/month; low card penetration. "
                  "Exact split is an assumption; run sensitivity analysis.",
        "url": "https://www.bi.go.id/id/publikasi/ruang-media/news-release/Pages/sp_2819026.aspx",
    },
    {
        "target": "Fraud typologies: impersonation/OTP phishing, 'salah transfer', "
                  "task-commission scams, illegal investment",
        "value": ["ATO", "APP", "MULE", "QRIS_ABUSE"],
        "kind": "official-informed",
        "source": "OJK/Satgas PASTI press release SP 16/STPASTI/VIII/2026 (31 Aug 2026): "
                  "IASC 637,055 reports; 607,210 accounts blocked; Rp724.1B blocked; "
                  "Rp204.3B returned (22 Nov 2024 - 31 Jul 2026); modus operandi list.",
        "url": "https://ojk.go.id/id/berita-dan-kegiatan/info-terkini/Pages/Satgas-PASTI-Terima-25-Ribu-Pengaduan-Aktivitas-Keuangan-Ilegal-dan-IASC-Berhasil-Blokir-Dana-Terkait-Scam-Rp-724-Miliar.aspx",
    },
    {
        "target": "Mule-network / gambling laundering pressure",
        "value": "Rp286.84T judol turnover 2025 (422.1M txns)",
        "kind": "official",
        "source": "PPATK 'Catatan Capaian Strategis PPATK Tahun 2025' (28 Jan 2026).",
        "url": "https://www.ppatk.go.id/siaran_pers/read/1594/",
    },
    {
        "target": "Per-rail fraud rates 0.05%-0.4%",
        "value": {"QRIS": 0.0008, "EWALLET": 0.0025, "BIFAST": 0.004, "VA": 0.0015, "CARD": 0.002},
        "kind": "assumption (plausible range; no official per-rail fraud rate published)",
        "source": "Assumption. Sensitivity analysis over +/-50% required in experiments.",
        "url": None,
    },
    {
        "target": "E-wallet / VA / card mean tickets",
        "value": {"EWALLET": 150_000, "VA": 250_000, "CARD": 500_000},
        "kind": "assumption",
        "source": "Assumption based on typical e-commerce/wallet usage; document and vary.",
        "url": None,
    },
    {
        "target": "Transaction caps: QRIS Rp10M, BI-FAST Rp250M, wallet Rp20M",
        "value": {"QRIS": 10_000_000, "BIFAST": 250_000_000, "EWALLET": 20_000_000},
        "kind": "official-informed (BI product limits; verify current PBI/PADG)",
        "source": "BI QRIS transaction limit Rp10M; BI-FAST limit Rp250M; e-money balance limits.",
        "url": "https://www.bi.go.id",
    },
    {
        "target": "Payday (gajian, 25th-1st) and Harbolnas (11.11, 12.12) seasonality",
        "value": {"payday_mult": 1.6, "promo_mult": 2.6},
        "kind": "assumption (well-documented behavioral pattern)",
        "source": "Assumption; uplift magnitudes are tunable and validated for presence, not exact size.",
        "url": None,
    },
    {
        "target": "Institution size skew (KBMI-4 banks >> BPD >> e-wallet PJP)",
        "value": "bank shares 36/25/15/11/7.5/3.5/2%; wallets 45/32/18%",
        "kind": "assumption (structure official, shares assumed)",
        "source": "OJK KBMI grouping exists; exact market shares are assumptions for silo profiles.",
        "url": "https://ojk.go.id",
    },
]


def calibration_markdown() -> str:
    lines = ["| Target | Kind | Source |", "|---|---|---|"]
    for c in CALIBRATION_SOURCES:
        src = c["source"] + (f" ({c['url']})" if c.get("url") else "")
        lines.append(f"| {c['target']} | {c['kind']} | {src} |")
    return "\n".join(lines)
