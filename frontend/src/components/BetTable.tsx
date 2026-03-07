import { useEffect, useState } from "react";
import { api, type BetRow, type BetsResponse } from "../api/client";

type HitFilter = "all" | "hit" | "miss";

const FILTER_LABELS: { key: HitFilter; label: string }[] = [
    { key: "all", label: "全件" },
    { key: "hit", label: "的中のみ" },
    { key: "miss", label: "外れのみ" },
];

export default function BetTable() {
    const [data, setData] = useState<BetsResponse | null>(null);
    const [page, setPage] = useState(1);
    const [filter, setFilter] = useState<HitFilter>("all");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        setLoading(true);
        setError(null);
        api
            .bets(page, filter)
            .then(setData)
            .catch((e: Error) => setError(e.message))
            .finally(() => setLoading(false));
    }, [page, filter]);

    const handleFilter = (f: HitFilter) => {
        setFilter(f);
        setPage(1);
    };

    return (
        <>
            {/* フィルタボタン */}
            <div className="table-controls">
                {FILTER_LABELS.map(({ key, label }) => (
                    <button
                        key={key}
                        className={`filter-btn ${filter === key ? "active" : ""}`}
                        onClick={() => handleFilter(key)}
                        id={`filter-${key}`}
                    >
                        {label}
                    </button>
                ))}
                {data && (
                    <span style={{ marginLeft: "auto", color: "#8b95b0", fontSize: 13 }}>
                        {data.total} 件
                    </span>
                )}
            </div>

            {/* テーブル */}
            {loading ? (
                <div className="loading">読み込み中...</div>
            ) : error ? (
                <div className="error-box">{error}</div>
            ) : (
                <div className="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>日付</th>
                                <th>馬名</th>
                                <th>組番/馬番</th>
                                <th>P(的中)</th>
                                <th>オッズ</th>
                                <th>EV</th>
                                <th>ベット額</th>
                                <th>結果</th>
                                <th>損益</th>
                            </tr>
                        </thead>
                        <tbody>
                            {data?.items.map((row: BetRow, i) => (
                                <tr key={i}>
                                    <td style={{ color: "#8b95b0" }}>{row.race_date}</td>
                                    <td style={{ fontWeight: 500 }}>{row.horse_name}</td>
                                    <td style={{ color: "#8b95b0" }}>{row.kumiban ? row.kumiban : `${row.horse_no}番`}</td>
                                    <td>{(row.p_win * 100).toFixed(1)}%</td>
                                    <td>{row.odds_final.toFixed(1)}倍</td>
                                    <td
                                        style={{
                                            color: row.ev_profit > 0 ? "#10b981" : "#ef4444",
                                        }}
                                    >
                                        {row.ev_profit > 0 ? "+" : ""}
                                        {(row.ev_profit * 100).toFixed(1)}%
                                    </td>
                                    <td>{row.bet_yen ? `${row.bet_yen.toLocaleString()}円` : "100円"}</td>
                                    <td>
                                        {row.is_hit ? (
                                            <span className="badge badge-hit">✓ 的中</span>
                                        ) : (
                                            <span className="badge badge-miss">✗ 外れ</span>
                                        )}
                                    </td>
                                    <td
                                        style={{
                                            color: row.profit > 0 ? "#10b981" : "#ef4444",
                                            fontWeight: 600,
                                        }}
                                    >
                                        {row.profit > 0 ? "+" : ""}
                                        {row.profit.toLocaleString()}円
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}

            {/* ページネーション */}
            {data && data.total_pages > 1 && (
                <div className="pagination">
                    <button
                        className="page-btn"
                        id="page-prev"
                        onClick={() => setPage((p) => p - 1)}
                        disabled={page <= 1}
                    >
                        ← 前
                    </button>
                    <span className="page-info">
                        {page} / {data.total_pages} ページ
                    </span>
                    <button
                        className="page-btn"
                        id="page-next"
                        onClick={() => setPage((p) => p + 1)}
                        disabled={page >= data.total_pages}
                    >
                        次 →
                    </button>
                </div>
            )}
        </>
    );
}
