import { useEffect, useState } from "react";
import { api, type Summary, type MonthlyRow } from "./api/client";
import KpiCards from "./components/KpiCards";
import MonthlyRoiChart from "./components/MonthlyRoiChart";
import BetTable from "./components/BetTable";

export default function App() {
    const [summary, setSummary] = useState<Summary | null>(null);
    const [monthly, setMonthly] = useState<MonthlyRow[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        Promise.all([api.summary(), api.monthly()])
            .then(([s, m]) => {
                setSummary(s);
                setMonthly(m);
            })
            .catch((e: Error) => setError(e.message))
            .finally(() => setLoading(false));
    }, []);

    if (loading) {
        return (
            <div className="app">
                <div className="loading">⏳ データを読み込み中...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="app">
                <div className="error-box">
                    <strong>⚠️ エラー:</strong> {error}
                    <br />
                    <small>
                        バックテスト結果ファイルが存在するか確認してください:{" "}
                        <code>data/backtest_result.json</code>
                    </small>
                </div>
            </div>
        );
    }

    return (
        <div className="app">
            {/* ヘッダー */}
            <header className="header">
                <h1 className="header-title">
                    <span className="emoji">🏇</span>
                    競馬予測 バックテスト結果
                </h1>
                {summary && (
                    <span className="header-period">
                        {summary.period_from} 〜 {summary.period_to}
                    </span>
                )}
            </header>

            {/* KPIカード */}
            {summary && <KpiCards summary={summary} />}

            {/* 月別ROIチャート */}
            <section className="section">
                <h2 className="section-title">月別ROI推移</h2>
                <MonthlyRoiChart data={monthly} />
            </section>

            {/* ベット一覧テーブル */}
            <section className="section">
                <h2 className="section-title">ベット一覧</h2>
                <BetTable />
            </section>
        </div>
    );
}
