import type { Summary } from "../api/client";

interface Props {
    summary: Summary;
}

/** 数値を % 表示にフォーマット */
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
/** 円表示（千円単位） */
const yen = (v: number) => `¥${v.toLocaleString()}`;

export default function KpiCards({ summary }: Props) {
    const roiColor =
        summary.roi >= 1.0 ? "kpi-pos" : summary.roi >= 0.9 ? "kpi-neutral" : "kpi-neg";

    return (
        <div className="kpi-grid">
            {/* ROI */}
            <div className="kpi-card green">
                <div className="kpi-label">回収率 (ROI)</div>
                <div className={`kpi-value ${roiColor}`}>
                    {(summary.roi * 100).toFixed(1)}%
                </div>
                <div className="kpi-sub">
                    収支 {yen(summary.total_return - summary.total_bet)}
                </div>
            </div>

            {/* 的中率 */}
            <div className="kpi-card purple">
                <div className="kpi-label">的中率</div>
                <div className="kpi-value kpi-neutral">{pct(summary.hit_rate)}</div>
                <div className="kpi-sub">
                    {summary.n_hits} / {summary.n_bets} ベット
                </div>
            </div>

            {/* ベット数 */}
            <div className="kpi-card blue">
                <div className="kpi-label">ベット数 / 総レース</div>
                <div className="kpi-value kpi-neutral">{summary.n_bets}</div>
                <div className="kpi-sub">
                    対象 {summary.n_races.toLocaleString()} レース 中
                </div>
            </div>

            {/* 最大ドローダウン */}
            <div className="kpi-card red">
                <div className="kpi-label">最大ドローダウン</div>
                <div className="kpi-value kpi-neg">{yen(summary.max_drawdown)}</div>
                <div className="kpi-sub">
                    AUC {summary.auc.toFixed(3)} / Logloss {summary.logloss.toFixed(3)}
                </div>
            </div>
        </div>
    );
}
