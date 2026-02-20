import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ReferenceLine,
    ResponsiveContainer,
    Cell,
} from "recharts";
import type { MonthlyRow } from "../api/client";

interface Props {
    data: MonthlyRow[];
}

/** カスタムツールチップ */
const CustomTooltip = ({
    active,
    payload,
    label,
}: {
    active?: boolean;
    payload?: { value: number }[];
    label?: string;
}) => {
    if (!active || !payload?.length) return null;
    const roi = payload[0].value;
    return (
        <div
            style={{
                background: "#1e2130",
                border: "1px solid rgba(99,102,241,0.4)",
                borderRadius: 8,
                padding: "10px 14px",
                fontSize: 13,
                color: "#f0f2f8",
            }}
        >
            <div style={{ fontWeight: 600, marginBottom: 4 }}>{label}</div>
            <div style={{ color: roi >= 1.0 ? "#10b981" : "#ef4444" }}>
                ROI: {(roi * 100).toFixed(1)}%
            </div>
        </div>
    );
};

export default function MonthlyRoiChart({ data }: Props) {
    if (!data.length) {
        return <div className="loading">データがありません</div>;
    }

    return (
        <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                    dataKey="month"
                    tick={{ fill: "#556080", fontSize: 11 }}
                    tickLine={false}
                    axisLine={{ stroke: "rgba(255,255,255,0.08)" }}
                />
                <YAxis
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: "#556080", fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    width={50}
                />
                <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(99,102,241,0.08)" }} />
                {/* ROI=1.0 の基準線 */}
                <ReferenceLine y={1.0} stroke="rgba(99,102,241,0.5)" strokeDasharray="4 4" />
                <Bar dataKey="roi" radius={[4, 4, 0, 0]}>
                    {data.map((entry, i) => (
                        <Cell
                            key={i}
                            fill={entry.roi >= 1.0 ? "#10b981" : "#ef4444"}
                            fillOpacity={0.85}
                        />
                    ))}
                </Bar>
            </BarChart>
        </ResponsiveContainer>
    );
}
