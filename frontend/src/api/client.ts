/** APIクライアント – FastAPI /ui/* エンドポイントへのfetchラッパー */

export interface Summary {
    period_from: string;
    period_to: string;
    n_races: number;
    n_bets: number;
    n_hits: number;
    hit_rate: number;
    total_bet: number;
    total_return: number;
    roi: number;
    max_drawdown: number;
    logloss: number;
    auc: number;
}

export interface MonthlyRow {
    month: string;
    n_bets: number;
    n_hits: number;
    roi: number;
}

export interface BetRow {
    race_date: string;
    race_id: number;
    horse_name: string;
    horse_no: number;
    p_win: number;
    odds_final: number;
    ev_profit: number;
    is_hit: boolean;
    payout: number;
    profit: number;
}

export interface BetsResponse {
    total: number;
    page: number;
    page_size: number;
    total_pages: number;
    items: BetRow[];
}

async function get<T>(path: string): Promise<T> {
    const res = await fetch(path);
    if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail?.detail ?? `HTTP ${res.status}`);
    }
    return res.json() as Promise<T>;
}

export const api = {
    summary: () => get<Summary>("/ui/summary"),
    monthly: () => get<MonthlyRow[]>("/ui/monthly"),
    bets: (page: number, hit: "all" | "hit" | "miss") =>
        get<BetsResponse>(`/ui/bets?page=${page}&hit=${hit}`),
};
