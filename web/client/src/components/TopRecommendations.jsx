import React, { useEffect, useMemo, useState } from "react";
import { stockAPI } from "../services/api";
import Spinner from "./Spinner";

const PAGE_SIZE = 10; // items per page per group

const Pager = ({ page, total, onChange }) => {
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  return (
    <div className="flex items-center justify-center gap-2 mt-3">
      <button
        className="px-3 py-1 rounded border border-slate-600 bg-[#161B26] hover:bg-slate-700 disabled:opacity-50 text-slate-200"
        disabled={page <= 1}
        onClick={() => onChange(page - 1)}
      >
        Trước
      </button>
      <span className="text-slate-300 text-sm">
        Trang {page} / {totalPages}
      </span>
      <button
        className="px-3 py-1 rounded border border-slate-600 bg-[#161B26] hover:bg-slate-700 disabled:opacity-50 text-slate-200"
        disabled={page >= totalPages}
        onClick={() => onChange(page + 1)}
      >
        Sau
      </button>
    </div>
  );
};

const GroupList = ({ title, items, page, onPageChange, borderColor, titleColor }) => {
  const start = (page - 1) * PAGE_SIZE;
  const slice = items.slice(start, start + PAGE_SIZE);
  return (
    <div className={`mb-6 p-5 rounded-lg bg-[#161B26] shadow-lg border border-slate-700 ${borderColor} border-l-4`}>
      <h3 className={`text-lg font-bold mb-4 ${titleColor}`}>{title}</h3>
      {slice.length === 0 ? (
        <p className="text-slate-400 text-sm">Không có dữ liệu.</p>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3">
          {slice.map((row, idx) => (
            <div key={`${row.symbol}-${idx}`} className="bg-[#1e293b] rounded-lg p-3 border border-slate-600 hover:border-slate-500 transition">
              <div className="text-center">
                <div className="text-base font-bold text-slate-100 mb-1">{row.symbol}</div>
              </div>
            </div>
          ))}
        </div>
      )}
      <Pager page={page} total={items.length} onChange={onPageChange} />
    </div>
  );
};

const TopRecommendations = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [g1Page, setG1Page] = useState(1);
  const [g2Page, setG2Page] = useState(1);
  const [g3Page, setG3Page] = useState(1);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Prefer live API compute; fallback to CSV if it fails
        let res;
        try {
          // res = await stockAPI.getTopRecommendations({ source: "VNStock", limit: 100, save_csv: false, days: 60 });
        // } catch (e1) {
          res = await stockAPI.getTopRecommendationsCsv(100, true);
        }
        catch (e2) {
          throw new Error("Không thể lấy dữ liệu đề xuất");
        }
        const rows = Array.isArray(res?.data) ? res.data : [];
        // Filter valid rows (status ok and numeric prob)
        const cleaned = rows
          .filter((r) => r && r.status === "ok")
          .map((r) => ({
            symbol: String(r.symbol || "").toUpperCase(),
            date: r.date || null,
            prediction: r.prediction,
            prob_buy: typeof r.prob_buy === "number" ? r.prob_buy : (r.prob_buy != null ? Number(r.prob_buy) : null),
            status: r.status,
          }));
        setData(cleaned);
      } catch (e) {
        console.error(e);
        setError("Không thể lấy dữ liệu đề xuất");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const groups = useMemo(() => {
    const g1 = [];
    const g2 = [];
    const g3 = [];
    for (const row of data) {
      const p = row.prob_buy;
      if (p != null && p >= 0.7) g1.push(row);
      else if (p != null && p >= 0.6) g2.push(row);
      else g3.push(row);
    }
    return { g1, g2, g3 };
  }, [data]);

  if (loading) {
    return (
      <div className="mb-10 p-5 rounded-lg bg-[#161B26] shadow-lg border border-slate-700">
        <h2 className="text-slate-200 text-2xl md:text-[28px] font-bold text-center mb-2">Khuyến nghị</h2>
        <Spinner text="Đang tải khuyến nghị…" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-10 p-5 rounded-lg bg-red-900/20 border border-red-700">
        <h2 className="text-red-400 text-2xl md:text-[28px] font-bold text-center mb-6">{error}</h2>
      </div>
    );
  }

  return (
    <div className="mb-10">
      <h2 className="text-slate-100 text-2xl md:text-[28px] font-bold text-center mb-6"> Khuyến nghị theo nhóm xác suất</h2>

      <GroupList
        title="Cổ phiếu đáng mua"
        items={groups.g1}
        page={g1Page}
        onPageChange={setG1Page}
        borderColor="border-emerald-500"
        titleColor="text-emerald-400"
      />

      <GroupList
        title="Đáng cân nhắc"
        items={groups.g2}
        page={g2Page}
        onPageChange={setG2Page}
        borderColor="border-amber-500"
        titleColor="text-amber-400"
      />

      <GroupList
        title="Chỉ nên theo dõi"
        items={groups.g3}
        page={g3Page}
        onPageChange={setG3Page}
        borderColor="border-rose-500"
        titleColor="text-rose-400"
      />
    </div>
  );
};

export default TopRecommendations;
