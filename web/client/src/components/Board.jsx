import React from "react";
import { stockAPI } from "../services/api";
import { useEffect, useState } from "react";
import Spinner from "./Spinner";
import {
  ResponsiveContainer,
  ComposedChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  Line,
  Bar,
  Brush,
} from "recharts";

const numberFormat = (value) => new Intl.NumberFormat("vi-VN").format(value);

const priceClass = (delta) => {
  if (delta > 0) return "text-emerald-400";
  if (delta < 0) return "text-red-400";
  return "text-yellow-300";
};

const Board = () => {
  // Removed sample rows; strictly show loading or empty/data states

  const [data, setData] = React.useState([]);
  const [historyMap, setHistoryMap] = React.useState({});
  const [isLoading, setIsLoading] = React.useState(false);
  const [page, setPage] = React.useState(1);
  const [selectedSymbol, setSelectedSymbol] = React.useState(null);
  const pageSize = 10;

  const normalizeHistoryToRows = (historyObj) => {
    const rows = [];
    const makeEmptyRow = (sym) => ({
      symbol: sym,
      tran: 0,
      san: 0,
      tc: 0,
      buy: { g3: 0, kl3: 0, g2: 0, kl2: 0, g1: 0, kl1: 0 },
      match: { price: 0, vol: 0, delta: 0, deltaPct: 0 },
      sell: { g1: 0, kl1: 0, g2: 0, kl2: 0, g3: 0, kl3: 0 },
      totalVol: 0,
      high: 0,
      low: 0,
      foreign: { buy: 0, sell: 0, room: 0 },
    });

    if (!historyObj || typeof historyObj !== "object") return rows;

    for (const [sym, series] of Object.entries(historyObj)) {
      if (!Array.isArray(series) || series.length === 0) {
        rows.push(makeEmptyRow(sym));
        continue;
      }

      const last = series[series.length - 1] || {};
      const prev = series.length >= 2 ? series[series.length - 2] : last;
      const totalVol = series.reduce((sum, d) => sum + (d.volume || 0), 0);
      const highs = series.map((d) => d.high ?? d.close ?? 0);
      const lows = series.map((d) => d.low ?? d.close ?? 0);
      const high = highs.length
        ? Math.max(...highs)
        : last.high ?? last.close ?? 0;
      const low = lows.length ? Math.min(...lows) : last.low ?? last.close ?? 0;
      const lastClose = last.close ?? 0;
      const prevClose = prev.close ?? last.open ?? lastClose;
      const delta = lastClose - (prevClose || 0);
      const deltaPct = prevClose ? (delta / prevClose) * 100 : 0;
      rows.push({
        symbol: sym,
        tran: high,
        san: low,
        tc: prevClose || lastClose,
        buy: {
          g3: lastClose,
          kl3: 0,
          g2: lastClose,
          kl2: 0,
          g1: lastClose,
          kl1: last.volume || 0,
        },
        match: { price: lastClose, vol: last.volume || 0, delta, deltaPct },
        sell: {
          g1: lastClose,
          kl1: 0,
          g2: lastClose,
          kl2: 0,
          g3: lastClose,
          kl3: 0,
        },
        totalVol: totalVol,
        high,
        low,
        foreign: { buy: 0, sell: 0, room: 0 },
      });
    }
    return rows;
  };

  React.useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      try {
        const result = await stockAPI.getTop100History(30);
        const dataObj = result?.data || {};
        const rows = normalizeHistoryToRows(dataObj);
        setData(rows);
        setHistoryMap(dataObj);
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchData();
  }, []);

  const rows = Array.isArray(data) ? data : [];
  const totalPages = Math.max(1, Math.ceil((rows.length || 0) / pageSize));
  const visibleRows = rows.slice((page - 1) * pageSize, page * pageSize);

  if (isLoading) {
    return (
      <div className="mt-6 rounded-xl border border-slate-700 bg-slate-900 p-6">
        <h2 className="text-slate-200 text-xl font-semibold mb-2">Bảng giá Top 100</h2>
        <Spinner text="Đang tải dữ liệu bảng…" />
      </div>
    );
  }

  if (!rows.length) {
    return (
      <div className="mt-6 rounded-xl border border-slate-700 bg-slate-900 p-6">
        <h2 className="text-slate-200 text-xl font-semibold mb-2">Bảng giá Top 100</h2>
        <p className="text-slate-400 text-sm">Không có dữ liệu.</p>
      </div>
    );
  }

  return (
    <div className='mt-6 overflow-x-auto rounded-xl border border-slate-700 bg-slate-900'>
      <table className='min-w-max w-full text-xs'>
        <thead className='bg-slate-800 text-slate-200'>
          <tr>
            <th className='sticky left-0 bg-slate-800 px-3 py-2 text-left font-semibold'>
              CK
            </th>
            <th className='px-3 py-2 font-semibold'>Trần</th>
            <th className='px-3 py-2 font-semibold'>Sàn</th>
            <th className='px-3 py-2 font-semibold'>TC</th>
            <th className='px-3 py-2 font-semibold' colSpan={6}>
              Bên mua
            </th>
            <th className='px-3 py-2 font-semibold' colSpan={4}>
              Khớp lệnh
            </th>
            <th className='px-3 py-2 font-semibold' colSpan={6}>
              Bên bán
            </th>
            <th className='px-3 py-2 font-semibold'>Tổng KL</th>
            <th className='px-3 py-2 font-semibold'>Cao</th>
            <th className='px-3 py-2 font-semibold'>Thấp</th>

          </tr>
          <tr className='bg-slate-800 text-slate-400'>
            <th className='sticky left-0 bg-slate-800 px-3 py-2'></th>
            <th className='px-3 py-2'></th>
            <th className='px-3 py-2'></th>
            <th className='px-3 py-2'></th>
            {/* Bên mua */}
            <th className='px-3 py-2'>Giá 3</th>
            <th className='px-3 py-2'>KL 3</th>
            <th className='px-3 py-2'>Giá 2</th>
            <th className='px-3 py-2'>KL 2</th>
            <th className='px-3 py-2'>Giá 1</th>
            <th className='px-3 py-2'>KL 1</th>
            {/* Khớp lệnh */}
            <th className='px-3 py-2'>Giá</th>
            <th className='px-3 py-2'>KL</th>
            <th className='px-3 py-2'>+/−</th>
            <th className='px-3 py-2'>+/− (%)</th>
            {/* Bên bán */}
            <th className='px-3 py-2'>Giá 1</th>
            <th className='px-3 py-2'>KL 1</th>
            <th className='px-3 py-2'>Giá 2</th>
            <th className='px-3 py-2'>KL 2</th>
            <th className='px-3 py-2'>Giá 3</th>
            <th className='px-3 py-2'>KL 3</th>
            {/* Others */}
            <th className='px-3 py-2'>Tổng KL</th>
            <th className='px-3 py-2'>Cao</th>
            <th className='px-3 py-2'>Thấp</th>

          </tr>
        </thead>
        <tbody>
          {visibleRows.map((r) => (
            <tr
              key={r.symbol}
              className={`border-t border-slate-800 bg-slate-900 hover:bg-slate-800 cursor-pointer ${
                selectedSymbol === r.symbol ? "ring-1 ring-indigo-500" : ""
              }`}
              onClick={() => setSelectedSymbol(r.symbol)}
            >
              <td className='sticky left-0 bg-slate-900 px-3 py-2 text-left font-bold text-red-400'>
                {r.symbol}
              </td>
              <td className='px-3 py-2 text-purple-300'>{r.tran.toFixed(2)}</td>
              <td className='px-3 py-2 text-cyan-300'>{r.san.toFixed(2)}</td>
              <td className='px-3 py-2 text-yellow-300'>{r.tc.toFixed(2)}</td>
              {/* Bên mua */}
              <td className='px-3 py-2 text-slate-200'>
                {r.buy.g3.toFixed(2)}
              </td>
              <td className='px-3 py-2 text-red-400'>
                {numberFormat(r.buy.kl3)}
              </td>
              <td className='px-3 py-2 text-slate-200'>
                {r.buy.g2.toFixed(2)}
              </td>
              <td className='px-3 py-2 text-red-400'>
                {numberFormat(r.buy.kl2)}
              </td>
              <td className='px-3 py-2 text-slate-200'>
                {r.buy.g1.toFixed(2)}
              </td>
              <td className='px-3 py-2 text-red-400'>
                {numberFormat(r.buy.kl1)}
              </td>
              {/* Khớp lệnh */}
              <td className={`px-3 py-2 ${priceClass(r.match.delta)}`}>
                {r.match.price.toFixed(2)}
              </td>
              <td className='px-3 py-2 text-slate-200'>
                {numberFormat(r.match.vol)}
              </td>
              <td
                className={
                  r.match.delta >= 0
                    ? "px-3 py-2 text-emerald-400"
                    : "px-3 py-2 text-red-400"
                }
              >
                {r.match.delta > 0 ? "+" : ""}
                {r.match.delta.toFixed(2)}
              </td>
              <td
                className={
                  r.match.deltaPct >= 0
                    ? "px-3 py-2 text-emerald-400"
                    : "px-3 py-2 text-red-400"
                }
              >
                {r.match.deltaPct > 0 ? "+" : ""}
                {r.match.deltaPct.toFixed(2)}%
              </td>
              {/* Bên bán */}
              <td className='px-3 py-2 text-slate-200'>
                {r.sell.g1.toFixed(2)}
              </td>
              <td className='px-3 py-2 text-green-400'>
                {numberFormat(r.sell.kl1)}
              </td>
              <td className='px-3 py-2 text-yellow-300'>
                {r.sell.g2.toFixed(2)}
              </td>
              <td className='px-3 py-2 text-green-400'>
                {numberFormat(r.sell.kl2)}
              </td>
              <td className='px-3 py-2 text-slate-200'>
                {r.sell.g3.toFixed(2)}
              </td>
              <td className='px-3 py-2 text-green-400'>
                {numberFormat(r.sell.kl3)}
              </td>
              {/* Others */}
              <td className='px-3 py-2 text-slate-200'>
                {numberFormat(r.totalVol)}
              </td>
              <td className='px-3 py-2 text-emerald-400'>
                {r.high.toFixed(2)}
              </td>
              <td className='px-3 py-2 text-red-400'>{r.low.toFixed(2)}</td>

            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination Controls */}
      <div className='bg-[#161B26] border-t border-slate-700 px-4 py-3 flex items-center justify-between'>
        <div className='text-sm text-slate-300'>
          Trang {page} / {totalPages} • Tổng: {rows.length} mã
        </div>
        <div className='flex gap-2'>
          <button
            className='px-3 py-1 rounded border border-slate-600 bg-[#1e293b] hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 text-sm'
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
          >
            ← Trước
          </button>
          <button
            className='px-3 py-1 rounded border border-slate-600 bg-[#1e293b] hover:bg-slate-700 disabled:opacity-50 disabled:cursor-not-allowed text-slate-200 text-sm'
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
          >
            Sau →
          </button>
        </div>
      </div>

      {selectedSymbol && Array.isArray(historyMap[selectedSymbol]) && (
  <div className='bg-[#0B0F1A] border-t border-slate-800 p-6 rounded-b-xl shadow-2xl'>
    <div className='flex items-center justify-between mb-6'>
      <div className="flex items-center">
        {/* Thanh trang trí màu sắc xanh theo style AI */}
        <div className="w-1 h-6 bg-gradient-to-b from-emerald-400 to-cyan-500 rounded-full mr-3 shadow-[0_0_10px_rgba(16,185,129,0.4)]"></div>
        <h3 className='text-base font-bold text-slate-100 uppercase tracking-wider'>
          Lịch sử phân tích: <span className='text-emerald-400'>{selectedSymbol}</span>
        </h3>
      </div>
      <button
        className='text-[10px] uppercase font-bold tracking-widest px-4 py-2 border border-slate-700 rounded-md text-slate-400 hover:bg-slate-800 hover:text-emerald-400 hover:border-emerald-500/50 transition-all duration-300'
        onClick={() => setSelectedSymbol(null)}
      >
        Đóng
      </button>
    </div>

    <div className='h-80 w-full'>
      <ResponsiveContainer width='100%' height='100%'>
        <ComposedChart
          data={historyMap[selectedSymbol] || []}
          margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
        >
          {/* Định nghĩa Gradient cho đường Line đổ bóng phía dưới */}
          <defs>
            <linearGradient id="lineGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
            </linearGradient>
          </defs>

          {/* Lưới: stroke hẹp và mờ */}
          <CartesianGrid strokeDasharray='3 3' stroke="#1E293B" vertical={false} />
          
          <XAxis
            dataKey='time'
            tickFormatter={(iso) => new Date(iso).toLocaleDateString("vi-VN")}
            tick={{ fontSize: 10, fill: '#64748B', fontWeight: 500 }}
            axisLine={{ stroke: '#1E293B' }}
            tickLine={false}
            dy={10}
          />
          
          <YAxis
            yAxisId='price'
            domain={["auto", "auto"]}
            tickFormatter={(v) => v.toLocaleString()}
            tick={{ fontSize: 10, fill: '#64748B', fontWeight: 500 }}
            axisLine={false}
            tickLine={false}
          />

          {/* Tooltip: Quan trọng nhất để xóa màu trắng */}
          <Tooltip
            contentStyle={{
              backgroundColor: "#161B26",
              border: "1px solid #334155",
              borderRadius: "8px",
              boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.5)",
            }}
            itemStyle={{ fontSize: '12px', color: '#10B981', fontWeight: 'bold' }}
            labelStyle={{ color: '#94A3B8', fontSize: '11px', marginBottom: '4px' }}
            cursor={{ stroke: '#334155', strokeWidth: 1 }}
            labelFormatter={(label) => `Thời gian: ${new Date(label).toLocaleDateString("vi-VN")}`}
          />
          
          {/* Chú thích (Legend): Chỉnh màu chữ thành Slate-300 */}
          <Legend 
            verticalAlign="top" 
            align="right"
            iconType="diamond"
            wrapperStyle={{ paddingBottom: '20px', color: '#CBD5E1', fontSize: '11px' }} 
          />

          {/* Đường Line chính */}
          <Line
            yAxisId='price'
            type='linear'
            dataKey='close' /* Thường dùng giá đóng cửa (close) thay vì open cho lịch sử */
            stroke='#10B981'
            strokeWidth={2.5}
            dot={{ r: 0 }} /* Ẩn các chấm mặc định để trông mượt hơn */
            activeDot={{ r: 5, fill: '#10B981', stroke: '#fff', strokeWidth: 2 }}
            name='Giá Đóng Cửa'
            style={{ filter: "drop-shadow(0px 4px 8px rgba(16, 185, 129, 0.3))" }}
            animationDuration={1000}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  </div>
)}
    </div>
  );
};

export default Board;
