import React from "react";

const Spinner = ({ text = "Đang tải…", size = 28 }) => {
  const px = typeof size === "number" ? `${size}px` : size;
  return (
    <div className="flex items-center justify-center gap-3 py-6">
      <div
        className="animate-spin rounded-full border-4 border-slate-600 border-t-transparent"
        style={{ width: px, height: px }}
        aria-label="loading"
      />
      {text && <span className="text-slate-300 text-sm">{text}</span>}
    </div>
  );
};

export default Spinner;
