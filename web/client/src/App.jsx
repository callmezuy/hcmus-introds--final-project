import React from "react";
import TopRecommendations from "./components/TopRecommendations";
import Board from "./components/Board";

function App() {
  return (
    <div className='min-h-screen flex flex-col bg-[#0B0F1A] font-inter'>
      <header className='bg-[#1e293b] text-white py-8 px-5 shadow-lg'>
        <div className='max-w-350 mx-auto'>
          <h1 className='text-3xl md:text-[36px] font-extrabold mb-2'>
            🇻🇳 Intelligent Stock Advisory System
          </h1>
          <p className='text-base md:text-[16px] opacity-95 font-light'>
            Vietnamese Market Analysis & Recommendations
          </p>
        </div>
      </header>

      <main className='flex-1 py-8 px-5'>
        <div className='max-w-350 mx-auto'>
          <TopRecommendations />
          <Board />
        </div>
      </main>

      <footer className='bg-[#1e293b] text-white text-center p-5 mt-10'>
        <p className='text-sm opacity-80'>
          © 2025 Intelligent Stock Advisory System. Data from VNStock API.
        </p>
      </footer>
    </div>
  );
}

export default App;
