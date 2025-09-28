import React, { useState } from 'react'
import Chat from "./components/Chat";

const API = 'http://127.0.0.1:8000' // backend FastAPI

export default function App() {
  const [q, setQ] = useState('')
  const [ans, setAns] = useState('')
  const [pred, setPred] = useState(null)
  const [ticker, setTicker] = useState('AAPL')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const ask = async () => {
    setLoading(true); setError(''); setAns('')
    try {
      const r = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
      })
      const j = await r.json()
      if (!r.ok) throw new Error(j.detail || 'Search failed')
      setAns(j.answer || '(no answer)'); 
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setLoading(false)
    }
  }
  function App() {
  return (
    <div className="App">
      <Chat apiUrl="http://127.0.0.1:8000" />
    </div>
  );
}


  const check = async () => {
    setLoading(true); setError(''); setPred(null)
    try {
      const r = await fetch(`${API}/predict?ticker=${encodeURIComponent(ticker)}`)
      const j = await r.json()
      if (!r.ok) throw new Error(j.detail || 'Predict failed')
      setPred(j)
    } catch (e) {
      setError(String(e.message || e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{fontFamily:'system-ui, sans-serif', padding:24, maxWidth:900, margin:'0 auto', color:'#e6e6e6'}}>
      <h1 style={{marginBottom:8}}>Investment Twin</h1>
      <p style={{opacity:.8, marginTop:0}}>Ask Buffett-style questions or score a ticker.</p>

      <div style={{display:'grid', gap:12, gridTemplateColumns:'1fr auto'}}>
        <input
          placeholder="Ask Buffett… e.g., What’s his view on buybacks?"
          value={q}
          onChange={e=>setQ(e.target.value)}
          style={{padding:12, borderRadius:8, border:'1px solid #333', background:'#111', color:'#e6e6e6'}}
        />
        <button onClick={ask} disabled={loading || !q}
          style={{padding:'12px 16px', borderRadius:8, border:'none', background:'#7c5cff', color:'white'}}>
          Ask
        </button>
      </div>

      <div style={{display:'grid', gap:12, gridTemplateColumns:'1fr auto', marginTop:16}}>
        <input
          placeholder="Ticker (e.g., AAPL)"
          value={ticker}
          onChange={e=>setTicker(e.target.value.toUpperCase())}
          style={{padding:12, borderRadius:8, border:'1px solid #333', background:'#111', color:'#e6e6e6'}}
        />
        <button onClick={check} disabled={loading || !ticker}
          style={{padding:'12px 16px', borderRadius:8, border:'none', background:'#00b37e', color:'white'}}>
          Predict
        </button>
      </div>

      {loading && <div style={{marginTop:16}}>Loading…</div>}
      {error && <div style={{marginTop:16, color:'#ff6b6b'}}>Error: {error}</div>}

      {ans && (
        <div style={{whiteSpace:'pre-wrap', background:'#151515', padding:16, borderRadius:12, marginTop:16, border:'1px solid #2a2a2a'}}>
          <strong>Answer</strong>
          <div style={{marginTop:8}}>{ans}</div>
        </div>
      )}

      {pred && (
        <div style={{background:'#151515', padding:16, borderRadius:12, marginTop:16, border:'1px solid #2a2a2a'}}>
          <strong>{pred.company} ({pred.ticker})</strong>
          <div style={{marginTop:8}}>
            Verdict: <span style={{color: pred.predictionColor}}>{pred.prediction}</span>
          </div>
          <ul style={{marginTop:8}}>
            {pred.reasons?.map((r,i)=><li key={i}>{r}</li>)}
          </ul>
          <div style={{display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:8, marginTop:8}}>
            <div>ROE: {pred.metrics?.roe}</div>
            <div>P/E: {pred.metrics?.pe}</div>
            <div>D/E: {pred.metrics?.debtToEquity}</div>
            <div>Growth: {pred.metrics?.earningsGrowth}</div>
            <div>EPS: {pred.metrics?.eps}</div>
            <div>Price: {pred.metrics?.price}</div>
            <div>Intrinsic: {pred.metrics?.intrinsicValue}</div>
          </div>
        </div>
      )}
    </div>
  )
}
