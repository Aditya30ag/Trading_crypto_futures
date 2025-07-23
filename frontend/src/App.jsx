import { useEffect, useState, useMemo } from 'react'
import './App.css'

const PAGES = ['Dashboard', 'Balance', 'Instruments', 'Logs', 'Settings']

function App() {
  const [signals, setSignals] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tradingMode, setTradingMode] = useState('Simulated')
  const [apiStatus, setApiStatus] = useState('Disconnected')
  const [search, setSearch] = useState('')
  const [strategyFilter, setStrategyFilter] = useState('All')
  const [page, setPage] = useState(1)
  const [activePage, setActivePage] = useState('Dashboard')
  const [balanceINR, setBalanceINR] = useState(null)
  const [balanceLoading, setBalanceLoading] = useState(true)
  const [balanceError, setBalanceError] = useState(null)
  const signalsPerPage = 8

  useEffect(() => {
    const fetchSignals = async () => {
      try {
        setLoading(true)
        const res = await fetch('/api/signals')
        if (!res.ok) throw new Error('Failed to fetch signals')
        const data = await res.json()
        setSignals(data)
        setApiStatus('Connected')
        setError(null)
      } catch (err) {
        setError('Failed to fetch signals')
        setApiStatus('Disconnected')
        setSignals([])
      } finally {
        setLoading(false)
      }
    }
    fetchSignals()
    const interval = setInterval(fetchSignals, 2000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    const fetchBalance = async () => {
      try {
        setBalanceLoading(true)
        const res = await fetch('/api/balance')
        if (!res.ok) throw new Error('Failed to fetch balance')
        const data = await res.json()
        setBalanceINR(data.INR)
        setBalanceError(null)
      } catch (err) {
        setBalanceError('Failed to fetch balance')
        setBalanceINR(null)
      } finally {
        setBalanceLoading(false)
      }
    }
    fetchBalance()
    const interval = setInterval(fetchBalance, 20000)
    return () => clearInterval(interval)
  }, [])
  useEffect(() => {
    if (activePage !== 'Logs') return
    let isMounted = true
    const fetchLogs = async () => {
      try {
        setLogsLoading(true)
        const res = await fetch('/api/logs')
        if (!res.ok) throw new Error('Failed to fetch logs')
        const data = await res.json()
        if (isMounted) {
          setLogs(data)
          setLogsError(null)
        }
      } catch (err) {
        if (isMounted) setLogsError('Failed to fetch logs')
      } finally {
        if (isMounted) setLogsLoading(false)
      }
    }
    fetchLogs()
    const interval = setInterval(fetchLogs, 2000)
    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [activePage])
  // Get unique strategies for filter dropdown
  const strategies = useMemo(() => {
    const set = new Set(signals.map(s => s.strategy))
    return ['All', ...Array.from(set)]
  }, [signals])

  // Filtered and searched signals
  const filteredSignals = useMemo(() => {
    let filtered = signals
    if (search.trim()) {
      filtered = filtered.filter(s => s.symbol.toLowerCase().includes(search.trim().toLowerCase()))
    }
    if (strategyFilter !== 'All') {
      filtered = filtered.filter(s => s.strategy === strategyFilter)
    }
    return filtered
  }, [signals, search, strategyFilter])

  // Pagination
  const totalPages = Math.ceil(filteredSignals.length / signalsPerPage) || 1
  const paginatedSignals = filteredSignals.slice((page - 1) * signalsPerPage, page * signalsPerPage)

  // Reset to page 1 if filter/search changes
  useEffect(() => { setPage(1) }, [search, strategyFilter])

  // Top Volatile Instruments (by ATR)
  const topVolatile = useMemo(() => {
    const latestBySymbol = {}
    for (const s of signals) {
      if (!latestBySymbol[s.symbol] || new Date(s.timestamp) > new Date(latestBySymbol[s.symbol].timestamp)) {
        latestBySymbol[s.symbol] = s
      }
    }
    const arr = Object.values(latestBySymbol).filter(s => s.indicators && s.indicators.atr)
    arr.sort((a, b) => b.indicators.atr - a.indicators.atr)
    const atrs = arr.map(s => s.indicators.atr)
    const getVolLevel = (atr) => {
      if (atrs.length === 0) return 'Unknown'
      const idx = atrs.findIndex(a => a === atr)
      if (idx < atrs.length * 0.25) return 'Very High'
      if (idx < atrs.length * 0.5) return 'High'
      if (idx < atrs.length * 0.75) return 'Medium'
      return 'Low'
    }
    return arr.slice(0, 4).map(s => ({
      symbol: s.symbol,
      entry_price: s.entry_price,
      atr: s.indicators.atr,
      volLevel: getVolLevel(s.indicators.atr)
    }))
  }, [signals])

  // Dashboard stats (profit only)
  const stats = useMemo(() => {
    const now = new Date()
    const profit30d = signals.reduce((sum, s) => {
      if (!s.timestamp) return sum
      const ts = new Date(s.timestamp)
      if ((now - ts) / (1000 * 60 * 60 * 24) <= 30) {
        return sum + (s.estimated_profit_inr || 0)
      }
      return sum
    }, 0)
    return {
      profit30d
    }
  }, [signals])

  // Sidebar navigation logic
  function renderMain() {
    if (activePage === 'Dashboard') {
      if (error) {
        // If failed to fetch signals, show logs in the center
        return (
          <div className="logs-page">
            <h2>Signal Logs</h2>
            {logsLoading && <div className="spinner"></div>}
            {logsError && <div className="error">{logsError}</div>}
            <div className="logs-list" style={{fontFamily: 'monospace', whiteSpace: 'pre-wrap', maxHeight: 500, overflowY: 'auto', background: '#181c24', padding: 16, borderRadius: 8}}>
              {logs.length === 0 && !logsLoading ? <div>No signal logs found.</div> : logs.map((line, i) => <div key={i}>{line}</div>)}
            </div>
          </div>
        )
      }
      return (
        <>
          <header className="dashboard-header">
            <h1>Trading Dashboard <span className="live-badge">Live</span></h1>
            <div className="dashboard-stats-row">
              <div className="dashboard-stat">Balance<br />
                {balanceLoading ? (
                  <span className="spinner" style={{width:24,height:24,borderWidth:4}}></span>
                ) : balanceError ? (
                  <span style={{color:'#ff6b6b'}}>{balanceError}</span>
                ) : (
                  <span>₹{balanceINR?.toLocaleString(undefined, {maximumFractionDigits: 2})}</span>
                )}
              </div>
              <div className="dashboard-stat">Active Signals<br /><span>{filteredSignals.length}</span></div>
              <div className="dashboard-stat">Profit (30d)<br /><span>₹{stats.profit30d.toLocaleString(undefined, {maximumFractionDigits: 2})}</span></div>
            </div>
          </header>
          <section className="signals-section">
            <h2>Trading Signals</h2>
            <div className="signals-toolbar">
              <input className="signals-search" placeholder="Search symbols..." value={search} onChange={e => setSearch(e.target.value)} />
              <select className="signals-strategy-filter" value={strategyFilter} onChange={e => setStrategyFilter(e.target.value)}>
                {strategies.map(strat => <option key={strat}>{strat}</option>)}
              </select>
              <button className="signals-filter-btn" onClick={() => { setSearch(''); setStrategyFilter('All'); }}>Clear</button>
            </div>
            {loading ? (
              <div className="spinner"></div>
            ) : paginatedSignals.length > 0 ? (
              <div className="signals-table-wrapper">
                <table className="signals-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Strategy</th>
                      <th>Side</th>
                      <th>Entry</th>
                      <th>Exit</th>
                      <th>Stop Loss</th>
                      <th>Profit (INR)</th>
                      <th>Timestamp</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedSignals.map((s, i) => (
                      <tr key={i} className="signals-row">
                        <td>{s.symbol}</td>
                        <td>{s.strategy}</td>
                        <td>{s.side === 'long' ? 'short' : s.side === 'short' ? 'long' : s.side}</td>
                        <td>{s.entry_price}</td>
                        <td>{s.stop_loss ?? '-'}</td>
                        <td>{s.take_profit ?? s.tp1 ?? s.exit_price ?? '-'}</td>
                        <td>{s.estimated_profit_inr ? `₹${s.estimated_profit_inr.toFixed(2)}` : '-'}</td>
                        <td>{s.timestamp ? new Date(s.timestamp).toLocaleString() : '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="signals-pagination">
                  <button disabled={page === 1} onClick={() => setPage(page - 1)}>&lt; Prev</button>
                  <span>Page {page} of {totalPages}</span>
                  <button disabled={page === totalPages} onClick={() => setPage(page + 1)}>Next &gt;</button>
                </div>
              </div>
            ) : error ? (
              <div className="error">{error}</div>
            ) : (
              <div>No signals</div>
            )}
          </section>
          <section className="volatile-section">
            <h2>Top Volatile Instruments</h2>
            <div className="volatile-instruments-row">
              {topVolatile.length === 0 ? (
                <div style={{color:'#bfc7d5'}}>No data</div>
              ) : topVolatile.map((v, i) => (
                <div className="volatile-card" key={v.symbol}>
                  {v.symbol}<br />
                  <span>₹{v.entry_price}</span><br />
                  Volatility: {v.volLevel}
                </div>
              ))}
            </div>
          </section>
        </>
      )
    }
    if (activePage === 'Logs') {
      return (
        <div className="logs-page">
          <h2>Signal Logs</h2>
          {logsLoading && <div className="spinner"></div>}
          {logsError && <div className="error">{logsError}</div>}
          <div className="logs-list" style={{fontFamily: 'monospace', whiteSpace: 'pre-wrap', maxHeight: 500, overflowY: 'auto', background: '#181c24', padding: 16, borderRadius: 8}}>
            {logs.length === 0 && !logsLoading ? <div>No signal logs found.</div> : logs.map((line, i) => <div key={i}>{line}</div>)}
          </div>
        </div>
      )
    }

    // Placeholder for other pages
    return (
      <div className="page-placeholder">
        <h2>{activePage}</h2>
        <p>Coming soon...</p>
      </div>
    )
  }

  return (
    <div className="app-layout">
      <aside className="sidebar dark-theme">
        <div className="sidebar-header">
          <div className="sidebar-logo">⚡</div>
          <div className="sidebar-title">CoinDCX Bot</div>
        </div>
        <nav className="sidebar-nav">
          {PAGES.map(pageName => (
            <a
              key={pageName}
              className={activePage === pageName ? 'active' : ''}
              onClick={() => setActivePage(pageName)}
            >
              {pageName}
            </a>
          ))}
        </nav>
        <div className="sidebar-section">
          <div className="trading-mode-toggle">
            <span>Trading Mode</span>
            <div className="toggle-row">
              <span className={tradingMode === 'Simulated' ? 'toggle-active' : ''}>Simulated</span>
              <label className="switch">
                <input type="checkbox" checked={tradingMode === 'Real'} onChange={() => setTradingMode(tradingMode === 'Simulated' ? 'Real' : 'Simulated')} />
                <span className="slider"></span>
              </label>
              <span className={tradingMode === 'Real' ? 'toggle-active' : ''}>Real</span>
            </div>
          </div>
        </div>
        <div className="sidebar-section">
          <div className={`api-status-card ${apiStatus === 'Connected' ? 'connected' : 'disconnected'}`}> 
            <div>API Status</div>
            <div className="api-status-dot" />
            <span>{apiStatus}</span>
            <div className="api-status-time">Last checked: {new Date().toLocaleString()}</div>
          </div>
        </div>
      </aside>
      <main className="dashboard dark-theme">
        {renderMain()}
      </main>
    </div>
  )
}

export default App
