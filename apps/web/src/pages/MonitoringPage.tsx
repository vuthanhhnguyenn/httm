export function MonitoringPage() {
  return (
    <section className="page-grid" aria-labelledby="monitoring-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Operator view</p>
          <h2 id="monitoring-title">Monitoring dashboard</h2>
        </div>
        <span className="status-pill status-pill-muted">Camera stopped</span>
      </div>
      <div className="metric-grid">
        <article className="metric-card">
          <span className="metric-label">Current risk</span>
          <strong className="metric-value">0</strong>
          <span className="metric-caption">No active session</span>
        </article>
        <article className="metric-card">
          <span className="metric-label">Processing</span>
          <strong className="metric-value">Idle</strong>
          <span className="metric-caption">Analyzer status pending</span>
        </article>
        <article className="metric-card">
          <span className="metric-label">Events</span>
          <strong className="metric-value">0</strong>
          <span className="metric-caption">Timeline is clear</span>
        </article>
      </div>
      <div className="camera-placeholder" role="img" aria-label="Camera preview placeholder">
        <div className="camera-placeholder-icon" aria-hidden="true">◉</div>
        <p>Camera preview will appear after a session starts.</p>
      </div>
    </section>
  );
}

