interface NotFoundPageProps {
  onBack: () => void;
}

export function NotFoundPage({ onBack }: NotFoundPageProps) {
  return (
    <section className="empty-state" aria-labelledby="not-found-title">
      <p className="eyebrow">404</p>
      <h2 id="not-found-title">Page not found</h2>
      <p>The requested workspace view does not exist.</p>
      <button className="primary-button" onClick={onBack} type="button">Return to monitoring</button>
    </section>
  );
}

