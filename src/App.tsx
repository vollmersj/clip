import { useCallback, useEffect, useState } from 'react';
import { chapters } from './chapters';

function chapterIndexFromHash(): number {
  const slug = window.location.hash.replace(/^#\/?/, '');
  const i = chapters.findIndex((c) => c.slug === slug);
  return i >= 0 ? i : 0;
}

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--accent)" />
      <rect x="6" y="6" width="8" height="8" rx="2" fill="#fff" opacity="0.95" />
      <rect x="18" y="18" width="8" height="8" rx="2" fill="#fff" opacity="0.95" />
      <rect x="18" y="6" width="8" height="8" rx="2" fill="#fff" opacity="0.35" />
      <rect x="6" y="18" width="8" height="8" rx="2" fill="#fff" opacity="0.35" />
    </svg>
  );
}

export default function App() {
  const [index, setIndex] = useState(chapterIndexFromHash);
  const [tocOpen, setTocOpen] = useState(false);
  const [theme, setTheme] = useState(
    () => document.documentElement.dataset.theme ?? 'light',
  );

  const goTo = useCallback((i: number) => {
    const clamped = Math.min(chapters.length - 1, Math.max(0, i));
    window.location.hash = `#/${chapters[clamped].slug}`;
  }, []);

  useEffect(() => {
    const onHash = () => {
      setIndex(chapterIndexFromHash());
      setTocOpen(false);
      window.scrollTo({ top: 0 });
    };
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement;
      if (/^(input|textarea|select)$/i.test(target.tagName)) return;
      if (e.key === 'ArrowRight') goTo(chapterIndexFromHash() + 1);
      if (e.key === 'ArrowLeft') goTo(chapterIndexFromHash() - 1);
      if (e.key === 'Escape') setTocOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [goTo]);

  const toggleTheme = () => {
    const next = theme === 'dark' ? 'light' : 'dark';
    setTheme(next);
    document.documentElement.dataset.theme = next;
    localStorage.setItem('clip-theme', next);
  };

  const chapter = chapters[index];
  const Component = chapter.component;

  return (
    <>
      <header className="app-header">
        <div className="app-header-inner">
          <button className="app-logo" onClick={() => goTo(0)} title="Back to start">
            <Logo />
            <span>Inside CLIP</span>
          </button>
          <div className="header-spacer" />
          <span className="header-chapter-label">
            {index + 1} / {chapters.length} · {chapter.short}
          </span>
          <button className="icon-btn" onClick={() => setTocOpen(true)}>
            Contents
          </button>
          <button
            className="icon-btn"
            onClick={toggleTheme}
            title="Toggle light/dark theme"
            aria-label="Toggle light/dark theme"
          >
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </div>
        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${((index + 1) / chapters.length) * 100}%` }}
          />
        </div>
      </header>

      {tocOpen && (
        <>
          <div className="toc-backdrop" onClick={() => setTocOpen(false)} />
          <nav className="toc-panel" aria-label="Chapters">
            <p className="toc-title">Chapters</p>
            {chapters.map((c, i) => (
              <button
                key={c.slug}
                className={`toc-item${i === index ? ' active' : ''}`}
                onClick={() => goTo(i)}
              >
                <span className="toc-num">{i + 1}</span>
                <span>{c.title}</span>
              </button>
            ))}
          </nav>
        </>
      )}

      <main className="chapter" key={chapter.slug}>
        <p className="chapter-kicker">
          Chapter {index + 1} of {chapters.length}
        </p>
        <Component />
      </main>

      <nav className="chapter-nav" aria-label="Chapter navigation">
        {index > 0 ? (
          <button className="nav-card" onClick={() => goTo(index - 1)}>
            <span className="dir">← Previous</span>
            <span className="name">{chapters[index - 1].short}</span>
          </button>
        ) : (
          <div className="nav-card empty" />
        )}
        {index < chapters.length - 1 ? (
          <button className="nav-card next" onClick={() => goTo(index + 1)}>
            <span className="dir">Next →</span>
            <span className="name">{chapters[index + 1].short}</span>
          </button>
        ) : (
          <div className="nav-card empty" />
        )}
      </nav>
    </>
  );
}
