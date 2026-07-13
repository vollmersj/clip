import { useMemo } from 'react';
import katex from 'katex';

interface FormulaProps {
  tex: string;
  display?: boolean;
  className?: string;
}

/** Renders a LaTeX formula with KaTeX (bundled locally, no CDN). */
export function Formula({ tex, display = false, className }: FormulaProps) {
  const html = useMemo(
    () =>
      katex.renderToString(tex, {
        displayMode: display,
        throwOnError: false,
        strict: false,
      }),
    [tex, display],
  );
  const cls = [display ? 'formula-display' : 'formula-inline', className]
    .filter(Boolean)
    .join(' ');
  return <span className={cls} dangerouslySetInnerHTML={{ __html: html }} />;
}
