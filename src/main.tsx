import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/global.css';
import 'katex/dist/katex.min.css';

const stored = localStorage.getItem('clip-theme');
const dark = stored
  ? stored === 'dark'
  : window.matchMedia('(prefers-color-scheme: dark)').matches;
document.documentElement.dataset.theme = dark ? 'dark' : 'light';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
