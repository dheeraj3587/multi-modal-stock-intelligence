import { useState, useRef, useEffect, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send,
  Bot,
  User,
  Sparkles,
  TrendingUp,
  Loader2,
  Zap,
  RotateCcw,
  ChevronDown,
  Brain,
  FlaskConical,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { api } from '../lib/api';
import { cn } from '../lib/utils';
import { PageTransition } from '../components/shared/PageTransition';

/* ── Types ───────────────────────────────────────────────── */
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  cached?: boolean;
  model?: string;
  deep?: boolean;
}

/* ── Suggested prompts ───────────────────────────────────── */
const SUGGESTIONS = [
  { icon: TrendingUp, text: 'How is Reliance performing today?', symbol: 'RELIANCE' },
  { icon: Sparkles, text: 'Give me a quick analysis of TCS', symbol: 'TCS' },
  { icon: Zap, text: 'Which sectors are trending up?', symbol: undefined },
  { icon: TrendingUp, text: 'Should I look at HDFC Bank now?', symbol: 'HDFCBANK' },
];

/* ── Markdown rendering with react-markdown ──────────────── */
function MarkdownContent({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => <h1 className="text-xl font-bold text-text-primary mt-4 mb-2">{children}</h1>,
        h2: ({ children }) => <h2 className="text-lg font-bold text-text-primary mt-4 mb-2">{children}</h2>,
        h3: ({ children }) => <h3 className="text-base font-semibold text-text-primary mt-3 mb-1">{children}</h3>,
        h4: ({ children }) => <h4 className="text-sm font-semibold text-text-primary mt-2 mb-1">{children}</h4>,
        p: ({ children }) => <p className="text-text-secondary text-sm leading-relaxed mb-2">{children}</p>,
        strong: ({ children }) => <strong className="text-text-primary font-semibold">{children}</strong>,
        em: ({ children }) => <em className="text-text-secondary italic">{children}</em>,
        ul: ({ children }) => <ul className="ml-4 list-disc space-y-1 mb-2">{children}</ul>,
        ol: ({ children }) => <ol className="ml-4 list-decimal space-y-1 mb-2">{children}</ol>,
        li: ({ children }) => <li className="text-text-secondary text-sm leading-relaxed">{children}</li>,
        hr: () => <hr className="border-border my-3" />,
        blockquote: ({ children }) => (
          <blockquote className="border-l-2 border-accent/50 pl-3 my-2 text-text-tertiary italic text-sm">{children}</blockquote>
        ),
        a: ({ href, children }) => (
          <a href={href} target="_blank" rel="noopener noreferrer" className="text-accent hover:underline">{children}</a>
        ),
        code: ({ className, children }) => {
          const isBlock = className?.includes('language-');
          if (isBlock) {
            return (
              <pre className="bg-surface-0 border border-border rounded-lg p-3 my-2 overflow-x-auto">
                <code className="text-xs text-text-secondary font-mono">{children}</code>
              </pre>
            );
          }
          return <code className="bg-surface-0 text-accent text-xs px-1.5 py-0.5 rounded font-mono">{children}</code>;
        },
        pre: ({ children }) => <>{children}</>,
        table: ({ children }) => (
          <div className="overflow-x-auto my-2">
            <table className="min-w-full text-sm border border-border rounded-lg">{children}</table>
          </div>
        ),
        th: ({ children }) => <th className="px-3 py-1.5 text-left font-semibold text-text-primary bg-surface-0 border-b border-border">{children}</th>,
        td: ({ children }) => <td className="px-3 py-1.5 text-text-secondary border-b border-border">{children}</td>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

/* ── Main Component ──────────────────────────────────────── */
export function AIChatPage() {
  const [searchParams] = useSearchParams();
  const initialSymbol = searchParams.get('symbol') || '';

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [symbol, setSymbol] = useState(initialSymbol);
  const [loading, setLoading] = useState(false);
  const [deepMode, setDeepMode] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  // Track scroll position for "scroll to bottom" button
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      const gap = el.scrollHeight - el.scrollTop - el.clientHeight;
      setShowScrollBtn(gap > 150);
    };
    el.addEventListener('scroll', onScroll);
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  const sendMessage = async (text?: string, sym?: string) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: msg,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    // Build conversation history for context
    const history = messages.map((m) => ({ role: m.role, content: m.content }));

    try {
      const activeSymbol = sym ?? symbol ?? undefined;
      const res = await api.chat(msg, activeSymbol, history, deepMode);

      const botMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: res.reply,
        timestamp: res.timestamp,
        cached: res.cached,
        model: res.model,
        deep: res.deep,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const errMsg: Message = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
  };

  const isEmpty = messages.length === 0;

  return (
    <PageTransition>
      <div className="flex flex-col h-[calc(100vh-5rem)] max-w-4xl mx-auto">
        {/* ── Header ── */}
        <div className="flex items-center justify-between px-2 py-3 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-accent to-accent-hover flex items-center justify-center">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-text-primary leading-tight">StockSense AI</h1>
              <p className="text-xs text-text-tertiary">Ask anything about Indian stocks</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Deep mode toggle */}
            <button
              onClick={() => setDeepMode(!deepMode)}
              className={cn(
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border',
                deepMode
                  ? 'bg-accent/10 text-accent border-accent/30 shadow-sm'
                  : 'text-text-tertiary hover:text-text-primary hover:bg-surface-2 border-transparent',
              )}
              title="Deep mode gathers fundamentals, news, RAG context, and scorecard for thorough analysis"
            >
              <Brain className={cn('w-3.5 h-3.5', deepMode && 'animate-pulse')} />
              Deep
            </button>
            {/* Symbol filter */}
            <div className="flex items-center bg-surface-2 rounded-lg px-3 py-1.5 gap-2">
              <TrendingUp className="w-3.5 h-3.5 text-text-tertiary" />
              <input
                type="text"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="Symbol"
                className="bg-transparent text-sm text-text-primary placeholder:text-text-tertiary outline-none w-20 uppercase"
              />
            </div>
            {messages.length > 0 && (
              <button
                onClick={clearChat}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium text-text-tertiary hover:text-text-primary hover:bg-surface-2 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Clear
              </button>
            )}
          </div>
        </div>

        {/* ── Messages area ── */}
        <div
          ref={containerRef}
          className="flex-1 overflow-y-auto px-2 space-y-1 relative"
        >
          {/* Empty state */}
          <AnimatePresence>
            {isEmpty && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex flex-col items-center justify-center h-full gap-6"
              >
                <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent/20 to-accent/5 flex items-center justify-center">
                  <Sparkles className="w-8 h-8 text-accent" />
                </div>
                <div className="text-center">
                  <h2 className="text-xl font-bold text-text-primary mb-1">Ask StockSense AI</h2>
                  <p className="text-sm text-text-tertiary max-w-md">
                    Get instant AI-powered analysis on any NSE/BSE stock. Ask about fundamentals,
                    technicals, sentiment, news, or market trends.
                  </p>
                </div>
                {/* Suggestion chips */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 w-full max-w-lg">
                  {SUGGESTIONS.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => {
                        if (s.symbol) setSymbol(s.symbol);
                        sendMessage(s.text, s.symbol);
                      }}
                      className="flex items-center gap-3 px-4 py-3 rounded-xl border border-border hover:border-accent/40 hover:bg-accent/5 transition-all text-left group"
                    >
                      <s.icon className="w-4 h-4 text-text-tertiary group-hover:text-accent transition-colors shrink-0" />
                      <span className="text-sm text-text-secondary group-hover:text-text-primary transition-colors">{s.text}</span>
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Chat messages */}
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2 }}
              className={cn(
                'flex gap-3 py-3',
                msg.role === 'user' ? 'justify-end' : 'justify-start',
              )}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-hover flex items-center justify-center shrink-0 mt-0.5">
                  <Bot className="w-4 h-4 text-white" />
                </div>
              )}
              <div
                className={cn(
                  'max-w-[85%] rounded-2xl px-4 py-3',
                  msg.role === 'user'
                    ? 'bg-accent text-white rounded-br-md'
                    : 'bg-surface-2 rounded-bl-md',
                )}
              >
                {msg.role === 'user' ? (
                  <p className="text-sm leading-relaxed">{msg.content}</p>
                ) : (
                  <div className="space-y-0.5"><MarkdownContent content={msg.content} /></div>
                )}
                {/* Meta */}
                <div className={cn(
                  'flex items-center gap-2 mt-2 text-[10px]',
                  msg.role === 'user' ? 'text-white/50 justify-end' : 'text-text-tertiary',
                )}>
                  {msg.cached && (
                    <span className="flex items-center gap-0.5">
                      <Zap className="w-2.5 h-2.5" /> cached
                    </span>
                  )}
                  {msg.deep && (
                    <span className="flex items-center gap-0.5 text-accent">
                      <FlaskConical className="w-2.5 h-2.5" /> deep
                    </span>
                  )}
                  <span>{new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                </div>
              </div>
              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-lg bg-surface-3 flex items-center justify-center shrink-0 mt-0.5">
                  <User className="w-4 h-4 text-text-secondary" />
                </div>
              )}
            </motion.div>
          ))}

          {/* Typing indicator */}
          <AnimatePresence>
            {loading && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-3 py-3"
              >
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-hover flex items-center justify-center shrink-0">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-surface-2 rounded-2xl rounded-bl-md px-4 py-3 flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-accent animate-spin" />
                  <span className="text-sm text-text-tertiary">
                    {deepMode ? 'Deep analyzing — gathering fundamentals, news & RAG context...' : 'Thinking...'}
                  </span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          <div ref={messagesEndRef} />
        </div>

        {/* Scroll-to-bottom fab */}
        <AnimatePresence>
          {showScrollBtn && (
            <motion.button
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8 }}
              onClick={scrollToBottom}
              className="absolute bottom-28 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-surface-0 border border-border shadow-elevated flex items-center justify-center text-text-tertiary hover:text-text-primary transition-colors z-10"
            >
              <ChevronDown className="w-4 h-4" />
            </motion.button>
          )}
        </AnimatePresence>

        {/* ── Input area ── */}
        <div className="shrink-0 px-2 pb-3 pt-2">
          <div className="flex items-end gap-2 bg-surface-2 border border-border rounded-2xl px-4 py-2 focus-within:border-accent/50 transition-colors">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={deepMode ? `Deep analysis${symbol ? ` of ${symbol}` : ''}...` : symbol ? `Ask about ${symbol}...` : 'Ask about any stock...'}
              rows={1}
              className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-tertiary outline-none resize-none max-h-[120px] py-1.5 leading-relaxed"
            />
            <button
              onClick={() => sendMessage()}
              disabled={!input.trim() || loading}
              className={cn(
                'w-9 h-9 rounded-xl flex items-center justify-center transition-all shrink-0',
                input.trim() && !loading
                  ? 'bg-accent text-white hover:bg-accent-hover'
                  : 'bg-surface-3 text-text-tertiary cursor-not-allowed',
              )}
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </button>
          </div>
          <p className="text-[10px] text-text-tertiary text-center mt-1.5">
            StockSense AI can make mistakes. Not financial advice. Verify with your own research.
          </p>
        </div>
      </div>
    </PageTransition>
  );
}
