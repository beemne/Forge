/**
 * NewChallengeChat — Two-turn conversational challenge creation for FORGE.
 *
 * Turn 1: Free-text collection of Challenge Name, Platform/Event Name, and Challenge Type.
 * Turn 2: Optional Target Address, optional File Attachment(s), and mandatory Goal Description.
 * On commit: Backend commits challenge row, syncs backend data, and routes the operator
 * directly into the Challenge Workspace / pipeline investigation view.
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import {
  MessageSquare,
  X,
  ArrowLeft,
  Loader2,
  AlertTriangle,
  Send,
  Bot,
  User,
  RefreshCw
} from 'lucide-react';
import { apiService } from '../../services/api';
import { Challenge, NavTab } from '../../types';
import { soundEngine } from '../../utils/soundEngine';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface ChatMessage {
  id: string;
  role: 'bot' | 'user';
  text: string;
  timestamp: Date;
  stepState?: 1 | 2 | 'committed';
  meta?: {
    name?: string;
    platform?: string;
    type?: string;
    target?: string;
    description?: string;
    files?: { name: string; size: number }[];
  };
}

interface NewChallengeChatProps {
  onOpenWorkspace: (challenge: Challenge) => void;
  onRefreshBackendData?: () => Promise<void> | void;
  setActiveTab?: (tab: NavTab) => void;
  onClose?: () => void;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mkId() {
  return Math.random().toString(36).slice(2, 10);
}

/** Minimal markdown formatter for bot instructions: **bold** and `code` tags */
function renderMarkdown(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={i} className="text-cyber-cyan font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    if (part.startsWith('`') && part.endsWith('`')) {
      return (
        <code key={i} className="bg-obsidian-900 text-cyber-amber px-1.5 py-0.5 rounded text-[11px] border border-amber-500/20">
          {part.slice(1, -1)}
        </code>
      );
    }
    return part.split('\n').map((line, j, arr) => (
      <React.Fragment key={`${i}-${j}`}>
        {line}
        {j < arr.length - 1 && <br />}
      </React.Fragment>
    ));
  });
}

// ---------------------------------------------------------------------------
// Main Component: NewChallengeChat
// ---------------------------------------------------------------------------

export const NewChallengeChat: React.FC<NewChallengeChatProps> = ({
  onOpenWorkspace,
  onRefreshBackendData,
  setActiveTab,
  onClose
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentStep, setCurrentStep] = useState<number | string>(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [redirecting, setRedirecting] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  // Initialize chat session on mount
  const initSession = useCallback(async () => {
    setLoading(true);
    setError(null);
    setRedirecting(false);
    try {
      const resp = await apiService.startChatSession();
      setSessionId(resp.session_id);
      setCurrentStep(resp.step);
      setMessages([
        {
          id: mkId(),
          role: 'bot',
          text: resp.bot_message,
          timestamp: new Date(),
          stepState: 1
        }
      ]);
    } catch (e: any) {
      setError(e?.message || 'Failed to establish challenge session with backend.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    initSession();
  }, [initSession]);

  const addUserMessage = (text: string, meta?: ChatMessage['meta']) => {
    setMessages((prev) => [
      ...prev,
      {
        id: mkId(),
        role: 'user',
        text,
        timestamp: new Date(),
        meta
      }
    ]);
  };

  const addBotMessage = (text: string, stepState?: 1 | 2 | 'committed') => {
    setMessages((prev) => [
      ...prev,
      {
        id: mkId(),
        role: 'bot',
        text,
        timestamp: new Date(),
        stepState
      }
    ]);
  };

  // Pure-chat send (Workstream F): one free-text message per turn. The backend slot-fills
  // the next missing field and drives the conversation (asks only for what's missing),
  // creating the challenge once everything is gathered.
  const [input, setInput] = useState('');
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || !sessionId || loading || redirecting) return;
    addUserMessage(text);
    setInput('');
    setLoading(true);
    setError(null);
    try {
      const resp = await apiService.sendChatMessage(sessionId, { message: text });
      setCurrentStep(resp.step);
      const stepState = resp.step === 'committed' ? 'committed' : (resp.step === 2 ? 2 : 1);
      addBotMessage(resp.bot_message, stepState as 1 | 2 | 'committed');

      if (resp.step === 'committed' && resp.challenge) {
        soundEngine.playSuccess();
        setRedirecting(true);
        const raw = resp.challenge;
        const formattedChallenge: Challenge = {
          id: raw.id,
          name: raw.name,
          category: raw.category,
          difficulty: raw.difficulty || 'MEDIUM',
          target: raw.target_address || (raw.targets && raw.targets[0]?.current_address) || '127.0.0.1',
          status: raw.status || 'RUNNING',
          progress: raw.progress || 0,
          lastActivity: 'Just now',
          flagStatus: raw.flag_status || 'UNFOUND',
          flag: raw.flag,
          description: raw.description || '',
          workingDirectory: raw.working_directory,
          platformName: raw.platform_name,
          createdAt: raw.created_at,
          created_at: raw.created_at,
          startedAt: raw.started_at,
          started_at: raw.started_at,
          missionPlan: raw.mission_plan,
          mission_plan: raw.mission_plan
        };
        if (onRefreshBackendData) await onRefreshBackendData();
        setTimeout(() => onOpenWorkspace(formattedChallenge), 800);
      }
    } catch (e: any) {
      setError(e?.message || 'Failed to send message.');
    } finally {
      setLoading(false);
    }
  }, [input, sessionId, loading, redirecting, onRefreshBackendData, onOpenWorkspace]);

  return (
    <div className="flex flex-col h-[calc(100vh-6.5rem)] font-mono text-slate-100 select-text max-w-5xl mx-auto w-full">
      {/* Header bar matching Forge standards */}
      <div className="flex items-center justify-between p-4 glass-panel rounded-xl border border-cyan-500/20 shrink-0 mb-4 bg-obsidian-950/80 shadow-[0_0_20px_rgba(0,0,0,0.5)]">
        <div className="flex items-center space-x-3.5">
          <div className="w-10 h-10 rounded-lg bg-obsidian-900 border border-cyber-cyan/50 flex items-center justify-center shadow-[0_0_15px_rgba(0,240,255,0.25)]">
            <MessageSquare className="w-5 h-5 text-cyber-cyan" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="text-sm font-display font-bold tracking-wider text-slate-100 uppercase neon-text-cyan">
                Conversational Challenge Ingestion
              </h1>
              <span className="px-2 py-0.5 rounded text-[9px] font-bold bg-cyan-950/80 border border-cyber-cyan/40 text-cyber-cyan uppercase">
                Interactive Assistant
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              Two-turn agent intake flow — collect target parameters, objectives, and auto-dispatch
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => {
              soundEngine.playClick();
              initSession();
            }}
            title="Reset Chat Session"
            className="p-2 rounded-lg border border-slate-800 hover:border-cyber-cyan/40 bg-obsidian-900 text-slate-400 hover:text-cyber-cyan transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            id="chat-back-btn"
            onClick={() => {
              soundEngine.playClick();
              if (onClose) {
                onClose();
              } else if (setActiveTab) {
                setActiveTab('challenges');
              }
            }}
            className="flex items-center space-x-2 text-xs text-slate-300 hover:text-cyber-cyan px-3.5 py-2 border border-slate-800 hover:border-cyber-cyan/40 rounded-lg bg-obsidian-900 transition-colors uppercase tracking-wider font-bold"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Challenges List</span>
          </button>
        </div>
      </div>

      {/* Main Conversation Stream */}
      <div className="flex-1 overflow-y-auto space-y-4 pr-1 pb-4 cyber-scrollbar">
        {/* Connection Loader */}
        {loading && messages.length === 0 && (
          <div className="flex flex-col items-center justify-center p-12 glass-panel rounded-xl border border-cyan-500/20 text-slate-400 space-y-3">
            <Loader2 className="w-8 h-8 animate-spin text-cyber-cyan" />
            <span className="text-xs tracking-wider uppercase font-bold text-cyber-cyan">
              Initializing AI Challenge Ingestion Session…
            </span>
          </div>
        )}

        {/* Global Error Banner */}
        {error && (
          <div className="flex items-start space-x-3 bg-rose-950/40 border border-cyber-rose/60 rounded-xl p-3.5 text-xs text-rose-300 shadow-[0_0_15px_rgba(255,0,85,0.2)]">
            <AlertTriangle className="w-4 h-4 shrink-0 text-cyber-rose mt-0.5" />
            <div className="flex-1">
              <span className="font-bold uppercase tracking-wider block text-cyber-rose">Ingestion Error</span>
              <span>{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-rose-400 hover:text-white"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        {/* Messages List */}
        {messages.map((msg, idx) => {
          const isBot = msg.role === 'bot';
          const isLastMsg = idx === messages.length - 1;

          return (
            <div
              key={msg.id}
              className={`flex items-start ${isBot ? 'justify-start' : 'justify-end'} space-x-3`}
            >
              {isBot && (
                <div className="w-8 h-8 rounded-lg bg-obsidian-900 border border-cyber-cyan/40 flex items-center justify-center shrink-0 shadow-[0_0_12px_rgba(0,240,255,0.2)] mt-0.5">
                  <Bot className="w-4 h-4 text-cyber-cyan" />
                </div>
              )}

              <div
                className={`max-w-[88%] md:max-w-[78%] rounded-xl p-4 transition-all ${
                  isBot
                    ? 'bg-obsidian-900/90 border border-cyan-500/30 text-slate-200 shadow-[0_0_20px_rgba(0,0,0,0.4)]'
                    : 'bg-gradient-to-br from-cyan-950/60 to-obsidian-900 border border-cyber-cyan/40 text-slate-100 shadow-[0_0_15px_rgba(0,240,255,0.15)]'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5 pb-1 border-b border-white/5">
                  <div className="flex items-center space-x-2">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-cyber-cyan">
                      {isBot ? 'FORGE Ingestion Assistant' : 'Operator'}
                    </span>
                    {msg.stepState === 1 && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-cyan-950 border border-cyber-cyan/30 text-cyber-cyan">
                        Step 1 of 2
                      </span>
                    )}
                    {msg.stepState === 2 && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-emerald-950 border border-cyber-emerald/30 text-cyber-emerald">
                        Step 2 of 2
                      </span>
                    )}
                    {msg.stepState === 'committed' && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-emerald-950 border border-cyber-emerald/50 text-cyber-emerald font-bold">
                        Committed
                      </span>
                    )}
                  </div>
                  <span className="text-[9px] text-slate-500 font-mono">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </span>
                </div>

                <div className="text-xs leading-relaxed whitespace-pre-wrap font-sans">
                  {renderMarkdown(msg.text)}
                </div>

                {/* Redirecting Banner */}
                {isBot && isLastMsg && (currentStep === 'committed' || redirecting) && (
                  <div className="mt-3 pt-3 border-t border-emerald-500/30 flex items-center space-x-2.5 text-xs text-cyber-emerald bg-emerald-950/30 p-2.5 rounded-lg border border-emerald-500/20">
                    <Loader2 className="w-4 h-4 animate-spin shrink-0" />
                    <span className="font-bold uppercase tracking-wider">
                      Challenge created! Routing to Investigation Workspace…
                    </span>
                  </div>
                )}
              </div>

              {!isBot && (
                <div className="w-8 h-8 rounded-lg bg-cyan-950 border border-cyber-cyan/60 flex items-center justify-center shrink-0 shadow-[0_0_12px_rgba(0,240,255,0.3)] mt-0.5">
                  <User className="w-4 h-4 text-cyber-cyan" />
                </div>
              )}
            </div>
          );
        })}

        {/* Processing Indicator */}
        {loading && messages.length > 0 && (
          <div className="flex items-center space-x-3 text-xs text-slate-400 pl-11 py-2">
            <div className="w-6 h-6 rounded-md bg-obsidian-900 border border-cyber-cyan/40 flex items-center justify-center">
              <Loader2 className="w-3.5 h-3.5 animate-spin text-cyber-cyan" />
            </div>
            <span className="text-cyber-cyan font-bold uppercase tracking-wider text-[11px] animate-pulse">
              FORGE is evaluating inputs & compiling challenge pipeline…
            </span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Pure-chat input bar (Workstream F): one free-text reply per turn; the backend
          asks only for what's still missing and creates the challenge when ready. */}
      <div className="shrink-0 mt-3 flex items-end gap-2 glass-panel rounded-xl border border-cyan-500/20 p-3 bg-obsidian-950/80">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
          }}
          rows={1}
          disabled={loading || redirecting || currentStep === 'committed' || !sessionId}
          placeholder={
            currentStep === 'committed'
              ? 'Challenge created — routing to workspace…'
              : 'Type your reply… (the challenge name, category, difficulty, or goal)'
          }
          className="flex-1 resize-none bg-obsidian-900 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:outline-none focus:border-cyber-cyan focus:ring-1 focus:ring-cyber-cyan/40 font-sans max-h-40"
        />
        <button
          onClick={handleSend}
          disabled={!input.trim() || loading || redirecting || currentStep === 'committed' || !sessionId}
          className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-cyber-cyan/15 border border-cyber-cyan/50 text-cyber-cyan text-xs font-bold uppercase tracking-wider hover:bg-cyber-cyan/25 transition-colors disabled:opacity-40 shrink-0"
        >
          <Send className="w-3.5 h-3.5" /> Send
        </button>
      </div>
    </div>
  );
};
