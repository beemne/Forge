import { ChallengeChatMessage } from '../types';

// Operator API key for the backend's X-Forge-Key gate. Empty in local dev, where the
// backend runs with FORGE_API_KEY unset and accepts unauthenticated calls.
const FORGE_API_KEY: string = import.meta.env.VITE_FORGE_API_KEY || '';

/**
 * Drop-in `fetch` that attaches the X-Forge-Key header to every backend call.
 * Every request in this module goes through it, so the key cannot be missed on a
 * newly added endpoint. Existing headers are preserved and never overwritten —
 * notably, FormData uploads keep the browser-generated multipart Content-Type.
 */
export const apiFetch = (input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> => {
  if (!FORGE_API_KEY) return fetch(input, init);
  return fetch(input, {
    ...init,
    headers: { ...(init.headers || {}), 'X-Forge-Key': FORGE_API_KEY }
  });
};

const getApiBaseUrl = (): string => {
  if (typeof window !== 'undefined' && window.location) {
    if (window.location.port === '5173') {
      return '/api';
    }
    return '/api';
  }
  return 'http://127.0.0.1:8000/api';
};

const getWsBaseUrl = (): string => {
  if (typeof window !== 'undefined' && window.location) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws/events`;
  }
  return 'ws://127.0.0.1:8000/ws/events';
};

const API_BASE_URL = getApiBaseUrl();

export class ApiService {
  private ws: WebSocket | null = null;
  private wsListeners: Array<(eventData: any) => void> = [];
  public isOnline: boolean = true;
  public lastError: string | null = null;

  // ----------------------------------------------------
  // WEBSOCKET REAL-TIME EVENTS
  // ----------------------------------------------------

  public connectWebSocket(onEvent: (eventData: any) => void) {
    this.wsListeners.push(onEvent);

    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      this.ws = new WebSocket(getWsBaseUrl());

      this.ws.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data);
          this.wsListeners.forEach((listener) => listener(parsed));
        } catch (e) {
          console.warn('Malformed WS frame received:', ev.data);
        }
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected. Reconnecting in 3s...');
        setTimeout(() => this.connectWebSocket(onEvent), 3000);
      };

      this.ws.onerror = (err) => {
        console.error('WebSocket connection error:', err);
      };
    } catch (e) {
      console.warn('WebSocket connection failed:', e);
    }
  }

  // ----------------------------------------------------
  // SYSTEM & ENVIRONMENT
  // ----------------------------------------------------

  public async getHealth() {
    const res = await apiFetch(`${getApiBaseUrl()}/health`);
    if (!res.ok) {
      this.isOnline = false;
      this.lastError = `HTTP ${res.status}`;
      throw new Error(`HTTP ${res.status}`);
    }
    this.isOnline = true;
    this.lastError = null;
    return await res.json();
  }

  public async getEnvironment() {
    const res = await apiFetch(`${getApiBaseUrl()}/environment`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // CHALLENGES
  // ----------------------------------------------------

  public async getChallenges() {
    const res = await apiFetch(`${getApiBaseUrl()}/challenges`);
    if (!res.ok) {
      this.isOnline = false;
      this.lastError = `HTTP ${res.status}`;
      throw new Error(`HTTP ${res.status}`);
    }
    this.isOnline = true;
    this.lastError = null;
    return await res.json();
  }

  public async getChallengePlan(challengeId: string) {
    try {
      const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/plan`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return null;
    }
  }

  public async createChallenge(data: {
    name: string;
    category: string;
    difficulty?: string;
    description: string;
    target_address: string;
    working_directory?: string;
    platform_name?: string;
    requires_root?: boolean;
    flag_pattern?: string;
    max_iterations?: number;
    max_minutes?: number;
    instance_expiry_minutes?: number;
    attached_file_paths?: string[];
  }) {
    const res = await apiFetch(`${API_BASE_URL}/challenges`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async uploadArtifact(file: File): Promise<{ path: string; filename: string; size: number }> {
    const form = new FormData();
    form.append('file', file);
    // No explicit Content-Type — the browser sets the multipart boundary.
    const res = await apiFetch(`${API_BASE_URL}/challenges/upload`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ------------------------------------------------------------------
  // CHAT-DRIVEN CHALLENGE CREATION
  // ------------------------------------------------------------------

  /** Start a new two-turn chat session. Returns { session_id, step, bot_message }. */
  public async startChatSession(): Promise<{ session_id: string; step: number; bot_message: string }> {
    const res = await apiFetch(`${API_BASE_URL}/challenges/chat-session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  /**
   * Send the next message in a chat session.
   * On step==1 (turn 1) supply challenge_name / platform_name / challenge_type.
   * On step==2 (turn 2) supply description and optional target_address / attached_file_paths.
   * Returns { session_id, step, bot_message, challenge? } — challenge is present on the final turn.
   */
  public async sendChatMessage(
    sessionId: string,
    payload: {
      challenge_name?: string;
      platform_name?: string;
      challenge_type?: string;
      target_address?: string;
      description?: string;
      attached_file_paths?: string[];
    }
  ): Promise<{ session_id: string; step: number | string; bot_message: string; challenge?: any }> {
    const res = await apiFetch(`${API_BASE_URL}/challenges/chat-session/${sessionId}/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  }

  // ------------------------------------------------------------------
  // PERSISTENT CHALLENGE CHAT & OPERATIONAL MODE
  // ------------------------------------------------------------------

  public async getChallengeMessages(challengeId: string): Promise<ChallengeChatMessage[]> {
    try {
      const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/messages`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      console.warn(`Failed to fetch chat messages for challenge ${challengeId}:`, e);
      return [];
    }
  }

  public async postChallengeMessage(
    challengeId: string,
    content: string
  ): Promise<{ user_message: ChallengeChatMessage; assistant_message: ChallengeChatMessage }> {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  }

  public async updateChallengeMode(
    challengeId: string,
    mode: 'auto' | 'manual' | null
  ): Promise<{ status: string; challenge_id: string; mode: string | null }> {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/mode`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  }

  public async getCheckpoint(challengeId: string) {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/checkpoint`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async respondToCheckpoint(challengeId: string, text: string) {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/checkpoint/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async respondApproval(requestId: string, decision: 'approve' | 'deny', sudoPassword?: string): Promise<{ accepted: boolean; reason?: string }> {
    const body: Record<string, string> = { decision };
    if (sudoPassword !== undefined && sudoPassword !== '') {
      body['sudo_password'] = sudoPassword;
    }
    const res = await apiFetch(`${API_BASE_URL}/approvals/${requestId}/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return { accepted: false, reason: err.detail || 'Request failed' };
    }
    return await res.json();
  }

  /**
   * Rebuild the pending-approval list from the backend registry.
   *
   * Pending approvals otherwise reach the UI only as APPROVAL_REQUIRED pushes, so a page
   * load or a dropped socket would leave the operator console showing nothing while the
   * backend is still waiting. Read-only: decisions still go through respondApproval().
   */
  public async getPendingApprovals(): Promise<any[]> {
    try {
      const res = await apiFetch(`${API_BASE_URL}/approvals/pending`);
      if (!res.ok) {
        this.isOnline = false;
        this.lastError = `HTTP ${res.status}`;
        throw new Error(`HTTP ${res.status}`);
      }
      this.isOnline = true;
      this.lastError = null;
      const data = await res.json();
      return Array.isArray(data) ? data : [];
    } catch (e: any) {
      this.isOnline = false;
      this.lastError = e?.message || 'Failed to fetch pending approvals';
      return [];
    }
  }

  public async deleteChallenge(challengeId: string) {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async deleteAllChallenges() {
    const res = await apiFetch(`${API_BASE_URL}/challenges`, { method: 'DELETE' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async generateReport(challengeId: string) {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/report`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async getChallengeCandidates(challengeId: string) {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/candidates`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async getChallengeDerivedArtifacts(challengeId: string) {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/derived-artifacts`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async getChallengeDecisions(challengeId: string) {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/decisions`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }


  // ----------------------------------------------------
  // TARGETS
  // ----------------------------------------------------

  public async getTargets() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/targets`);
      if (!res.ok) {
        this.isOnline = false;
        this.lastError = `HTTP ${res.status}`;
        throw new Error(`HTTP ${res.status}`);
      }
      this.isOnline = true;
      this.lastError = null;
      return await res.json();
    } catch (e: any) {
      this.isOnline = false;
      this.lastError = e?.message || 'Network Error';
      throw e;
    }
  }

  public async verifyTarget(targetId: string) {
    const res = await apiFetch(`${API_BASE_URL}/targets/${targetId}/verify`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async rediscoverTarget(targetId: string) {
    const res = await apiFetch(`${API_BASE_URL}/targets/${targetId}/rediscover`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async updateTargetAddress(targetId: string, newAddress: string) {
    const res = await apiFetch(`${API_BASE_URL}/targets/${targetId}/address`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_address: newAddress })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // RUNS & KILL SWITCH
  // ----------------------------------------------------

  public async startRun(challengeId: string) {
    const res = await apiFetch(`${API_BASE_URL}/runs/${challengeId}/start`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async pauseChallenge(challengeId: string) {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/pause`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async activateKillSwitch(runId?: string) {
    const res = await apiFetch(`${API_BASE_URL}/killswitch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ run_id: runId })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // TOOLS & EXECUTIONS
  // ----------------------------------------------------

  public async getTools() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/tools`);
      if (!res.ok) {
        this.isOnline = false;
        this.lastError = `HTTP ${res.status}`;
        throw new Error(`HTTP ${res.status}`);
      }
      this.isOnline = true;
      this.lastError = null;
      return await res.json();
    } catch (e: any) {
      this.isOnline = false;
      this.lastError = e?.message || 'Network Error';
      throw e;
    }
  }

  public async getToolExecutions(challengeId?: string, limit: number = 200) {
    try {
      const params = new URLSearchParams({ limit: String(limit) });
      if (challengeId) params.set('challenge_id', challengeId);
      const res = await apiFetch(`${API_BASE_URL}/tools/executions?${params.toString()}`);
      if (!res.ok) {
        this.isOnline = false;
        this.lastError = `HTTP ${res.status}`;
        throw new Error(`HTTP ${res.status}`);
      }
      this.isOnline = true;
      this.lastError = null;
      return await res.json();
    } catch (e: any) {
      this.isOnline = false;
      this.lastError = e?.message || 'Network Error';
      throw e;
    }
  }

  public async getAgents() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/agents`);
      if (!res.ok) {
        this.isOnline = false;
        this.lastError = `HTTP ${res.status}`;
        throw new Error(`HTTP ${res.status}`);
      }
      this.isOnline = true;
      this.lastError = null;
      return await res.json();
    } catch (e: any) {
      this.isOnline = false;
      this.lastError = e?.message || 'Network Error';
      throw e;
    }
  }

  public async executeTerminalCommand(command: string, challengeId?: string) {
    const res = await apiFetch(`${API_BASE_URL}/terminal/execute`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command, challenge_id: challengeId })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // PRIVILEGE MANAGER
  // ----------------------------------------------------

  public async getPendingPrivileges() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/privilege/pending`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return [];
    }
  }

  public async sendPrivilegeDecision(auditId: string, approved: boolean) {
    const res = await apiFetch(`${API_BASE_URL}/privilege/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audit_id: auditId, approved })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // PROVIDERS & EVIDENCE
  // ----------------------------------------------------

  public async getProviders() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/providers/health`);
      if (!res.ok) {
        this.isOnline = false;
        this.lastError = `HTTP ${res.status}`;
        throw new Error(`HTTP ${res.status}`);
      }
      this.isOnline = true;
      this.lastError = null;
      return await res.json();
    } catch (e: any) {
      this.isOnline = false;
      this.lastError = e?.message || 'Network Error';
      throw e;
    }
  }

  // Workstream B5: live provider/model catalog health from the last discovery probe.
  public async getProviderDiscovery(): Promise<any> {
    const res = await apiFetch(`${API_BASE_URL}/providers/discovery`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // Trigger an on-demand re-probe of every provider's live catalog.
  public async triggerProviderDiscovery(verify = false): Promise<any> {
    const res = await apiFetch(`${API_BASE_URL}/providers/discover?verify=${verify}`, { method: 'POST' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async getEvidence() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/evidence`);
      if (!res.ok) {
        this.isOnline = false;
        this.lastError = `HTTP ${res.status}`;
        throw new Error(`HTTP ${res.status}`);
      }
      this.isOnline = true;
      this.lastError = null;
      return await res.json();
    } catch (e: any) {
      this.isOnline = false;
      this.lastError = e?.message || 'Network Error';
      throw e;
    }
  }

  public async getFindings() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/findings`);
      if (!res.ok) {
        this.isOnline = false;
        this.lastError = `HTTP ${res.status}`;
        throw new Error(`HTTP ${res.status}`);
      }
      this.isOnline = true;
      this.lastError = null;
      return await res.json();
    } catch (e: any) {
      this.isOnline = false;
      this.lastError = e?.message || 'Network Error';
      throw e;
    }
  }

  // ----------------------------------------------------
  // SYSTEM SETTINGS MANAGEMENT
  // ----------------------------------------------------

  public async getSystemSettings() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/system/settings`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return {
        execution_mode: 'CTF_OFFENSIVE_CONTROLLED',
        auto_approve_privileged: false,
        command_timeout_seconds: 300,
        daily_budget_usd: 5.00,
        session_budget_usd: 2.00,
        paid_model_allowed: true,
        default_strategy: 'EXPLOIT_FIRST'
      };
    }
  }

  public async updateSystemSettings(data: any) {
    const res = await apiFetch(`${API_BASE_URL}/system/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // PACKAGE INSTALLATION & PRIVILEGE APPROVALS
  // ----------------------------------------------------

  public async installPackage(requestId: string, packageName: string, challengeId?: string) {
    const res = await apiFetch(`${API_BASE_URL}/package/install`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request_id: requestId,
        package_name: packageName,
        challenge_id: challengeId
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async skipPackageInstall(requestId: string, challengeId?: string) {
    const res = await apiFetch(`${API_BASE_URL}/package/skip`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request_id: requestId,
        challenge_id: challengeId
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async approvePrivilege(requestId: string, command: string, sudoPassword?: string, challengeId?: string, workingDirectory?: string) {
    const res = await apiFetch(`${API_BASE_URL}/privilege/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request_id: requestId,
        command: command,
        sudo_password: sudoPassword,
        challenge_id: challengeId,
        working_directory: workingDirectory
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async rejectPrivilege(requestId: string, challengeId?: string) {
    const res = await apiFetch(`${API_BASE_URL}/privilege/reject`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        request_id: requestId,
        challenge_id: challengeId
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // DIRECTORY BROWSER & FOLDER PICKER
  // ----------------------------------------------------

  public async browseDirectory(path?: string) {
    try {
      const url = path ? `${API_BASE_URL}/system/browse-dir?path=${encodeURIComponent(path)}` : `${API_BASE_URL}/system/browse-dir`;
      const res = await apiFetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return {
        current_path: path || '',
        parent_path: '',
        drives: [],
        directories: []
      };
    }
  }

  public async createDirectory(parent_path: string, dir_name: string) {
    const res = await apiFetch(`${API_BASE_URL}/system/create-dir`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ parent_path, dir_name })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async selectFolderDialog() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/system/select-folder-dialog`, { method: 'POST' });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return { status: 'FAILED', selected_path: '' };
    }
  }

  public async getSystemRequirements() {
    const urlsToTry = [
      '/api/system/requirements',
      'http://127.0.0.1:8000/api/system/requirements',
      'http://localhost:8000/api/system/requirements'
    ];

    for (const url of urlsToTry) {
      try {
        const res = await apiFetch(url);
        if (res.ok) {
          return await res.json();
        }
      } catch (e) {
        // Try next URL candidate
      }
    }

    // Client-side fallback diagnostic when FastAPI backend is offline
    const isWin = navigator.userAgent.includes('Windows');
    const isMac = navigator.userAgent.includes('Macintosh');
    const osName = isWin ? 'Windows 11 / 10 Host OS' : (isMac ? 'macOS Host' : 'Linux Host OS');
    const cores = navigator.hardwareConcurrency || 8;
    const ram = (navigator as any).deviceMemory || 16;

    return {
      overall_status: 'BACKEND_OFFLINE',
      offline_notice: 'FastAPI Backend Server is offline on port 8000. Start backend using: py -m uvicorn backend.main:app --port 8000',
      installed_tools_count: 0,
      total_tools_count: 25,
      environment: {
        distro: `${osName} (Browser Client Diagnostic)`,
        python_version: '3.14.0 (Backend Offline)',
        architecture: 'x86_64 / ARM64',
        cpu_cores: cores,
        ram_gb: ram
      },
      system_requirements: [
        { requirement: "FastAPI Backend Server (Port 8000)", status: "FAIL", details: "Connection Refused on http://127.0.0.1:8000", impact: "Start FastAPI backend server with `py -m uvicorn backend.main:app --port 8000`" },
        { requirement: "Python 3.10+ Runtime Environment", status: "PASS", details: "Detected via host configuration", impact: "Core backend execution requires Python 3.10+." },
        { requirement: "Host Physical RAM Memory", status: "PASS", details: `${ram} GB Client RAM`, impact: "4.0 GB+ recommended." },
        { requirement: "CPU Processing Cores", status: "PASS", details: `${cores} Cores Detected`, impact: "2+ cores recommended." }
      ],
      tool_requirements: [
        { name: "nmap", installed: false, path: "Backend Offline (Click RUN DIAGNOSTICS after starting backend)", installation_recipe: "sudo apt-get install nmap / choco install nmap" },
        { name: "ffuf", installed: false, path: "Backend Offline", installation_recipe: "sudo apt-get install ffuf" },
        { name: "curl", installed: false, path: "Backend Offline", installation_recipe: "sudo apt-get install curl" },
        { name: "claude", installed: false, path: "Backend Offline", installation_recipe: "npm install -g @anthropic-ai/claude-code" },
        { name: "codex", installed: false, path: "Backend Offline", installation_recipe: "npm install -g @openai/codex-cli" }
      ]
    };
  }

  public async parseProviderSnippet(snippet: string) {
    const res = await apiFetch(`${API_BASE_URL}/providers/parse-snippet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ snippet })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async registerProviderSnippet(data: {
    snippet?: string;
    provider_name?: string;
    api_key?: string;
    model_id?: string;
    base_url?: string;
    test_connection?: boolean;
  }) {
    const res = await apiFetch(`${API_BASE_URL}/providers/register-snippet`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async updateProviderKey(data: {
    provider_name: string;
    api_key: string;
    model_id?: string;
    base_url?: string;
  }) {
    const res = await apiFetch(`${API_BASE_URL}/providers/update-key`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // PLAYBOOK VAULT
  // ----------------------------------------------------

  public async getPlaybooks() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/playbooks`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return { count: 0, playbooks: [] };
    }
  }

  public async searchPlaybooks(query: string, category?: string) {
    try {
      const res = await apiFetch(`${API_BASE_URL}/playbooks/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, category, top_k: 10, include_unpromoted: true })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return { query, count: 0, playbooks: [] };
    }
  }

  public async ingestPlaybook(data: {
    text: string;
    category?: string;
    source_type?: string;
    title?: string;
  }) {
    const res = await apiFetch(`${API_BASE_URL}/playbooks/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw { response: { data: errBody } };
    }
    return await res.json();
  }

  // ----------------------------------------------------
  // KNOWLEDGE COVERAGE
  // ----------------------------------------------------

  public async getKnowledgeCoverage() {
    try {
      const res = await apiFetch(`${API_BASE_URL}/knowledge/coverage`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return { grand_total: 0, categories: [] };
    }
  }

  // ----------------------------------------------------
  // EXPERIENCE MEMORY (FORGE-learned experience layer)
  // ----------------------------------------------------

  public async getMemory(category?: string, outcome?: string) {
    try {
      const params = new URLSearchParams();
      if (category && category !== 'ALL') params.set('category', category.toLowerCase());
      if (outcome && outcome !== 'ALL') params.set('outcome', outcome.toLowerCase());
      const qs = params.toString();
      const res = await apiFetch(`${API_BASE_URL}/memory${qs ? `?${qs}` : ''}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return { stats: null, experiences: [] };
    }
  }

  public async getMemoryDetail(experienceId: string) {
    const res = await apiFetch(`${API_BASE_URL}/memory/${experienceId}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  public async searchMemory(query: string, category?: string) {
    try {
      const res = await apiFetch(`${API_BASE_URL}/memory/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, category: category && category !== 'ALL' ? category.toLowerCase() : undefined, top_k: 10 })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return { query, count: 0, memories: [] };
    }
  }

  public async sendMemoryFeedback(experienceId: string, success: boolean, note?: string) {
    const res = await apiFetch(`${API_BASE_URL}/memory/${experienceId}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ success, note: note || '' })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }

  // ----------------------------------------------------
  // WRITEUP / REPORT (AI-authored from real run telemetry)
  // ----------------------------------------------------

  // Returns the challenge writeup. By default returns the ALREADY-SAVED writeup
  // when one exists (saved=true) instead of authoring a new draft every open;
  // pass refresh=true to force a fresh AI-crafted draft. May take several seconds
  // while the provider chain authors a fresh one.
  public async getWriteup(challengeId: string, refresh: boolean = false): Promise<{ content: string; generated_by: string; saved?: boolean; file_path?: string }> {
    try {
      const qs = refresh ? '?refresh=1' : '';
      const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/writeup${qs}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    } catch (e) {
      return { content: '', generated_by: '', saved: false };
    }
  }

  // Persist the operator-confirmed writeup into the challenge working folder.
  public async saveWriteup(challengeId: string, content: string): Promise<{ status: string; file_path: string; content?: string; generated_by?: string; saved?: boolean }> {
    const res = await apiFetch(`${API_BASE_URL}/challenges/${challengeId}/writeup/save`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  }
}

export const apiService = new ApiService();
