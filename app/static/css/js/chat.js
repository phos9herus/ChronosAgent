// static/js/chat.js

// ==========================================
// 全局状态与缓存机制
// ==========================================
const state = {
    currentRoleId: null,
    isGenerating: false,
    selectedImages: [],
    activeAiText: "",
    activeAiThoughtText: "",
    enableThink: false,
    currentSettings: {} // 缓存当前角色的完整配置，用于实时同步
};

// ==========================================
// DOM 节点引用
// ==========================================
const dom = {
    roleList: document.getElementById('role-list'),
    chatMessages: document.getElementById('chat-messages'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),
    fileInput: document.getElementById('file-input'),
    previewArea: document.getElementById('preview-area'),
    drawer: document.getElementById('settings-drawer'),
    thinkToggle: document.getElementById('think-toggle-btn'),
    setPrompt: document.getElementById('set-prompt'),
    setTemp: document.getElementById('set-temp'),
    setTopP: document.getElementById('set-top-p'),
    setTopK: document.getElementById('set-top-k'),
    setRepetition: document.getElementById('set-repetition'),
    setPresence: document.getElementById('set-presence'),
    setBudget: document.getElementById('set-budget'),
    openSettings: document.getElementById('open-settings'),
    closeSettings: document.getElementById('close-settings'),
    saveSettings: document.getElementById('save-settings')
};

let ws = null;
let currentAiBubble = null;
let currentAiThoughtNode = null;
let typingIndicatorNode = null; // 用于追踪等待动画 DOM
let reconnectAttempts = 0;

// 初始化 Markdown，默认关闭高亮以保证流式传输的极速性能
marked.setOptions({ highlight: null });

// 滚动节流锁
let scrollFrame = null;
function smoothScrollToBottom() {
    if (!scrollFrame) {
        scrollFrame = requestAnimationFrame(() => {
            dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
            scrollFrame = null;
        });
    }
}

// ==========================================
// 动画控制模块
// ==========================================
function showGlobalLoading(text = "加载中...") {
    let loader = document.getElementById('global-loader');
    if (!loader) {
        loader = document.createElement('div');
        loader.id = 'global-loader';
        loader.className = 'global-loading';
        dom.chatMessages.parentElement.appendChild(loader);
    }
    loader.innerHTML = `<span>${text}</span>`;
    loader.style.display = 'flex';
}

function hideGlobalLoading() {
    const loader = document.getElementById('global-loader');
    if (loader) loader.style.display = 'none';
}

function showTypingIndicator() {
    if (typingIndicatorNode) return;
    const row = document.createElement('div');
    row.className = 'message-row ai';
    row.id = 'typing-indicator-row';
    row.innerHTML = `
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
        </div>
    `;
    dom.chatMessages.appendChild(row);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    typingIndicatorNode = row;
}

function hideTypingIndicator() {
    if (typingIndicatorNode) {
        typingIndicatorNode.remove();
        typingIndicatorNode = null;
    }
}

// ==========================================
// 角色管理与历史记录加载
// ==========================================

async function fetchRoles() {
    try {
        const res = await fetch('/api/roles');
        const roles = await res.json();
        dom.roleList.innerHTML = roles.map(r =>
            `<div class="role-item" onclick="selectRole('${r.role_id}', '${r.role_name || r.name}', this)">
                ${r.role_name || r.name}
            </div>`
        ).join('');
    } catch (e) { console.error("无法加载角色列表:", e); }
}

async function selectRole(id, name, el) {
    state.currentRoleId = id;

    // UI 激活态切换
    document.querySelectorAll('.role-item').forEach(i => i.classList.remove('active'));
    if(el) el.classList.add('active');
    document.getElementById('current-role-title').innerText = name;

    // DOM 物理重建：强制销毁旧 DOM 节点及其隐式绑定的事件
    const oldContainer = dom.chatMessages;
    const newContainer = oldContainer.cloneNode(false);
    oldContainer.parentNode.replaceChild(newContainer, oldContainer);
    dom.chatMessages = newContainer;

    // 状态净化
    state.isGenerating = false;
    state.activeAiText = "";
    state.activeAiThoughtText = "";
    state.selectedImages = [];
    currentAiBubble = null;
    currentAiThoughtNode = null;

    dom.userInput.disabled = false;
    dom.sendBtn.disabled = false;
    dom.previewArea.innerHTML = '';

    showGlobalLoading(`正在连接 ${name} 的记忆网络...`);

    try {
        // 同步该角色的配置与状态
        const resSettings = await fetch(`/api/roles/${id}/settings`);
        const s = await resSettings.json();
        state.currentSettings = s; // 写入缓存
        updateThinkUI(!!s.enable_think);

        // 加载历史记录
        await loadChatHistory(id);
    } catch (e) {
        console.error("加载角色数据失败:", e);
        dom.chatMessages.innerHTML = `<div class="system-hint" style="color:#ff4d4f">加载失败，请重试</div>`;
    } finally {
        hideGlobalLoading();
    }
}

async function loadChatHistory(roleId) {
    try {
        const res = await fetch(`/api/roles/${roleId}/history`);

        // 强化错误提示：如果后端没加接口，这里会直接红字提示，不再静默白屏
        if (!res.ok) {
            dom.chatMessages.innerHTML = `<div class="system-hint" style="color:#ff4d4f">加载历史失败 (HTTP ${res.status})。请检查后端 history 路由是否正常。</div>`;
            return;
        }

        const history = await res.json();

        if (!history || history.length === 0) {
            dom.chatMessages.innerHTML = `<div class="system-hint">已连接到角色。这是你们的第一次对话。</div>`;
            return;
        }

        // 清空容器准备渲染
        dom.chatMessages.innerHTML = '';

        // 渲染历史消息
        history.forEach(msg => {
            if (msg.role === 'user') {
                appendUserMessage(msg.content, msg.images || []);
            } else if (msg.role !== 'system') {
                // 兼容性修复：只要不是 user，统一按 AI 角色渲染 (因为后端可能存的是"助手"等真名)
                const row = document.createElement('div');
                row.className = 'message-row ai';
                row.innerHTML = `
                    <div class="message-bubble">
                        <div class="answer-content markdown-body">${marked.parse(msg.content || "")}</div>
                    </div>
                `;
                dom.chatMessages.appendChild(row);
            }
        });

        // 插入记忆边界提示
        const boundary = document.createElement('div');
        boundary.className = 'system-hint';
        boundary.innerText = '--- 以上为历史记忆 ---';
        dom.chatMessages.appendChild(boundary);

        dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    } catch (e) {
        console.warn("历史记录加载异常:", e);
        dom.chatMessages.innerHTML = `<div class="system-hint" style="color:#ff4d4f">网络异常，无法加载历史记录。</div>`;
    }
}

// ==========================================
// 深度思考 (Thinking) 状态与 UI 实时联动
// ==========================================

function updateThinkUI(isEnable) {
    state.enableThink = isEnable;
    if (dom.thinkToggle) {
        if (isEnable) {
            dom.thinkToggle.className = 'think-btn-active';
            dom.thinkToggle.innerText = '深度思考: 开';
        } else {
            dom.thinkToggle.className = 'think-btn-inactive';
            dom.thinkToggle.innerText = '深度思考: 关';
        }
    }
}

if (dom.thinkToggle) {
    dom.thinkToggle.onclick = async () => {
        if (!state.currentRoleId) return alert("请先选择一个角色");
        if (state.isGenerating) return alert("AI正在输出中，请稍后再切换");

        const newState = !state.enableThink;
        updateThinkUI(newState);

        const payload = {
            system_prompt: state.currentSettings.system_prompt || "",
            settings: {
                ...state.currentSettings, // 继承其他参数
                enable_think: newState
            }
        };

        state.currentSettings.enable_think = newState;

        try {
            const res = await fetch(`/api/roles/${state.currentRoleId}/settings`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) throw new Error("后端保存失败");
        } catch (e) {
            console.error("同步失败:", e);
            alert("状态同步失败，已回滚");
            updateThinkUI(!newState);
            state.currentSettings.enable_think = !newState;
        }
    };
}

// ==========================================
// 全局 WebSocket 通信与渲染流水线
// ==========================================

function initGlobalWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
    }

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/chat`);

    ws.onopen = () => {
        reconnectAttempts = 0;
        console.log("WebSocket 全局持久连接已就绪");
    };

    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);

        // 核心路由拦截：如果收到的消息不属于当前正在看的角色，直接丢弃渲染
        if (data.role_id && data.role_id !== state.currentRoleId) {
            return;
        }

        // 一旦收到有效数据（非心跳/非状态初始化），隐藏打字动画
        if (data.msg_type !== "status" || data.content === "[DONE]") {
            hideTypingIndicator();
        }

        if (data.msg_type === "status" && data.content === "[DONE]") {
            finalizeAi();
        } else if (data.msg_type === "error") {
            alert(data.content);
            state.isGenerating = false;
        } else if (data.msg_type === "usage") {
            renderTokenUsage(data.content);
        } else {
            renderStream(data);
        }
    };

    ws.onclose = (e) => {
        // 断线自动重连，不再受限于角色切换
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts) + Math.random() * 1000, 30000);
        reconnectAttempts++;
        setTimeout(initGlobalWebSocket, delay);
    };
}

function renderStream(data) {
    if (!currentAiBubble) createAiRow();

    if (data.msg_type === "thought") {
        const details = currentAiBubble.parentElement.querySelector('.thought-details');
        details.style.display = "block";
        state.activeAiThoughtText += data.content;
        currentAiThoughtNode.innerHTML = marked.parse(state.activeAiThoughtText);
    }
    else if (data.msg_type === "answer") {
        state.activeAiText += data.content;
        currentAiBubble.innerHTML = marked.parse(state.activeAiText);
    }

    // 使用节流函数代替直接操作 DOM
    smoothScrollToBottom();
}

function createAiRow() {
    const row = document.createElement('div');
    row.className = 'message-row ai';
    row.innerHTML = `
        <div class="message-bubble">
            <details class="thought-details" style="display:none" open>
                <summary class="thought-summary"> 深度思考过程 (点击折叠/展开)</summary>
                <div class="thought-content markdown-body"></div>
            </details>
            <div class="answer-content markdown-body"></div>
            <div class="token-usage-bar"></div>
        </div>
    `;
    dom.chatMessages.appendChild(row);
    currentAiThoughtNode = row.querySelector('.thought-content');
    currentAiBubble = row.querySelector('.answer-content');
}

function renderTokenUsage(usageData) {
    if (!currentAiBubble) return;
    const tokenBar = currentAiBubble.parentElement.querySelector('.token-usage-bar');
    if (tokenBar) {
        tokenBar.innerHTML = `
            <span>输入: ${usageData.input}</span>
            <span>输出: ${usageData.output}</span>
            <span style="color: var(--text-primary); font-weight: 600;">总计: ${usageData.total}</span>
        `;
        tokenBar.classList.add('visible');
    }
}

function finalizeAi() {
    const details = currentAiBubble?.parentElement.querySelector('.thought-details');
    if (details && details.style.display !== "none") {
        details.removeAttribute('open');
    }

    // 收到 [DONE] 后，开启一次性代码高亮重新渲染最终气泡
    if (currentAiBubble) {
        marked.setOptions({ highlight: (code) => hljs.highlightAuto(code).value });
        currentAiBubble.innerHTML = marked.parse(state.activeAiText);
        if (currentAiThoughtNode) {
            currentAiThoughtNode.innerHTML = marked.parse(state.activeAiThoughtText);
        }
        marked.setOptions({ highlight: null }); // 渲染完立刻关掉，为下次流式做准备
    }

    state.isGenerating = false;
    currentAiBubble = null;
    currentAiThoughtNode = null;
    state.activeAiText = "";
    state.activeAiThoughtText = "";
}

// ==========================================
// 消息与图片发送逻辑
// ==========================================

function appendUserMessage(text, images) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (images && images.length > 0) {
        const imgContainer = document.createElement('div');
        imgContainer.className = 'message-images';
        images.forEach(img => {
            const el = document.createElement('img');
            el.src = img;
            imgContainer.appendChild(el);
        });
        bubble.appendChild(imgContainer);
    }

    if (text) {
        const textEl = document.createElement('div');
        textEl.innerText = text;
        bubble.appendChild(textEl);
    }

    row.appendChild(bubble);
    dom.chatMessages.appendChild(row);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
}

async function sendMessage() {
    if (!state.currentRoleId) return alert("请先选择一个角色");
    if (state.isGenerating) return;

    const text = dom.userInput.value.trim();
    if (!text && state.selectedImages.length === 0) return;

    appendUserMessage(text, state.selectedImages);
    showTypingIndicator(); // 立即触发等待动画

    const payload = {
        role_id: state.currentRoleId,
        user_input: text,
        images: state.selectedImages,
        enable_think: state.enableThink
    };

    state.isGenerating = true;
    dom.userInput.value = '';
    dom.previewArea.innerHTML = '';
    state.selectedImages = [];

    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
    } else {
        hideTypingIndicator();
        alert("WebSocket 未连接，请刷新重试");
        state.isGenerating = false;
    }
}

if (dom.sendBtn) dom.sendBtn.onclick = sendMessage;
if (dom.userInput) {
    dom.userInput.onkeydown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };
}

// ==========================================
// 图片上传与预览逻辑
// ==========================================

function handleFileSelect(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file.type.startsWith('image/')) continue;

        const reader = new FileReader();
        reader.onload = (e) => {
            const base64Data = e.target.result;
            state.selectedImages.push(base64Data);

            const wrap = document.createElement('div');
            wrap.className = 'preview-img';

            const imgEl = document.createElement('img');
            imgEl.src = base64Data;
            wrap.appendChild(imgEl);

            const removeBtn = document.createElement('button');
            removeBtn.className = 'remove-btn';
            removeBtn.innerText = 'x';
            removeBtn.onclick = () => {
                wrap.remove();
                const idx = state.selectedImages.indexOf(base64Data);
                if (idx > -1) state.selectedImages.splice(idx, 1);
            };
            wrap.appendChild(removeBtn);

            dom.previewArea.appendChild(wrap);
        };
        reader.readAsDataURL(file);
    }
    dom.fileInput.value = "";
}

if (dom.fileInput) dom.fileInput.addEventListener('change', handleFileSelect);

// ==========================================
// 角色设置抽屉与配置保存
// ==========================================

if (dom.openSettings) {
    dom.openSettings.onclick = async () => {
        if (!state.currentRoleId) return;
        try {
            const res = await fetch(`/api/roles/${state.currentRoleId}/settings`);
            const s = await res.json();

            if (dom.setPrompt) dom.setPrompt.value = s.system_prompt || "";
            if (dom.setTemp) dom.setTemp.value = s.temperature || 1.0;
            if (dom.setTopP) dom.setTopP.value = s.top_p || 0.8;
            if (dom.setTopK) dom.setTopK.value = s.top_k || 20;
            if (dom.setRepetition) dom.setRepetition.value = s.repetition_penalty || 1.1;
            if (dom.setPresence) dom.setPresence.value = s.presence_penalty || 0.0;
            if (dom.setBudget) dom.setBudget.value = s.thinking_budget || 81920;

            dom.drawer.classList.add('open');
        } catch (e) { console.error("配置加载失败", e); }
    };
}

if (dom.closeSettings) dom.closeSettings.onclick = () => dom.drawer.classList.remove('open');

if (dom.saveSettings) {
    dom.saveSettings.onclick = async () => {
        const payload = {
            system_prompt: dom.setPrompt ? dom.setPrompt.value : "",
            settings: {
                enable_think: state.enableThink,
                temperature: parseFloat(dom.setTemp.value) || 1.0,
                top_p: parseFloat(dom.setTopP.value) || 0.8,
                top_k: parseInt(dom.setTopK.value) || 20,
                repetition_penalty: parseFloat(dom.setRepetition.value) || 1.1,
                presence_penalty: parseFloat(dom.setPresence.value) || 0.0,
                thinking_budget: parseInt(dom.setBudget.value) || 81920
            }
        };

        const res = await fetch(`/api/roles/${state.currentRoleId}/settings`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (res.ok) {
            dom.drawer.classList.remove('open');
            state.currentSettings = { system_prompt: payload.system_prompt, ...payload.settings };
        } else {
            alert("保存配置失败，请检查后端日志。");
        }
    };
}

// 启动！
window.onload = () => {
    fetchRoles();
    initGlobalWebSocket();
};