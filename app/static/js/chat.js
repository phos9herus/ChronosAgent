// static/js/chat.js

// ==========================================
// 全局状态树
// ==========================================
const state = {
    userProfile: { display_name: "User", avatar_mode: "circle", avatar_circle: "", avatar_bg: "" },
    roles: [], // 缓存所有的角色列表
    currentRoleId: null,
    currentRoleMeta: {}, // 当前角色的 meta
    currentConversationId: null,
    conversations: [],
    isDeleteRoleMode: false,
    pendingDeleteRoleId: null,
    isGenerating: false,
    selectedImages: [],
    activeAiText: "",
    activeAiThoughtText: "",
    enableThink: false,
    enableSearch: false,
    currentModel: "qwen3.5-plus",
    depthRecallMode: "off", // "off" | "normal" | "enhanced"
    models: {}, // 从后端获取的模型详细信息
    pendingCitations: [] // 待发送的知识库引用元数据
};

const dom = {
    roleList: document.getElementById('role-list'),
    chatMessages: document.getElementById('chat-messages'),
    userInput: document.getElementById('user-input'),
    sendBtn: document.getElementById('send-btn'),
    fileInput: document.getElementById('file-input'),
    previewArea: document.getElementById('preview-area'),
    rightDrawer: document.getElementById('settings-drawer'),
    leftDrawer: document.getElementById('user-drawer'),
    thinkToggle: document.getElementById('think-toggle-btn'),
    searchToggle: document.getElementById('search-toggle-btn'),
    depthRecallBtn: document.getElementById('depth-recall-btn'),
    modelSelect: document.getElementById('model-select'),
    modelInfoPopup: document.getElementById('model-info-popup'),
    modelInfoTitle: document.getElementById('model-info-title'),
    modelInfoContent: document.getElementById('model-info-content'),
    connectionStatusLight: document.getElementById('connection-status-light'),
    conversationMenuContainer: document.getElementById('conversation-menu-container'),
    conversationMenuToggle: document.getElementById('conversation-menu-toggle'),
    conversationCardsContainer: document.getElementById('conversation-cards-container'),
    currentConversationTitle: document.getElementById('current-conversation-title'),
    openConversationSettings: document.getElementById('open-conversation-settings'),
    conversationSettingsModal: document.getElementById('conversation-settings-modal'),
    setConversationName: document.getElementById('set-conversation-name'),
    deleteConversationBtn: document.getElementById('delete-conversation-btn'),
    btnDeleteRoleMode: document.getElementById('btn-delete-role-mode'),
    deleteModeHint: document.getElementById('delete-mode-hint'),
    deleteRoleModal1: document.getElementById('delete-role-modal-1'),
    deleteRoleModal2: document.getElementById('delete-role-modal-2'),
    deleteConversationModal: document.getElementById('delete-conversation-modal'),
    inputBubble: document.getElementById('input-bubble'),
    bubbleTextarea: document.getElementById('bubble-textarea'),
    inputWrapper: document.getElementById('input-wrapper')
};

let isBubbleExpanded = false;

let ws = null;
let currentAiBubble = null;
let currentAiThoughtNode = null;
let typingIndicatorNode = null;
let reconnectAttempts = 0;
let reconnectTimeout = null;
const MIN_RECONNECT_DELAY = 2000;
const MAX_RECONNECT_DELAY = 30000;

marked.setOptions({ highlight: null });

function smoothScrollToBottom() {
    requestAnimationFrame(() => { dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight; });
}

function updateConnectionStatus(isConnected) {
    const light = dom.connectionStatusLight;
    if (!light) return;
    light.classList.remove('connected', 'disconnected');
    if (isConnected) {
        light.classList.add('connected');
    } else {
        light.classList.add('disconnected');
    }
}

let summarizingTextElement = null;

function updateSummarizingStatus(isSummarizing) {
    const light = dom.connectionStatusLight;
    if (!light) return;
    
    if (isSummarizing) {
        light.classList.add('summarizing');
        
        if (!summarizingTextElement) {
            summarizingTextElement = document.createElement('span');
            summarizingTextElement.className = 'summarizing-text';
            summarizingTextElement.textContent = '总结上下文中...';
            light.parentNode.appendChild(summarizingTextElement);
        }
    } else {
        light.classList.remove('summarizing');
        
        if (summarizingTextElement) {
            summarizingTextElement.remove();
            summarizingTextElement = null;
        }
    }
}

function calculateReconnectDelay() {
    const delay = Math.min(
        MIN_RECONNECT_DELAY * Math.pow(2, reconnectAttempts),
        MAX_RECONNECT_DELAY
    );
    return delay;
}

function stopReconnect() {
    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }
}

// ==========================================
// 头像渲染引擎
// ==========================================
function getFallbackAvatar(name) {
    const char = name ? name.charAt(0).toUpperCase() : '?';
    const bgColors = ['#0284c7', '#be185d', '#16a34a', '#ca8a04', '#4f46e5', '#ea580c'];
    const idx = char.charCodeAt(0) % bgColors.length;
    return `<div class="text-avatar" style="background-color: ${bgColors[idx]}; color: white;">${char}</div>`;
}

// 通用渲染器（遵守渐变模式侧边栏隐藏规则）
function renderAvatarDOM(mode, circlePath, bgPath, fallbackName) {
    if (mode === 'gradient' && bgPath) return '';
    if (circlePath) return `<img src="${circlePath}" alt="avatar">`;
    return getFallbackAvatar(fallbackName);
}

// === 新增：抽屉与个人中心专用的实体渲染器（无视 gradient 的隐藏规则）===
function renderPreviewDOM(mode, circlePath, bgPath, fallbackName) {
    const path = mode === 'gradient' ? bgPath : circlePath;
    if (path) return `<img src="${path}" alt="preview" style="width:100%; height:100%; object-fit:cover; border-radius:inherit;">`;
    return getFallbackAvatar(fallbackName);
}

// 给左侧边栏渲染卡片样式 (混合渐变逻辑)
function applyGradientCardStyle(el, mode, bgPath) {
    if (mode === 'gradient' && bgPath) {
        el.classList.add('gradient-bg');
        el.style.backgroundImage = `linear-gradient(to right, rgba(0,0,0,0.4) 0%, var(--bg-l1) 85%), url(${bgPath})`;
    } else {
        el.classList.remove('gradient-bg');
        el.style.backgroundImage = 'none';
    }
}

// ==========================================
// 数据加载与列表渲染
// ==========================================
async function fetchUserProfile() {
    try {
        const res = await fetch('/api/user');
        state.userProfile = await res.json();
        
        // 如果用户有首选模型，更新当前模型（需确保模型已加载）
        if (state.userProfile.preferred_model && state.models && state.models[state.userProfile.preferred_model]) {
            state.currentModel = state.userProfile.preferred_model;
        }
        
        renderUserSidebar();
    } catch (e) { console.error("加载用户配置失败:", e); }
}

function renderUserSidebar() {
    const userBtn = document.getElementById('btn-personal-center');
    const avatarDom = document.getElementById('sidebar-user-avatar');
    document.getElementById('sidebar-user-name').innerText = state.userProfile.display_name;

    avatarDom.innerHTML = renderAvatarDOM(state.userProfile.avatar_mode, state.userProfile.avatar_circle, state.userProfile.avatar_bg, state.userProfile.display_name);
    avatarDom.style.display = (state.userProfile.avatar_mode === 'gradient' && state.userProfile.avatar_bg) ? 'none' : 'flex';
    applyGradientCardStyle(userBtn, state.userProfile.avatar_mode, state.userProfile.avatar_bg);
}

async function fetchRoles() {
    try {
        const res = await fetch('/api/roles');
        state.roles = await res.json();
        renderRoleList();
    } catch (e) { console.error("无法加载角色列表:", e); }
}

function renderRoleList() {
    dom.roleList.innerHTML = state.roles.map(r => {
        const showDom = !(r.avatar_mode === 'gradient' && r.avatar_bg);
        const domStr = showDom ? `<div class="sidebar-avatar">${renderAvatarDOM(r.avatar_mode, r.avatar_circle, r.avatar_bg, r.display_name)}</div>` : '';
        const bgStyle = (r.avatar_mode === 'gradient' && r.avatar_bg) ? `style="background-image: linear-gradient(to right, rgba(0,0,0,0.4) 0%, var(--bg-l1) 85%), url('${r.avatar_bg}');"` : '';
        const gradientClass = (r.avatar_mode === 'gradient' && r.avatar_bg) ? 'gradient-bg' : '';
        const activeClass = state.currentRoleId === r.role_id ? 'active' : '';

        return `<div class="role-item flex-item ${gradientClass} ${activeClass}" ${bgStyle} onclick="selectRole('${r.role_id}')">
            ${domStr}
            <span class="item-name">${r.display_name}</span>
        </div>`;
    }).join('');
}

// ==========================================
// 对话菜单功能
// ==========================================
function toggleConversationMenu() {
    if (dom.conversationMenuContainer) {
        dom.conversationMenuContainer.classList.toggle('collapsed');
        if (!dom.conversationMenuContainer.classList.contains('collapsed') && state.currentRoleId) {
            fetchConversations(state.currentRoleId);
        }
    }
}

// ==========================================
// 手风琴式侧边栏功能
// ==========================================
function toggleAccordion(sectionId) {
    const allSections = document.querySelectorAll('.accordion-section');

    allSections.forEach(section => {
        if (section.id === sectionId) {
            const isActive = section.classList.contains('active');
            if (isActive) {
                section.classList.remove('active');
                const arrow = section.querySelector('.accordion-arrow');
                if (arrow) arrow.textContent = '▶';
            } else {
                section.classList.add('active');
                const arrow = section.querySelector('.accordion-arrow');
                if (arrow) arrow.textContent = '▼';
                if (sectionId === 'section-kb') {
                    loadKnowledgeBases();
                }
            }
        } else {
            section.classList.remove('active');
            const arrow = section.querySelector('.accordion-arrow');
            if (arrow) arrow.textContent = '▶';
        }
    });
}

function renderConversationCards() {
    if (!dom.conversationCardsContainer) return;
    
    let html = state.conversations.map(conv => {
        const activeClass = state.currentConversationId === conv.conversation_id ? 'active' : '';
        return `
            <div class="conversation-card ${activeClass}" onclick="selectConversation('${conv.conversation_id}')">
                <div class="conversation-title">${conv.name || '新对话'}</div>
                <div class="conversation-time">${new Date(conv.last_updated).toLocaleString()}</div>
            </div>
        `;
    }).join('');
    
    html += `
        <div class="conversation-card create-conversation-card" onclick="createNewConversation()">
            <span class="conversation-plus">+</span>
            <span class="conversation-name">新建对话</span>
        </div>
    `;
    
    dom.conversationCardsContainer.innerHTML = html;
}

async function fetchConversations(roleId) {
    try {
        const res = await fetch(`/api/roles/${roleId}/conversations`);
        if (res.ok) {
            const data = await res.json();
            state.conversations = data.conversations || [];
            renderConversationCards();
        }
    } catch (e) {
        console.error('加载对话列表失败:', e);
    }
}

async function selectConversation(convId) {
    state.currentConversationId = convId;
    renderConversationCards();
    
    const conv = state.conversations.find(c => c.conversation_id === convId);
    if (dom.currentConversationTitle && conv) {
        dom.currentConversationTitle.innerText = conv.name || '新对话';
    }
    
    const oldContainer = dom.chatMessages;
    const newContainer = oldContainer.cloneNode(false);
    oldContainer.parentNode.replaceChild(newContainer, oldContainer);
    dom.chatMessages = newContainer;
    
    state.isGenerating = false;
    currentAiBubble = null;
    currentAiThoughtNode = null;
    resetBubbleState();
    
    await loadChatHistory(state.currentRoleId, convId);
    
    if (dom.conversationMenuContainer) {
        dom.conversationMenuContainer.classList.remove('open');
    }
}

async function createNewConversation() {
    if (!state.currentRoleId) return;
    
    try {
        const res = await fetch(`/api/roles/${state.currentRoleId}/conversations`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name: '新对话' })
        });
        
        if (res.ok) {
            const data = await res.json();
            await fetchConversations(state.currentRoleId);
            await selectConversation(data.conversation_id);
        }
    } catch (e) {
        console.error('创建对话失败:', e);
    }
}

// ==========================================
// 对话设置功能
// ==========================================
async function saveConversationSettings() {
    if (!state.currentConversationId || !state.currentRoleId) return;
    
    const newTitle = dom.setConversationName.value.trim();
    
    try {
        const res = await fetch(`/api/roles/${state.currentRoleId}/conversations/${state.currentConversationId}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name: newTitle })
        });
        
        if (res.ok) {
            dom.conversationSettingsModal.style.display = 'none';
            const convIndex = state.conversations.findIndex(c => c.conversation_id === state.currentConversationId);
            if (convIndex > -1) {
                state.conversations[convIndex].name = newTitle;
                renderConversationCards();
            }
            if (dom.currentConversationTitle) {
                dom.currentConversationTitle.innerText = newTitle;
            }
        }
    } catch (e) {
        console.error('保存对话设置失败:', e);
    }
}

function showDeleteConversationModal() {
    if (!state.currentConversationId || !state.currentRoleId) return;
    if (dom.deleteConversationModal) {
        dom.deleteConversationModal.style.display = 'flex';
    }
}

async function confirmDeleteConversation() {
    if (!state.currentConversationId || !state.currentRoleId) return;
    
    try {
        const res = await fetch(`/api/roles/${state.currentRoleId}/conversations/${state.currentConversationId}`, {
            method: 'DELETE'
        });
        
        if (res.ok) {
            if (dom.deleteConversationModal) {
                dom.deleteConversationModal.style.display = 'none';
            }
            if (dom.conversationSettingsModal) {
                dom.conversationSettingsModal.style.display = 'none';
            }
            
            const convIndex = state.conversations.findIndex(c => c.conversation_id === state.currentConversationId);
            
            state.conversations = state.conversations.filter(c => c.conversation_id !== state.currentConversationId);
            
            if (state.conversations.length > 0) {
                let newIndex = convIndex > 0 ? convIndex - 1 : 0;
                await selectConversation(state.conversations[newIndex].conversation_id);
            } else {
                await createNewConversation();
            }
            
            renderConversationCards();
        }
    } catch (e) {
        console.error('删除对话失败:', e);
    }
}

// ==========================================
// 删除角色功能
// ==========================================
function toggleDeleteRoleMode() {
    state.isDeleteRoleMode = !state.isDeleteRoleMode;
    
    if (dom.btnDeleteRoleMode) {
        dom.btnDeleteRoleMode.classList.toggle('active', state.isDeleteRoleMode);
    }
    
    if (dom.deleteModeHint) {
        dom.deleteModeHint.style.display = state.isDeleteRoleMode ? 'block' : 'none';
    }
}

async function deleteRoleFirstStep(roleId, roleName) {
    console.log('deleteRoleFirstStep called, roleId:', roleId, 'roleName:', roleName);
    state.pendingDeleteRoleId = roleId;
    console.log('state.pendingDeleteRoleId set to:', state.pendingDeleteRoleId);
    
    if (dom.deleteRoleModal1) {
        const nameEl = dom.deleteRoleModal1.querySelector('.delete-role-name');
        if (nameEl) {
            nameEl.innerText = roleName;
        }
        dom.deleteRoleModal1.style.display = 'flex';
    }
}

async function deleteRoleSecondStep(roleId, roleName) {
    console.log('deleteRoleSecondStep called, roleId:', roleId, 'roleName:', roleName);
    
    if (dom.deleteRoleModal1) dom.deleteRoleModal1.style.display = 'none';
    
    let companionDays = 0;
    try {
        const res = await fetch(`/api/roles/${roleId}/companion_days`);
        if (res.ok) {
            const data = await res.json();
            companionDays = data.companion_days || 0;
        }
    } catch (e) {
        console.error('获取陪伴天数失败:', e);
    }
    
    if (dom.deleteRoleModal2) {
        const nameEl = dom.deleteRoleModal2.querySelector('.delete-role-name');
        const daysEl = dom.deleteRoleModal2.querySelector('.delete-days');
        if (nameEl) {
            nameEl.innerText = roleName;
        }
        if (daysEl) {
            daysEl.innerText = companionDays;
        }
        dom.deleteRoleModal2.style.display = 'flex';
    }
}

async function confirmDeleteRole(roleId) {
    console.log('开始删除角色, roleId:', roleId);
    
    try {
        const res = await fetch(`/api/roles/${roleId}`, {
            method: 'DELETE'
        });
        
        console.log('删除角色响应状态:', res.status);
        
        if (res.ok) {
            console.log('删除角色成功');
            
            if (dom.deleteRoleModal2) {
                dom.deleteRoleModal2.style.display = 'none';
            }
            
            state.roles = state.roles.filter(r => r.role_id !== roleId);
            
            if (state.currentRoleId === roleId) {
                state.currentRoleId = null;
                dom.chatMessages.innerHTML = '<div class="system-hint">请选择一个角色开始对话。</div>';
                if (dom.currentRoleTitle) dom.currentRoleTitle.innerText = '';
                if (dom.conversationMenuContainer) {
                    dom.conversationMenuContainer.style.display = 'none';
                }
            }
            
            if (state.isDeleteRoleMode) {
                toggleDeleteRoleMode();
            }
            
            renderRoleList();
        } else {
            console.error('删除角色失败, 响应状态:', res.status);
            const errorText = await res.text();
            console.error('错误信息:', errorText);
            alert('删除角色失败，请稍后重试');
        }
    } catch (e) {
        console.error('删除角色异常:', e);
        alert('删除角色失败：' + e.message);
    }
}

// ==========================================
// 核心切换与流式对话
// ==========================================
async function selectRole(id) {
    if (state.isDeleteRoleMode) {
        const role = state.roles.find(r => r.role_id === id);
        if (role) {
            deleteRoleFirstStep(id, role.display_name);
        }
        return;
    }
    
    if (dom.rightDrawer) dom.rightDrawer.classList.remove('open');
    if (dom.leftDrawer) dom.leftDrawer.classList.remove('open');

    state.currentRoleId = id;
    renderRoleList();

    const oldContainer = dom.chatMessages;
    const newContainer = oldContainer.cloneNode(false);
    oldContainer.parentNode.replaceChild(newContainer, oldContainer);
    dom.chatMessages = newContainer;

    state.isGenerating = false;
    currentAiBubble = null;
    currentAiThoughtNode = null;
    dom.userInput.disabled = false;
    dom.sendBtn.disabled = false;
    resetBubbleState();

    try {
        const resSettings = await fetch(`/api/roles/${id}/settings`);
        state.currentRoleMeta = await resSettings.json();
        document.getElementById('current-role-title').innerText = state.currentRoleMeta.display_name;

        updateThinkUI(!!state.currentRoleMeta.enable_think);
        
        // 加载深度回忆模式
        try {
            const resDepth = await fetch(`/api/roles/${id}/depth_recall_mode`);
            if (resDepth.ok) {
                const data = await resDepth.json();
                updateDepthRecallUI(data.depth_recall_mode || 'off');
            }
        } catch (e) {
            console.error('加载深度回忆模式失败:', e);
        }
        
        if (dom.conversationMenuContainer) {
            dom.conversationMenuContainer.style.display = 'block';
        }
        
        await fetchConversations(id);
        
        if (state.conversations.length === 0) {
            await createNewConversation();
        } else {
            await selectConversation(state.conversations[0].conversation_id);
        }
        
        if (dom.conversationMenuContainer) {
            dom.conversationMenuContainer.classList.remove('collapsed');
        }
    } catch (e) {
        console.error('加载角色失败:', e);
        dom.chatMessages.innerHTML = `<div class="system-hint" style="color:#ff4d4f">加载失败</div>`;
    }
}

function stripCitationContentFromText(text) {
    if (!text) return text;
    return text.replace(/\[来自知识库:.*?\][\s\S]*?\[\/知识库引用\]/g, '').trim();
}

async function loadChatHistory(roleId, conversationId = null) {
    let url = `/api/roles/${roleId}/history`;
    if (conversationId) {
        url = `/api/roles/${roleId}/conversations/${conversationId}/history`;
    }
    
    const res = await fetch(url);
    if (!res.ok) return;
    const history = await res.json();
    dom.chatMessages.innerHTML = '';

    if (history.length === 0) {
        dom.chatMessages.innerHTML = `<div class="system-hint">已连接，开始对话吧。</div>`;
        return;
    }
    history.forEach(msg => {
        if (msg.role === 'user') {
            var cleanContent = msg.content;
            if (msg.knowledge_citations && msg.knowledge_citations.length > 0) {
                cleanContent = stripCitationContentFromText(msg.content);
            }
            appendUserMessage(cleanContent, msg.images || [], msg.knowledge_citations || null);
        } else if (msg.role !== 'system') appendAIMessage(msg.content, msg.model, msg.token_usage);
    });
    const boundary = document.createElement('div');
    boundary.className = 'system-hint'; boundary.innerText = '--- 历史记忆 ---';
    dom.chatMessages.appendChild(boundary);
    smoothScrollToBottom();
}

function appendUserMessage(text, images, citations) {
    const row = document.createElement('div'); row.className = 'message-row user';
    const bubble = document.createElement('div'); bubble.className = 'message-bubble';
    if (images && images.length > 0) {
        const imgC = document.createElement('div'); imgC.className = 'message-images';
        images.forEach(img => { const el = document.createElement('img'); el.src = img; imgC.appendChild(el); });
        bubble.appendChild(imgC);
    }
    if (text) { const t = document.createElement('div'); t.innerText = text; bubble.appendChild(t); }

    if (citations && citations.length > 0) {
        const citationBar = renderCitationTags(citations, false);
        bubble.appendChild(citationBar);
    }

    const avatar = document.createElement('div'); avatar.className = 'msg-avatar';
    // 强制聊天气泡显示 1:1 圆形头像，忽略 gradient 模式
    avatar.innerHTML = renderAvatarDOM('circle', state.userProfile.avatar_circle, null, state.userProfile.display_name);

    row.appendChild(bubble); row.appendChild(avatar);
    dom.chatMessages.appendChild(row); smoothScrollToBottom();
}

function appendAIMessage(content, model = null, token_usage = null) {
    const row = document.createElement('div'); row.className = 'message-row ai';
    const bubble = document.createElement('div'); bubble.className = 'message-bubble';
    
    let bubbleHTML = `<div class="answer-content markdown-body">${marked.parse(content || "")}</div>`;
    
    // 添加模型信息和token使用量
    if (model || token_usage) {
        let barContent = '';
        const cachedValue = token_usage && token_usage.cached === '不可用' ? '不可用' : (token_usage ? (token_usage.cached || 0) : 0);
        const cachedDisplay = cachedValue === '不可用' ? '不可用' : `${cachedValue} token`;
        if (model) {
            barContent = `<span>模型：${getModelName(model)}`;
            if (token_usage) {
                barContent += ` | 输入 ${token_usage.input || 0} token | 输出 ${token_usage.output || 0} token | 缓存 ${cachedDisplay} | 总计 ${token_usage.total || 0} token`;
            }
            barContent += '</span>';
        } else if (token_usage) {
            barContent = `<span>输入 ${token_usage.input || 0} token | 输出 ${token_usage.output || 0} token | 缓存 ${cachedDisplay} | 总计 ${token_usage.total || 0} token</span>`;
        }
        bubbleHTML += `<div class="token-usage-bar visible">${barContent}</div>`;
    }
    
    bubble.innerHTML = bubbleHTML;

    const avatar = document.createElement('div'); avatar.className = 'msg-avatar';
    avatar.innerHTML = renderAvatarDOM('circle', state.currentRoleMeta.avatar_circle, null, state.currentRoleMeta.display_name);

    row.appendChild(avatar); row.appendChild(bubble);
    dom.chatMessages.appendChild(row);
}

function createAiStreamRow() {
    const row = document.createElement('div'); row.className = 'message-row ai';
    const avatar = document.createElement('div'); avatar.className = 'msg-avatar';
    avatar.innerHTML = renderAvatarDOM('circle', state.currentRoleMeta.avatar_circle, null, state.currentRoleMeta.display_name);

    const bubble = document.createElement('div'); bubble.className = 'message-bubble';
    bubble.innerHTML = `
        <details class="thought-details" style="display:none" open><summary class="thought-summary"> 思考过程</summary><div class="thought-content markdown-body"></div></details>
        <div class="answer-content markdown-body"></div><div class="token-usage-bar"></div>
    `;
    row.appendChild(avatar); row.appendChild(bubble);
    dom.chatMessages.appendChild(row);
    currentAiThoughtNode = row.querySelector('.thought-content');
    currentAiBubble = row.querySelector('.answer-content');
}

function getModelName(modelId) {
    if (state.models && state.models[modelId]) {
        return state.models[modelId].name || modelId;
    }
    return modelId;
}

// WebSocket 接收流
function initGlobalWebSocket(onReady) {
    if (ws && (ws.readyState === 1 || ws.readyState === 0)) { 
        if(onReady) onReady(); 
        return; 
    }
    stopReconnect();
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/chat`);
    updateConnectionStatus(false);
    ws.onopen = () => { 
        reconnectAttempts = 0; 
        updateConnectionStatus(true);
        if(onReady) onReady(); 
    };
    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.role_id && data.role_id !== state.currentRoleId) return;
        if (data.conversation_id && data.conversation_id !== state.currentConversationId) return;
        
        if (data.msg_type === "status" && data.content === "[DONE]") {
            hideTypingIndicator();
            if (currentAiBubble) currentAiBubble.innerHTML = marked.parse(state.activeAiText);
            state.isGenerating = false; 
            currentAiBubble = null; 
            state.activeAiText = ""; 
            state.activeAiThoughtText = "";
            dom.userInput.disabled = false;
            dom.sendBtn.disabled = false;
        } else if (data.msg_type === "summarizing") {
            updateSummarizingStatus(true);
        } else if (data.msg_type === "summarizing_done") {
            updateSummarizingStatus(false);
        } else if (data.msg_type === "error") {
            hideTypingIndicator();
            console.error("WebSocket Error:", data.content);
            state.isGenerating = false;
            dom.userInput.disabled = false;
            dom.sendBtn.disabled = false;
        } else if (data.msg_type === "usage") {
            const bar = currentAiBubble?.parentElement.querySelector('.token-usage-bar');
            if (bar) { 
                const cachedValue = data.content.cached === '不可用' ? '不可用' : (data.content.cached || 0);
                const cachedDisplay = cachedValue === '不可用' ? '不可用' : `${cachedValue} token`;
                let barContent = `<span>输入 ${data.content.input || 0} token | 输出 ${data.content.output || 0} token | 缓存 ${cachedDisplay} | 总计 ${data.content.total || 0} token</span>`;
                if (data.content.model) {
                    barContent = `<span>模型：${getModelName(data.content.model)} | 输入 ${data.content.input || 0} token | 输出 ${data.content.output || 0} token | 缓存 ${cachedDisplay} | 总计 ${data.content.total || 0} token</span>`;
                }
                bar.innerHTML = barContent; 
                bar.classList.add('visible'); 
            }
        } else {
            if (!currentAiBubble) {
                createAiStreamRow();
                hideTypingIndicator();
            }
            if (data.msg_type === "thought") {
                currentAiBubble.parentElement.querySelector('.thought-details').style.display = "block";
                state.activeAiThoughtText += data.content;
                currentAiThoughtNode.innerHTML = marked.parse(state.activeAiThoughtText);
            } else {
                state.activeAiText += data.content;
                currentAiBubble.innerHTML = marked.parse(state.activeAiText);
            }
            smoothScrollToBottom();
        }
    };
    ws.onclose = () => { 
        updateConnectionStatus(false);
        const delay = calculateReconnectDelay();
        reconnectAttempts++;
        reconnectTimeout = setTimeout(() => initGlobalWebSocket(), delay);
    };
}

function showTypingIndicator() {
    if (typingIndicatorNode) return;
    const row = document.createElement('div'); row.className = 'message-row ai';
    const avatar = document.createElement('div'); avatar.className = 'msg-avatar';
    avatar.innerHTML = renderAvatarDOM('circle', state.currentRoleMeta.avatar_circle, null, state.currentRoleMeta.display_name);
    row.innerHTML = `<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>`;
    row.insertBefore(avatar, row.firstChild);
    dom.chatMessages.appendChild(row); smoothScrollToBottom();
    typingIndicatorNode = row;
}
function hideTypingIndicator() { if (typingIndicatorNode) { typingIndicatorNode.remove(); typingIndicatorNode = null; } }

function toggleBubble() {
    if (isBubbleExpanded) {
        collapseBubble();
    } else {
        expandBubble();
    }
}

function expandBubble() {
    isBubbleExpanded = true;
    if (dom.inputBubble) dom.inputBubble.style.display = 'block';
    if (dom.inputWrapper) dom.inputWrapper.classList.add('bubble-active');
    if (dom.bubbleTextarea) {
        dom.bubbleTextarea.value = dom.userInput.value;
        dom.bubbleTextarea.focus();
    }
    const toggle = document.getElementById('bubble-toggle');
    if (toggle) {
        toggle.classList.add('expanded');
        toggle.title = '收起多行输入';
    }
}

function collapseBubble() {
    isBubbleExpanded = false;
    if (dom.inputBubble) {
        dom.inputBubble.classList.add('collapsing');
        setTimeout(() => {
            dom.inputBubble.style.display = 'none';
            dom.inputBubble.classList.remove('collapsing');
        }, 200);
    }
    if (dom.userInput) dom.userInput.value = dom.bubbleTextarea.value;
    if (dom.inputWrapper) dom.inputWrapper.classList.remove('bubble-active');
    const toggle = document.getElementById('bubble-toggle');
    if (toggle) {
        toggle.classList.remove('expanded');
        toggle.title = '展开多行输入';
    }
}

function syncBubbleContent(source) {
    if (source === 'bubble' && dom.userInput) {
        dom.userInput.value = dom.bubbleTextarea.value;
    } else if (source === 'input' && dom.bubbleTextarea) {
        dom.bubbleTextarea.value = dom.userInput.value;
    }
}

function resetBubbleState() {
    if (isBubbleExpanded) {
        collapseBubble();
    }
    if (dom.userInput) dom.userInput.value = '';
    if (dom.bubbleTextarea) dom.bubbleTextarea.value = '';
    const toggle = document.getElementById('bubble-toggle');
    if (toggle) {
        toggle.style.display = 'none';
        toggle.classList.remove('expanded');
        toggle.title = '展开多行输入';
    }
    if (dom.inputWrapper) dom.inputWrapper.classList.remove('bubble-active');
}

async function sendMessage() {
    if (!state.currentRoleId || !state.currentConversationId || state.isGenerating) return;

    const text = isBubbleExpanded ? dom.bubbleTextarea.value.trim() : dom.userInput.value.trim();
    if (!text && state.selectedImages.length === 0 && state.pendingCitations.length === 0) return;

    const currentCitations = state.pendingCitations.length > 0 ? state.pendingCitations.map(function(c) {
        return { kb_id: c.kb_id, kb_name: c.kb_name, doc_id: c.doc_id, doc_name: c.doc_name, char_start: c.char_start, char_end: c.char_end };
    }) : [];

    appendUserMessage(text, state.selectedImages, currentCitations.length > 0 ? currentCitations : null);
    showTypingIndicator();

    const payload = {
        role_id: state.currentRoleId,
        conversation_id: state.currentConversationId,
        user_input: text,
        images: state.selectedImages,
        enable_think: state.enableThink,
        enable_search: state.enableSearch,
        depth_recall_mode: state.depthRecallMode,
        model: state.currentModel,
        knowledge_citations: currentCitations
    };
    state.isGenerating = true;

    if (isBubbleExpanded) {
        dom.bubbleTextarea.value = '';
        collapseBubble();
    }
    dom.userInput.value = '';
    dom.previewArea.innerHTML = '';
    state.selectedImages = [];
    state.pendingCitations = [];
    renderCitationTags([], true);

    ws.send(JSON.stringify(payload));
}

dom.sendBtn.onclick = sendMessage;

let inputAreaFocused = false;
let toggleClicked = false;

function isFocusInInputArea() {
    const active = document.activeElement;
    return active === dom.userInput || active === dom.bubbleTextarea;
}

function updateExpandButtonVisibility() {
    const toggle = document.getElementById('bubble-toggle');
    if (!toggle) return;
    const isActive = inputAreaFocused || toggleClicked || isBubbleExpanded;
    toggle.style.display = isActive ? 'flex' : 'none';
}

// 阻止按钮抢夺焦点
const toggleBtn = document.getElementById('bubble-toggle');
if (toggleBtn) {
    toggleBtn.addEventListener('mousedown', function(e) {
        e.preventDefault();
        toggleClicked = true;
        setTimeout(() => { toggleClicked = false; }, 200);
    });
}

dom.userInput.addEventListener('input', function() {
    syncBubbleContent('input');
    updateExpandButtonVisibility();
});

dom.userInput.addEventListener('focus', function() {
    inputAreaFocused = true;
    updateExpandButtonVisibility();
});

dom.userInput.addEventListener('blur', function() {
    setTimeout(() => {
        inputAreaFocused = isFocusInInputArea();
        updateExpandButtonVisibility();
    }, 0);
});

dom.userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

if (dom.bubbleTextarea) {
    dom.bubbleTextarea.addEventListener('input', function() {
        syncBubbleContent('bubble');
    });

    dom.bubbleTextarea.addEventListener('focus', function() {
        inputAreaFocused = true;
        updateExpandButtonVisibility();
    });

    dom.bubbleTextarea.addEventListener('blur', function() {
        setTimeout(() => {
            inputAreaFocused = isFocusInInputArea();
            updateExpandButtonVisibility();
        }, 0);
    });

    dom.bubbleTextarea.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

function isInputOverflowing() {
    const input = dom.userInput;
    if (!input) return false;
    const style = window.getComputedStyle(input);
    const lineHeight = parseFloat(style.lineHeight) || 24;
    const paddingTop = parseFloat(style.paddingTop) || 0;
    const paddingBottom = parseFloat(style.paddingBottom) || 0;
    const maxHeight = parseFloat(style.maxHeight) || 150;
    const availableHeight = maxHeight - paddingTop - paddingBottom;
    const maxLines = Math.floor(availableHeight / lineHeight);

    const tempDiv = document.createElement('div');
    tempDiv.style.cssText = `
        position: absolute; visibility: hidden; white-space: pre-wrap; word-wrap: break-word;
        font-family: ${style.fontFamily}; font-size: ${style.fontSize};
        line-height: ${style.lineHeight}; width: ${input.clientWidth}px;
        padding: ${style.paddingTop} ${style.paddingRight} ${style.paddingBottom} ${style.paddingLeft};
    `;
    tempDiv.textContent = input.value || '\u00A0';
    document.body.appendChild(tempDiv);
    const actualHeight = tempDiv.offsetHeight;
    document.body.removeChild(tempDiv);

    return actualHeight > availableHeight;
}

// 工具控制

function updateThinkUI(isEnable) {
    state.enableThink = isEnable;
    dom.thinkToggle.className = isEnable ? 'think-btn-active' : 'think-btn-inactive';
    dom.thinkToggle.innerText = isEnable ? '深度思考: 开' : '深度思考: 关';
}
dom.thinkToggle.onclick = async () => {
    if (!state.currentRoleId || state.isGenerating) return;
    updateThinkUI(!state.enableThink);
    const payload = { settings: { enable_think: state.enableThink } };
    await fetch(`/api/roles/${state.currentRoleId}/settings`, { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
};

function updateSearchUI(isEnable) {
    state.enableSearch = isEnable;
    dom.searchToggle.className = isEnable ? 'think-btn-active' : 'think-btn-inactive';
    dom.searchToggle.innerText = isEnable ? '联网搜索: 开' : '联网搜索: 关';
}

function toggleSearch() {
    if (!state.currentRoleId || state.isGenerating) return;
    updateSearchUI(!state.enableSearch);
}

function updateSearchAvailability() {
    const currentModelInfo = state.models[state.currentModel];
    const isAvailable = currentModelInfo && currentModelInfo.supportsWebSearch;
    dom.searchToggle.disabled = !isAvailable;
    dom.searchToggle.style.opacity = isAvailable ? '1' : '0.5';
    dom.searchToggle.style.cursor = isAvailable ? 'pointer' : 'not-allowed';
    
    if (!isAvailable && state.enableSearch) {
        updateSearchUI(false);
    }
}

if (dom.searchToggle) {
    dom.searchToggle.onclick = toggleSearch;
}

function updateDepthRecallUI(mode) {
    state.depthRecallMode = mode;
    let className, text;
    switch (mode) {
        case 'off':
            className = 'depth-recall-off';
            text = '深度回忆: 关';
            break;
        case 'normal':
            className = 'depth-recall-normal';
            text = '深度回忆: 正常';
            break;
        case 'enhanced':
            className = 'depth-recall-enhanced';
            text = '深度回忆: 增强';
            break;
    }
    dom.depthRecallBtn.className = className;
    dom.depthRecallBtn.innerText = text;
}

function toggleDepthRecall() {
    if (!state.currentRoleId || state.isGenerating) return;
    const modes = ['off', 'normal', 'enhanced'];
    const currentIndex = modes.indexOf(state.depthRecallMode);
    const nextIndex = (currentIndex + 1) % modes.length;
    const nextMode = modes[nextIndex];
    updateDepthRecallUI(nextMode);
    fetch(`/api/roles/${state.currentRoleId}/depth_recall_mode`, {
        method: 'PUT',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ depth_recall_mode: nextMode })
    });
}

if (dom.depthRecallBtn) {
    dom.depthRecallBtn.onclick = toggleDepthRecall;
}

// ==========================================
// 上传下拉菜单控制
// ==========================================
const uploadDropdown = document.getElementById('upload-dropdown');
const uploadTrigger = document.getElementById('upload-trigger');
const uploadMenu = document.getElementById('upload-menu');

function toggleUploadMenu(event) {
    event.stopPropagation();
    const isShowing = uploadMenu.classList.contains('show');
    if (isShowing) {
        closeUploadMenu();
    } else {
        uploadMenu.style.display = 'block';
        void uploadMenu.offsetWidth;
        uploadMenu.classList.add('show');
    }
}

function closeUploadMenu() {
    uploadMenu.classList.remove('show');
    setTimeout(() => {
        if (!uploadMenu.classList.contains('show')) {
            uploadMenu.style.display = 'none';
        }
    }, 200);
}

function triggerImageUpload() {
    closeUploadMenu();
    document.getElementById('file-input').click();
}

if (uploadTrigger) {
    uploadTrigger.addEventListener('click', toggleUploadMenu);
}

document.addEventListener('click', (e) => {
    if (uploadDropdown && !uploadDropdown.contains(e.target)) {
        closeUploadMenu();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeUploadMenu();
    }
});

// ==========================================
// 图片上传与预览
// ==========================================
dom.fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    for (let f of files) {
        if (!f.type.startsWith('image/')) continue;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const b64 = ev.target.result; state.selectedImages.push(b64);
            const wrap = document.createElement('div'); wrap.className = 'preview-img';
            wrap.innerHTML = `<img src="${b64}"><button class="remove-btn">x</button>`;
            wrap.querySelector('button').onclick = () => { wrap.remove(); state.selectedImages = state.selectedImages.filter(i => i !== b64); };
            dom.previewArea.appendChild(wrap);
        };
        reader.readAsDataURL(f);
    }
    dom.fileInput.value = "";
});

// ==========================================
// 左右抽屉控制 (角色配置 & 个人中心)
// ==========================================
document.getElementById('open-settings').onclick = () => {
    if (!state.currentRoleId) return;
    dom.leftDrawer.classList.remove('open'); // 互斥
    document.getElementById('set-display-name').value = state.currentRoleMeta.display_name;
    document.getElementById('set-role-avatar-mode').value = state.currentRoleMeta.avatar_mode || 'circle';
    document.getElementById('set-prompt').value = state.currentRoleMeta.system_prompt || "";
    document.getElementById('set-temp').value = state.currentRoleMeta.temperature || 1.0;
    document.getElementById('set-budget').value = state.currentRoleMeta.thinking_budget || 81920;

    // 渲染大预览图 (使用实体预览引擎 renderPreviewDOM)
    const p = document.getElementById('set-role-avatar-preview');
    const mode = state.currentRoleMeta.avatar_mode;
    p.style.borderRadius = mode === 'gradient' ? '8px' : '50%';
    p.style.width = mode === 'gradient' ? '100%' : '80px';
    p.innerHTML = renderPreviewDOM(mode, state.currentRoleMeta.avatar_circle, state.currentRoleMeta.avatar_bg, state.currentRoleMeta.display_name);

    dom.rightDrawer.classList.add('open');
};
document.getElementById('close-settings').onclick = () => dom.rightDrawer.classList.remove('open');

document.getElementById('save-settings').onclick = async () => {
    const payload = {
        display_name: document.getElementById('set-display-name').value.trim(),
        avatar_mode: document.getElementById('set-role-avatar-mode').value,
        system_prompt: document.getElementById('set-prompt').value,
        settings: {
            temperature: parseFloat(document.getElementById('set-temp').value) || 1.0,
            thinking_budget: parseInt(document.getElementById('set-budget').value) || 81920
        }
    };

    // 1. 提交后端保存
    await fetch(`/api/roles/${state.currentRoleId}/settings`, { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    dom.rightDrawer.classList.remove('open');

    // 2. 【核心修复】直接在前端内存同步状态，彻底干掉破坏体验的 selectRole 重载
    state.currentRoleMeta.display_name = payload.display_name;
    state.currentRoleMeta.avatar_mode = payload.avatar_mode;
    state.currentRoleMeta.system_prompt = payload.system_prompt;
    document.getElementById('current-role-title').innerText = payload.display_name;

    // 3. 同步重绘左侧侧边栏，将最新名字和头像模式固定
    const rIndex = state.roles.findIndex(r => r.role_id === state.currentRoleId);
    if (rIndex > -1) {
        state.roles[rIndex].display_name = payload.display_name;
        state.roles[rIndex].avatar_mode = payload.avatar_mode;
        renderRoleList();
    }
};

// 个人中心抽屉
document.getElementById('btn-personal-center').onclick = () => {
    dom.rightDrawer.classList.remove('open'); // 互斥
    document.getElementById('set-user-name').value = state.userProfile.display_name;
    document.getElementById('set-user-avatar-mode').value = state.userProfile.avatar_mode;

    const p = document.getElementById('set-user-avatar-preview');
    p.style.borderRadius = state.userProfile.avatar_mode === 'gradient' ? '8px' : '50%';
    p.style.width = state.userProfile.avatar_mode === 'gradient' ? '100%' : '80px';
    p.innerHTML = renderPreviewDOM(state.userProfile.avatar_mode, state.userProfile.avatar_circle, state.userProfile.avatar_bg, state.userProfile.display_name);

    dom.leftDrawer.classList.add('open');
};
document.getElementById('close-user-drawer').onclick = () => dom.leftDrawer.classList.remove('open');
document.getElementById('save-user-settings').onclick = async () => {
    const payload = {
        display_name: document.getElementById('set-user-name').value.trim(),
        avatar_mode: document.getElementById('set-user-avatar-mode').value
    };
    await fetch('/api/user', { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
    dom.leftDrawer.classList.remove('open');
    fetchUserProfile(); // 刷新侧边栏
};

// 退出服务按钮
const btnShutdownService = document.getElementById('btn-shutdown-service');
if (btnShutdownService) {
    btnShutdownService.onclick = async () => {
        if (!confirm('确定要退出服务吗？\n\n退出前将自动保存所有未总结的记忆和未存档的记录。')) {
            return;
        }
        
        // 显示等待提示
        btnShutdownService.disabled = true;
        btnShutdownService.innerText = '正在保存数据并退出...';
        
        try {
            const res = await fetch('/api/shutdown', { method: 'POST' });
            if (res.ok) {
                alert('服务正在安全关闭...\n\n所有未总结的记忆和未存档的记录正在保存中。\n程序将在3秒后自动退出。');
            } else {
                alert('退出服务失败，请稍后重试或使用 Ctrl+C 直接退出。');
                btnShutdownService.disabled = false;
                btnShutdownService.innerText = '退出服务';
            }
        } catch (e) {
            console.error('退出服务失败:', e);
            alert('退出服务失败，请稍后重试或使用 Ctrl+C 直接退出。');
            btnShutdownService.disabled = false;
            btnShutdownService.innerText = '退出服务';
        }
    };
}

// ==========================================
// 纯手工 Vanilla JS 图片裁剪引擎 (二段式状态机)
// ==========================================
const cropper = {
    modal: document.getElementById('cropper-modal'),
    canvas: document.getElementById('cropper-canvas'),
    ctx: document.getElementById('cropper-canvas').getContext('2d'),
    box: document.getElementById('cropper-box'),
    fileInput: document.getElementById('cropper-file-input'),
    img: new Image(),
    targetType: null, // 'user' | 'role' | 'create_role'
    roleId: null,

    // 两段式缓存
    step: 1,
    cachedCircleBase64: null,
    cachedBgBase64: null,

    // 拖拽与缩放状态
    isDragging: false,
    startX: 0, startY: 0,
    imgX: 0, imgY: 0, imgScale: 1
};

function openCropper(targetType, roleId = null) {
    cropper.targetType = targetType;
    cropper.roleId = roleId;
    cropper.step = 1;

    // 第一步强制锁定为 1:1 圆形
    cropper.box.style.width = '240px';
    cropper.box.style.height = '240px';
    cropper.box.style.borderRadius = '50%';

    const btn = document.getElementById('cropper-confirm-btn');
    if (targetType === 'create_role') {
        btn.innerText = "确认裁剪";
    } else {
        btn.innerText = "下一步: 截取横向卡片背景 (1/2)";
    }

    cropper.ctx.clearRect(0, 0, cropper.canvas.width, cropper.canvas.height);
    cropper.modal.style.display = 'flex';
    cropper.fileInput.click();
}

function closeCropper() { cropper.modal.style.display = 'none'; cropper.fileInput.value = ''; }

cropper.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
        cropper.img.onload = () => { initCropperCanvas(); };
        cropper.img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
});

function initCropperCanvas() {
    const body = document.querySelector('.cropper-body');
    const cw = body.clientWidth, ch = body.clientHeight;
    cropper.canvas.width = cw; cropper.canvas.height = ch;

    const scaleX = cw / cropper.img.width;
    const scaleY = ch / cropper.img.height;
    cropper.imgScale = Math.max(scaleX, scaleY) * 1.1;

    cropper.imgX = (cw - cropper.img.width * cropper.imgScale) / 2;
    cropper.imgY = (ch - cropper.img.height * cropper.imgScale) / 2;

    drawCropper();
}

function drawCropper() {
    cropper.ctx.clearRect(0, 0, cropper.canvas.width, cropper.canvas.height);
    cropper.ctx.drawImage(cropper.img, cropper.imgX, cropper.imgY, cropper.img.width * cropper.imgScale, cropper.img.height * cropper.imgScale);
}

const cropperBody = document.querySelector('.cropper-body');
cropperBody.addEventListener('mousedown', dragStart);
cropperBody.addEventListener('mousemove', dragMove);
window.addEventListener('mouseup', dragEnd);

cropperBody.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = cropper.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = 1.1;
    const direction = e.deltaY < 0 ? 1 : -1;
    let newScale = direction > 0 ? cropper.imgScale * zoomFactor : cropper.imgScale / zoomFactor;

    const minScaleX = cropper.box.offsetWidth / cropper.img.width;
    const minScaleY = cropper.box.offsetHeight / cropper.img.height;
    const minScale = Math.max(minScaleX, minScaleY);
    const maxScale = 5.0;

    if (newScale < minScale) newScale = minScale;
    if (newScale > maxScale) newScale = maxScale;

    cropper.imgX = mouseX - (mouseX - cropper.imgX) * (newScale / cropper.imgScale);
    cropper.imgY = mouseY - (mouseY - cropper.imgY) * (newScale / cropper.imgScale);
    cropper.imgScale = newScale;

    drawCropper();
}, { passive: false });

function dragStart(e) { cropper.isDragging = true; cropper.startX = e.clientX; cropper.startY = e.clientY; }
function dragMove(e) {
    if (!cropper.isDragging) return;
    const dx = e.clientX - cropper.startX; const dy = e.clientY - cropper.startY;
    cropper.imgX += dx; cropper.imgY += dy;
    cropper.startX = e.clientX; cropper.startY = e.clientY;
    drawCropper();
}
function dragEnd() { cropper.isDragging = false; }

// 执行二段式裁剪上传引擎
async function confirmCrop() {
    if (!cropper.img.src) return alert("请先选择图片");

    const boxRect = cropper.box.getBoundingClientRect();
    const canvasRect = cropper.canvas.getBoundingClientRect();
    const cropX = boxRect.left - canvasRect.left;
    const cropY = boxRect.top - canvasRect.top;
    const cropW = boxRect.width;
    const cropH = boxRect.height;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = cropW; offCanvas.height = cropH;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(cropper.canvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    const base64Data = offCanvas.toDataURL('image/png');

    const btn = document.getElementById('cropper-confirm-btn');

    if (cropper.targetType === 'create_role') {
        window._pendingCreateAvatar = base64Data;
        document.getElementById('create-role-avatar-preview').innerHTML = `<img src="${base64Data}" style="border-radius:50%; width:100%; height:100%; object-fit:cover;">`;
        closeCropper();
        return;
    }

    if (cropper.step === 1) {
        cropper.cachedCircleBase64 = base64Data;
        cropper.step = 2;

        // 瞬间切换到横版卡片裁剪状态
        cropper.box.style.width = '400px';
        cropper.box.style.height = '100px';
        cropper.box.style.borderRadius = '8px';
        btn.innerText = "确认并上传 (2/2)";
        return;
    }

    if (cropper.step === 2) {
        cropper.cachedBgBase64 = base64Data;
        btn.innerText = "上传中..."; btn.disabled = true;

        try {
            const payload = {
                target_type: cropper.targetType,
                role_id: cropper.roleId,
                image_circle_base64: cropper.cachedCircleBase64,
                image_bg_base64: cropper.cachedBgBase64
            };
            const res = await fetch('/api/upload_avatar', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });

            if (res.ok) {
                closeCropper();
                const jsonRes = await res.json();
                const paths = jsonRes.paths;

                const previewEl = document.getElementById(cropper.targetType === 'user' ? 'set-user-avatar-preview' : 'set-role-avatar-preview');
                const currentMode = document.getElementById(cropper.targetType === 'user' ? 'set-user-avatar-mode' : 'set-role-avatar-mode').value;

                if (cropper.targetType === 'user') {
                    state.userProfile.avatar_circle = paths.avatar_circle;
                    state.userProfile.avatar_bg = paths.avatar_bg;
                    renderUserSidebar(); // 【直接重绘前端，绝不拉取旧数据】
                }
                if (cropper.targetType === 'role') {
                    state.currentRoleMeta.avatar_circle = paths.avatar_circle;
                    state.currentRoleMeta.avatar_bg = paths.avatar_bg;

                    const rIndex = state.roles.findIndex(r => r.role_id === cropper.roleId);
                    if (rIndex > -1) {
                        state.roles[rIndex].avatar_circle = paths.avatar_circle;
                        state.roles[rIndex].avatar_bg = paths.avatar_bg;
                        renderRoleList(); // 【直接重绘前端，绝不拉取旧数据】
                    }
                }

                // 强制实体预览 DOM 刷新
                if (previewEl) {
                    const meta = cropper.targetType === 'user' ? state.userProfile : state.currentRoleMeta;
                    previewEl.innerHTML = renderPreviewDOM(currentMode, meta.avatar_circle, meta.avatar_bg, meta.display_name);
                }

            } else { alert("上传失败"); }
        } catch(e) { console.error(e); }
        finally { btn.disabled = false; }
    }
}

// ==========================================
// 创建角色弹窗控制
// ==========================================
document.getElementById('btn-create-role').onclick = () => {
    document.getElementById('create-role-modal').style.display = 'flex';
    document.getElementById('create-role-name').value = '';
    window._pendingCreateAvatar = null;
    document.getElementById('create-role-avatar-preview').innerHTML = '';
};
document.getElementById('cancel-create-role').onclick = () => document.getElementById('create-role-modal').style.display = 'none';

document.getElementById('submit-create-role').onclick = async () => {
    const name = document.getElementById('create-role-name').value.trim();
    if (!name) return alert("必须给角色起个名字哦！");

    const payload = {
        name: name, system_prompt: document.getElementById('create-role-prompt').value,
        temperature: parseFloat(document.getElementById('create-temp').value) || 1.0,
        thinking_budget: 81920, enable_think: true
    };

    const btn = document.getElementById('submit-create-role');
    btn.disabled = true; btn.innerText = "创建中...";

    try {
        const res = await fetch('/api/roles', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
        if (res.ok) {
            const data = await res.json();
            if (window._pendingCreateAvatar) {
                await fetch('/api/upload_avatar', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ target_type: "role", role_id: data.role_id, image_circle_base64: window._pendingCreateAvatar })
                });
            }
            document.getElementById('create-role-modal').style.display = 'none';
            await fetchRoles(); selectRole(data.role_id);
        } else { alert("创建失败"); }
    } finally { btn.disabled = false; btn.innerText = "保存并创建"; }
};

// 监听模式切换时的实时预览与侧边栏“热更新”
document.getElementById('set-user-avatar-mode').addEventListener('change', (e) => {
    const mode = e.target.value;
    const p = document.getElementById('set-user-avatar-preview');
    p.style.borderRadius = mode === 'gradient' ? '8px' : '50%';
    p.style.width = mode === 'gradient' ? '100%' : '80px';
    p.innerHTML = renderPreviewDOM(mode, state.userProfile.avatar_circle, state.userProfile.avatar_bg, state.userProfile.display_name);

    // 【核心新增】：同步修改内存状态并实时重绘左下角个人中心，所见即所得
    state.userProfile.avatar_mode = mode;
    renderUserSidebar();
});

document.getElementById('set-role-avatar-mode').addEventListener('change', (e) => {
    const mode = e.target.value;
    const p = document.getElementById('set-role-avatar-preview');
    p.style.borderRadius = mode === 'gradient' ? '8px' : '50%';
    p.style.width = mode === 'gradient' ? '100%' : '80px';
    p.innerHTML = renderPreviewDOM(mode, state.currentRoleMeta.avatar_circle, state.currentRoleMeta.avatar_bg, state.currentRoleMeta.display_name);

    // 【核心新增】：同步修改内存状态并实时重绘左侧角色选项卡
    state.currentRoleMeta.avatar_mode = mode;
    const rIndex = state.roles.findIndex(r => r.role_id === state.currentRoleId);
    if (rIndex > -1) {
        state.roles[rIndex].avatar_mode = mode;
        renderRoleList();
    }
});

// ==========================================
// Boot Sequence 系统自检
// ==========================================
async function runBootSequence() {
    const pb = document.getElementById('boot-progress-bar'), pt = document.getElementById('boot-status-text'), ol = document.getElementById('boot-loader');
    if (!ol) return;
    pb.style.width = '20%'; pt.innerText = "唤醒内核...";

    let ok = false; for(let i=0; i<60; i++) { try { if ((await fetch('/health')).ok) {ok=true; break;} } catch(e){} await new Promise(r=>setTimeout(r,1000)); }
    if (!ok) return pt.innerText = "启动超时";

    pb.style.width = '60%'; pt.innerText = "加载模型列表...";
    await initModelSelector();
    await new Promise(r=>setTimeout(r,300));

    pb.style.width = '70%'; pt.innerText = "同步用户与角色信息...";
    await fetchUserProfile(); await fetchRoles(); await new Promise(r=>setTimeout(r,300));

    pb.style.width = '80%'; pt.innerText = "初始化模型选择...";
    await loadPreferredModel();
    // 确保模型选择器反映用户的首选模型
    if (state.currentModel) {
        dom.modelSelect.value = state.currentModel;
        updateMultimodalSupport();
    }
    await new Promise(r=>setTimeout(r,200));

    pb.style.width = '90%'; pt.innerText = "建立神经连接...";
    await new Promise(r => initGlobalWebSocket(r));

    pb.style.width = '100%'; pt.innerText = "就绪。";
    setTimeout(() => { ol.classList.add('hidden'); setTimeout(() => ol.style.display = 'none', 600); }, 500);
}

window.onload = runBootSequence;

// ==========================================
// 模型管理系统
// ==========================================

let modelCache = {
    models: null,
    lastUpdate: null,
    updateInterval: 24 * 60 * 60 * 1000 // 24小时
};

async function fetchAvailableModels() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();
        state.models = data.models;
        console.log('从后端获取模型列表:', Object.keys(state.models));
        return Object.keys(state.models);
    } catch (e) {
        console.error('获取模型列表失败:', e);
        return [];
    }
}

function renderModelOptions(models) {
    dom.modelSelect.innerHTML = '';
    models.forEach(modelId => {
        const modelInfo = state.models[modelId] || { name: modelId };
        const option = document.createElement('option');
        option.value = modelId;
        option.textContent = modelInfo.name || modelId;
        dom.modelSelect.appendChild(option);
    });
    dom.modelSelect.value = state.currentModel;
}

async function initModelSelector() {
    const models = await fetchAvailableModels();
    renderModelOptions(models);
    updateMultimodalSupport();
    updateSearchAvailability();
}

function updateMultimodalSupport() {
    const currentModelInfo = state.models[state.currentModel];
    const supportsMultimodal = currentModelInfo && currentModelInfo.multimodal;
    
    if (dom.fileInput && dom.fileInput.parentElement) {
        dom.fileInput.disabled = !supportsMultimodal;
        dom.fileInput.parentElement.style.opacity = supportsMultimodal ? '1' : '0.5';
        dom.fileInput.parentElement.style.cursor = supportsMultimodal ? 'pointer' : 'not-allowed';
    }
}

async function switchModel(modelId) {
    if (state.isGenerating) {
        alert("对话进行中，无法切换模型");
        dom.modelSelect.value = state.currentModel;
        return;
    }

    const previousModel = state.currentModel;
    state.currentModel = modelId;
    
    dom.modelSelect.disabled = true;
    dom.modelSelect.style.opacity = '0.5';

    try {
        const res = await fetch('/api/settings', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model: modelId })
        });

        if (res.ok) {
            updateMultimodalSupport();
            updateSearchAvailability();
            localStorage.setItem('preferredModel', modelId);
            console.log(`模型已从 ${previousModel} 切换到 ${modelId}`);
        } else {
            throw new Error('保存模型设置失败');
        }
    } catch (e) {
        console.error("模型切换失败:", e);
        state.currentModel = previousModel;
        dom.modelSelect.value = previousModel;
        alert("模型切换失败，请重试");
    } finally {
        dom.modelSelect.disabled = false;
        dom.modelSelect.style.opacity = '1';
    }
}

dom.modelSelect.addEventListener('change', (e) => {
    switchModel(e.target.value);
});

async function loadPreferredModel() {
    const preferredModel = localStorage.getItem('preferredModel');
    if (preferredModel && state.models[preferredModel]) {
        state.currentModel = preferredModel;
        dom.modelSelect.value = preferredModel;
        updateMultimodalSupport();
        updateSearchAvailability();
    }
}

window.onload = runBootSequence;

// ==========================================
// 统计页面功能
// ==========================================
let statsCharts = {
    modelConversations: null,
    modelTokens: null,
    roleConversations: null,
    tokenInput: null,
    tokenOutput: null
};
let statsCurrentModelStats = null;
let statsCurrentModelId = null;

const statsDom = {
    drawer: document.getElementById('stats-drawer'),
    closeBtn: document.getElementById('close-stats-drawer'),
    openBtn: document.getElementById('open-stats-drawer'),
    mainPage: document.getElementById('stats-main-page'),
    modelsPage: document.getElementById('stats-models-page'),
    modelDetailPage: document.getElementById('stats-model-detail-page'),
    rolesPage: document.getElementById('stats-roles-page'),
    roleDetailPage: document.getElementById('stats-role-detail-page'),
    usagePage: document.getElementById('stats-usage-page'),
    tokenDetailPage: document.getElementById('stats-token-detail-page'),
    modelsList: document.getElementById('stats-models-list'),
    modelDetailContent: document.getElementById('stats-model-detail-content'),
    modelDetailTitle: document.getElementById('stats-model-detail-title'),
    rolesList: document.getElementById('stats-roles-list'),
    roleDetailContent: document.getElementById('stats-role-detail-content'),
    roleDetailTitle: document.getElementById('stats-role-detail-title'),
    usageContent: document.getElementById('stats-usage-content'),
    tokenDetailContent: document.getElementById('stats-token-detail-content')
};

// 打开统计抽屉
if (statsDom.openBtn) {
    statsDom.openBtn.onclick = () => {
        if (dom.leftDrawer) dom.leftDrawer.classList.remove('open');
        if (dom.rightDrawer) dom.rightDrawer.classList.remove('open');
        showStatsPage('main');
        statsDom.drawer.classList.add('open');
    };
}

if (statsDom.closeBtn) {
    statsDom.closeBtn.onclick = () => {
        statsDom.drawer.classList.remove('open');
        destroyStatsCharts();
    };
}

function showStatsPage(page, id = null) {
    statsDom.mainPage.style.display = 'none';
    statsDom.modelsPage.style.display = 'none';
    statsDom.modelDetailPage.style.display = 'none';
    statsDom.rolesPage.style.display = 'none';
    statsDom.roleDetailPage.style.display = 'none';
    statsDom.usagePage.style.display = 'none';
    statsDom.tokenDetailPage.style.display = 'none';
    
    destroyStatsCharts();
    
    switch(page) {
        case 'main':
            statsDom.mainPage.style.display = 'block';
            break;
        case 'models':
            statsDom.modelsPage.style.display = 'block';
            loadModelsStats();
            break;
        case 'modelDetail':
            statsDom.modelDetailPage.style.display = 'block';
            loadModelDetailStats(id);
            break;
        case 'roles':
            statsDom.rolesPage.style.display = 'block';
            loadRolesStats();
            break;
        case 'roleDetail':
            statsDom.roleDetailPage.style.display = 'block';
            loadRoleDetailStats(id);
            break;
        case 'usage':
            statsDom.usagePage.style.display = 'block';
            loadUsageStats();
            break;
        case 'tokenDetail':
            statsDom.tokenDetailPage.style.display = 'block';
            renderTokenDetail();
            break;
    }
}

function destroyStatsCharts() {
    Object.values(statsCharts).forEach(chart => {
        if (chart) chart.destroy();
    });
    statsCharts = {
        modelConversations: null,
        modelTokens: null,
        roleConversations: null,
        tokenInput: null,
        tokenOutput: null
    };
}

async function loadModelsStats() {
    try {
        const res = await fetch('/api/stats/models');
        const data = await res.json();
        renderModelsList(data.models);
    } catch (e) {
        console.error('加载模型统计失败:', e);
        statsDom.modelsList.innerHTML = '<div class="stats-empty">加载失败</div>';
    }
}

function renderModelsList(models) {
    if (!models || models.length === 0) {
        statsDom.modelsList.innerHTML = '<div class="stats-empty">暂无统计数据</div>';
        return;
    }
    
    statsDom.modelsList.innerHTML = models.map(m => `
        <div class="stats-list-item" onclick="showStatsPage('modelDetail', '${m.model_id}')">
            <span class="stats-item-name">${getModelName(m.model_id)}</span>
            <div class="stats-item-data">
                <div class="stats-item-count">${m.total_conversations} 次对话</div>
                <div class="stats-item-tokens">${m.total_tokens.toLocaleString()} Tokens</div>
            </div>
        </div>
    `).join('');
}

async function loadModelDetailStats(modelId) {
    try {
        const res = await fetch(`/api/stats/models/${modelId}`);
        const stats = await res.json();
        statsDom.modelDetailTitle.textContent = getModelName(modelId);
        statsCurrentModelStats = stats;
        statsCurrentModelId = modelId;
        document.getElementById('stats-token-detail-back').textContent = getModelName(modelId);
        renderModelDetail(stats);
    } catch (e) {
        console.error('加载模型详情失败:', e);
        statsDom.modelDetailContent.innerHTML = '<div class="stats-empty">加载失败</div>';
    }
}

function renderModelDetail(stats) {
    const totalInput = stats.total_input_tokens || 0;
    const totalOutput = stats.total_output_tokens || 0;
    const totalCached = stats.total_cached_tokens || 0;
    
    statsDom.modelDetailContent.innerHTML = `
        <div class="stats-detail-grid">
            <div class="stats-detail-card">
                <div class="stats-detail-title">总对话次数</div>
                <div class="stats-detail-value">${stats.total_conversations}</div>
            </div>
            <div class="stats-detail-card" style="cursor: pointer;" onclick="showStatsPage('tokenDetail')">
                <div class="stats-detail-title">总消耗 Tokens</div>
                <div class="stats-detail-value">${stats.total_tokens.toLocaleString()}</div>
                <div style="font-size: 0.75rem; color: rgba(255,255,255,0.6); margin-top: 4px;">点击查看详情</div>
            </div>
        </div>
        <div class="stats-chart-container">
            <div class="stats-chart-title">使用次数-时间图</div>
            <canvas id="model-conversations-chart"></canvas>
        </div>
        <div class="stats-chart-container">
            <div class="stats-chart-title">消耗Token-时间图</div>
            <canvas id="model-tokens-chart"></canvas>
        </div>
    `;
    
    setTimeout(() => {
        const convCtx = document.getElementById('model-conversations-chart');
        if (convCtx) {
            statsCharts.modelConversations = new Chart(convCtx, {
                type: 'line',
                data: {
                    labels: stats.conversations_timeline.map(t => t.date),
                    datasets: [{
                        label: '对话次数',
                        data: stats.conversations_timeline.map(t => t.count),
                        borderColor: 'oklch(0.60 0.15 250)',
                        backgroundColor: 'rgba(96, 150, 250, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: 'rgba(255,255,255,0.7)' } },
                        x: { ticks: { color: 'rgba(255,255,255,0.7)' } }
                    }
                }
            });
        }
        
        const tokensCtx = document.getElementById('model-tokens-chart');
        if (tokensCtx) {
            statsCharts.modelTokens = new Chart(tokensCtx, {
                type: 'bar',
                data: {
                    labels: stats.tokens_timeline.map(t => t.date),
                    datasets: [{
                        label: 'Token 消耗',
                        data: stats.tokens_timeline.map(t => t.count),
                        backgroundColor: 'rgba(96, 150, 250, 0.6)',
                        borderColor: 'oklch(0.60 0.15 250)',
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: 'rgba(255,255,255,0.7)' } },
                        x: { ticks: { color: 'rgba(255,255,255,0.7)' } }
                    }
                }
            });
        }
    }, 100);
}

async function loadRolesStats() {
    try {
        const res = await fetch('/api/stats/roles');
        const data = await res.json();
        renderRolesList(data.roles);
    } catch (e) {
        console.error('加载角色统计失败:', e);
        statsDom.rolesList.innerHTML = '<div class="stats-empty">加载失败</div>';
    }
}

function renderRolesList(roles) {
    if (!roles || roles.length === 0) {
        statsDom.rolesList.innerHTML = '<div class="stats-empty">暂无统计数据</div>';
        return;
    }
    
    const roleIdToName = {};
    state.roles.forEach(r => {
        roleIdToName[r.role_id] = r.display_name;
    });
    
    statsDom.rolesList.innerHTML = roles.map(r => {
        const roleName = roleIdToName[r.role_id] || r.role_id;
        return `
            <div class="stats-list-item" onclick="showStatsPage('roleDetail', '${r.role_id}')">
                <span class="stats-item-name">${roleName}</span>
                <div class="stats-item-data">
                    <div class="stats-item-count">${r.total_conversations} 次对话</div>
                </div>
            </div>
        `;
    }).join('');
}

async function loadRoleDetailStats(roleId) {
    try {
        const res = await fetch(`/api/stats/roles/${roleId}`);
        const stats = await res.json();
        
        const role = state.roles.find(r => r.role_id === roleId);
        statsDom.roleDetailTitle.textContent = role ? role.display_name : roleId;
        renderRoleDetail(stats);
    } catch (e) {
        console.error('加载角色详情失败:', e);
        statsDom.roleDetailContent.innerHTML = '<div class="stats-empty">加载失败</div>';
    }
}

function renderRoleDetail(stats) {
    let createdText;
    if (stats.created_seconds !== undefined) {
        const hours = Math.floor(stats.created_seconds / 3600);
        if (hours < 24) {
            createdText = `${hours} 小时`;
        } else {
            const days = Math.floor(stats.created_seconds / 86400);
            createdText = `${days} 天`;
        }
    } else {
        createdText = `${stats.created_days} 天`;
    }
    
    statsDom.roleDetailContent.innerHTML = `
        <div class="stats-detail-grid">
            <div class="stats-detail-card">
                <div class="stats-detail-title">总对话次数</div>
                <div class="stats-detail-value">${stats.total_conversations}</div>
            </div>
            <div class="stats-detail-card">
                <div class="stats-detail-title">创建时长</div>
                <div class="stats-detail-value">${createdText}</div>
            </div>
        </div>
        <div class="stats-chart-container">
            <div class="stats-chart-title">对话次数-时间图</div>
            <canvas id="role-conversations-chart"></canvas>
        </div>
        <div class="stats-placeholder">
            📝 记忆总览占位
        </div>
    `;
    
    setTimeout(() => {
        const convCtx = document.getElementById('role-conversations-chart');
        if (convCtx) {
            statsCharts.roleConversations = new Chart(convCtx, {
                type: 'line',
                data: {
                    labels: stats.conversations_timeline.map(t => t.date),
                    datasets: [{
                        label: '对话次数',
                        data: stats.conversations_timeline.map(t => t.count),
                        borderColor: 'oklch(0.60 0.15 250)',
                        backgroundColor: 'rgba(96, 150, 250, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: 'rgba(255,255,255,0.7)' } },
                        x: { ticks: { color: 'rgba(255,255,255,0.7)' } }
                    }
                }
            });
        }
    }, 100);
}

async function loadUsageStats() {
    try {
        const res = await fetch('/api/stats/usage');
        const stats = await res.json();
        renderUsageStats(stats);
    } catch (e) {
        console.error('加载用量统计失败:', e);
        statsDom.usageContent.innerHTML = '<div class="stats-empty">加载失败</div>';
    }
}

function renderUsageStats(stats) {
    const totalInput = stats.total_input_tokens || 0;
    const totalOutput = stats.total_output_tokens || 0;
    
    statsDom.usageContent.innerHTML = `
        <div class="stats-detail-grid">
            <div class="stats-detail-card">
                <div class="stats-detail-title">总输入 Tokens</div>
                <div class="stats-detail-value">${totalInput.toLocaleString()}</div>
            </div>
            <div class="stats-detail-card">
                <div class="stats-detail-title">总输出 Tokens</div>
                <div class="stats-detail-value">${totalOutput.toLocaleString()}</div>
            </div>
        </div>
        <div style="margin-top: 16px; padding: 12px; background: rgba(255,255,255,0.05); border-radius: 8px;">
            <div style="font-size: 0.875rem; color: rgba(255,255,255,0.6);">
                此 token 统计是所有模型的总和
            </div>
        </div>
    `;
}

function renderTokenDetail() {
    if (!statsCurrentModelStats) {
        statsDom.tokenDetailContent.innerHTML = '<div class="stats-empty">暂无数据</div>';
        return;
    }
    
    const stats = statsCurrentModelStats;
    const totalInput = stats.total_input_tokens || 0;
    const totalOutput = stats.total_output_tokens || 0;
    const totalCached = stats.total_cached_tokens || 0;
    
    statsDom.tokenDetailContent.innerHTML = `
        <div class="stats-detail-grid">
            <div class="stats-detail-card">
                <div class="stats-detail-title">总输入 Tokens</div>
                <div class="stats-detail-value">${totalInput.toLocaleString()}</div>
            </div>
            <div class="stats-detail-card">
                <div class="stats-detail-title">总输出 Tokens</div>
                <div class="stats-detail-value">${totalOutput.toLocaleString()}</div>
            </div>
            <div class="stats-detail-card">
                <div class="stats-detail-title">总缓存命中 Tokens</div>
                <div class="stats-detail-value">${totalCached.toLocaleString()}</div>
            </div>
        </div>
        <div class="stats-chart-container">
            <div class="stats-chart-title">总输入 Token-时间图</div>
            <canvas id="token-input-chart"></canvas>
        </div>
        <div class="stats-chart-container">
            <div class="stats-chart-title">总输出 Token-时间图</div>
            <canvas id="token-output-chart"></canvas>
        </div>
    `;
    
    setTimeout(() => {
        const inputCtx = document.getElementById('token-input-chart');
        if (inputCtx && stats.input_tokens_timeline) {
            statsCharts.tokenInput = new Chart(inputCtx, {
                type: 'line',
                data: {
                    labels: stats.input_tokens_timeline.map(t => t.date),
                    datasets: [{
                        label: '输入 Token',
                        data: stats.input_tokens_timeline.map(t => t.count),
                        borderColor: 'oklch(0.60 0.15 250)',
                        backgroundColor: 'rgba(96, 150, 250, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: 'rgba(255,255,255,0.7)' } },
                        x: { ticks: { color: 'rgba(255,255,255,0.7)' } }
                    }
                }
            });
        }
        
        const outputCtx = document.getElementById('token-output-chart');
        if (outputCtx && stats.output_tokens_timeline) {
            statsCharts.tokenOutput = new Chart(outputCtx, {
                type: 'line',
                data: {
                    labels: stats.output_tokens_timeline.map(t => t.date),
                    datasets: [{
                        label: '输出 Token',
                        data: stats.output_tokens_timeline.map(t => t.count),
                        borderColor: 'oklch(0.60 0.15 150)',
                        backgroundColor: 'rgba(96, 250, 150, 0.1)',
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: 'rgba(255,255,255,0.7)' } },
                        x: { ticks: { color: 'rgba(255,255,255,0.7)' } }
                    }
                }
            });
        }
    }, 100);
}

// ==========================================
// 新功能事件绑定
// ==========================================

if (dom.conversationMenuToggle) {
    dom.conversationMenuToggle.onclick = toggleConversationMenu;
}

const btnNewConversation = document.getElementById('btn-new-conversation');
if (btnNewConversation) {
    btnNewConversation.onclick = createNewConversation;
}

if (dom.openConversationSettings) {
    dom.openConversationSettings.onclick = () => {
        if (!state.currentConversationId) return;
        const conv = state.conversations.find(c => c.conversation_id === state.currentConversationId);
        if (conv && dom.setConversationName) {
            dom.setConversationName.value = conv.name || '';
        }
        if (dom.conversationSettingsModal) {
            dom.conversationSettingsModal.style.display = 'flex';
        }
    };
}

const cancelConversationSettings = document.getElementById('cancel-conversation-settings');
if (cancelConversationSettings) {
    cancelConversationSettings.onclick = () => {
        if (dom.conversationSettingsModal) {
            dom.conversationSettingsModal.style.display = 'none';
        }
    };
}

const saveConversationSettingsBtn = document.getElementById('save-conversation-settings');
if (saveConversationSettingsBtn) {
    saveConversationSettingsBtn.onclick = saveConversationSettings;
}

if (dom.deleteConversationBtn) {
    dom.deleteConversationBtn.onclick = showDeleteConversationModal;
}

const cancelDeleteConversation = document.getElementById('cancel-delete-conversation');
if (cancelDeleteConversation) {
    cancelDeleteConversation.onclick = () => {
        if (dom.deleteConversationModal) {
            dom.deleteConversationModal.style.display = 'none';
        }
    };
}

const confirmDeleteConversationBtn = document.getElementById('confirm-delete-conversation');
if (confirmDeleteConversationBtn) {
    confirmDeleteConversationBtn.onclick = confirmDeleteConversation;
}

if (dom.btnDeleteRoleMode) {
    dom.btnDeleteRoleMode.onclick = toggleDeleteRoleMode;
}

const cancelDeleteRole1 = document.getElementById('cancel-delete-1');
if (cancelDeleteRole1) {
    cancelDeleteRole1.onclick = () => {
        if (dom.deleteRoleModal1) dom.deleteRoleModal1.style.display = 'none';
    };
}

const confirmDeleteRole1 = document.getElementById('confirm-delete-1');
if (confirmDeleteRole1) {
    confirmDeleteRole1.onclick = async () => {
        const role = state.roles.find(r => r.role_id === state.pendingDeleteRoleId);
        if (role) {
            await deleteRoleSecondStep(state.pendingDeleteRoleId, role.display_name);
        }
    };
}

const cancelDeleteRole2 = document.getElementById('cancel-delete-2');
if (cancelDeleteRole2) {
    cancelDeleteRole2.onclick = () => {
        if (dom.deleteRoleModal2) dom.deleteRoleModal2.style.display = 'none';
    };
}

const confirmDeleteRole2 = document.getElementById('confirm-delete-2');
console.log('confirmDeleteRole2 element:', confirmDeleteRole2);
if (confirmDeleteRole2) {
    confirmDeleteRole2.onclick = async () => {
        console.log('confirmDeleteRole2 clicked!');
        console.log('state.pendingDeleteRoleId:', state.pendingDeleteRoleId);
        if (state.pendingDeleteRoleId) {
            await confirmDeleteRole(state.pendingDeleteRoleId);
        } else {
            console.error('state.pendingDeleteRoleId is empty!');
        }
    };
}

if (dom.deleteModeHint) {
    dom.deleteModeHint.style.display = 'none';
}

// ==========================================
// 知识库管理
// ==========================================
let kbState = {
    list: [],
    pendingDeleteId: null
};

function openKbDrawer() {
    dom.rightDrawer.classList.remove('open');
    loadKnowledgeBases();
    document.getElementById('kb-drawer').classList.add('open');
}

function closeKbDrawer() {
    document.getElementById('kb-drawer').classList.remove('open');
}

async function loadKnowledgeBases() {
    try {
        const res = await fetch('/api/knowledge-bases');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        kbState.list = await res.json();
        renderKbList();
        updateKbStats();
        renderSidebarKbList();
    } catch (e) {
        console.error('加载知识库失败:', e);
        const container = document.getElementById('kb-management-list');
        const emptyHint = document.getElementById('kb-empty-hint');
        if (container) {
            container.innerHTML = '<div style="text-align:center; padding:20px; color:#ff4d4f;">加载失败，请检查网络连接</div>';
        }
        if (emptyHint) emptyHint.style.display = 'none';
    }
}

function renderKbList() {
    const container = document.getElementById('kb-management-list');
    const emptyHint = document.getElementById('kb-empty-hint');

    if (kbState.list.length === 0) {
        container.innerHTML = '';
        emptyHint.style.display = 'block';
        return;
    }

    emptyHint.style.display = 'none';
    container.innerHTML = kbState.list.map(kb => `
        <div class="kb-card" data-kb-id="${kb.kb_id}">
            <div class="kb-card-header">
                <span class="kb-name">${escapeHtml(kb.name)}</span>
                <span class="kb-doc-count">${kb.document_count} 个文档</span>
            </div>
            ${kb.description ? `<div class="kb-card-body"><p class="kb-description">${escapeHtml(kb.description)}</p></div>` : ''}
            <div class="kb-card-footer">
                <span>${formatDate(kb.created_at)}</span>
                <button class="kb-delete-btn" onclick="event.stopPropagation(); showDeleteKbDialog('${kb.kb_id}', '${escapeHtml(kb.name)}')">删除</button>
            </div>
        </div>
    `).join('');
}

function renderSidebarKbList() {
    const container = document.getElementById('kb-list');
    if (!container) return;

    if (kbState.list.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = kbState.list.map(kb => `
        <div class="kb-sidebar-item flex-item" data-kb-id="${kb.kb_id}" onclick="openKbDrawer(); openKbDetail('${kb.kb_id}')" style="background: linear-gradient(135deg, var(--bg-l2), var(--bg-l3));">
            <span class="kb-sidebar-name">${escapeHtml(kb.name)}</span>
            <span class="kb-sidebar-count">${kb.document_count}文档</span>
        </div>
    `).join('');
}

function updateKbStats() {
    const total = kbState.list.length;
    const docTotal = kbState.list.reduce((sum, kb) => sum + kb.document_count, 0);
    document.getElementById('kb-total-count').textContent = total;
    document.getElementById('kb-doc-total-count').textContent = docTotal;
}

function showCreateKbDialog() {
    document.getElementById('new-kb-name').value = '';
    document.getElementById('new-kb-desc').value = '';
    document.getElementById('create-kb-modal').style.display = 'flex';
}

function closeCreateKbModal() {
    document.getElementById('create-kb-modal').style.display = 'none';
}

async function confirmCreateKb() {
    const name = document.getElementById('new-kb-name').value.trim();
    const desc = document.getElementById('new-kb-desc').value.trim();

    if (!name) {
        alert('请输入知识库名称');
        return;
    }

    try {
        const res = await fetch('/api/knowledge-bases', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, description: desc})
        });

        if (res.ok) {
            closeCreateKbModal();
            await loadKnowledgeBases();
        } else {
            const err = await res.json();
            alert('创建失败: ' + (err.detail || '未知错误'));
        }
    } catch (e) {
        alert('网络错误: ' + e.message);
    }
}

function showDeleteKbDialog(kbId, kbName) {
    kbState.pendingDeleteId = kbId;
    document.getElementById('delete-kb-message').textContent =
        `确定要删除知识库「${kbName}」吗？此操作将同时删除该知识库下的所有文档，且无法恢复。`;
    document.getElementById('delete-kb-modal').style.display = 'flex';
}

function closeDeleteKbModal() {
    document.getElementById('delete-kb-modal').style.display = 'none';
    kbState.pendingDeleteId = null;
}

async function confirmDeleteKb() {
    if (!kbState.pendingDeleteId) return;

    try {
        const res = await fetch(`/api/knowledge-bases/${kbState.pendingDeleteId}`, {
            method: 'DELETE'
        });

        if (res.ok) {
            closeDeleteKbModal();
            await loadKnowledgeBases();
        } else {
            alert('删除失败');
        }
    } catch (e) {
        alert('网络错误: ' + e.message);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(isoStr) {
    try {
        const d = new Date(isoStr);
        return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
    } catch { return isoStr; }
}

// 知识库事件绑定
document.getElementById('btn-create-kb').addEventListener('click', () => {
    openKbDrawer();
    setTimeout(showCreateKbDialog, 300);
});

document.getElementById('btn-create-kb-drawer').addEventListener('click', showCreateKbDialog);

// ===== 知识库详情视图 =====
let currentKbId = null;
let pendingDeleteDoc = null;  // { docId, filename }

function openKbDetail(kbId) {
    currentKbId = kbId;

    document.getElementById('kb-main-view').style.display = 'none';
    const detailView = document.getElementById('kb-detail-view');
    detailView.style.display = 'block';

    loadKbDetail(kbId);
}

function backToKbList() {
    currentKbId = null;
    document.getElementById('kb-detail-view').style.display = 'none';
    document.getElementById('kb-main-view').style.display = 'block';
}

async function loadKbDetail(kbId) {
    try {
        const res = await fetch(`/api/knowledge-bases/${kbId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const kb = await res.json();

        document.getElementById('detail-kb-name').textContent = kb.name;
        document.getElementById('edit-kb-name-input').value = kb.name || '';
        document.getElementById('edit-kb-desc-input').value = kb.description || '';

        renderDocList(kb.documents || []);

    } catch (e) {
        console.error('加载知识库详情失败:', e);
        const container = document.getElementById('kb-detail-doc-list');
        if (container) {
            container.innerHTML = '<div style="text-align:center; padding:20px; color:#ff4d4f;">加载失败，请检查网络连接</div>';
        }
    }
}

function renderDocList(documents) {
    const container = document.getElementById('kb-detail-doc-list');
    const emptyHint = document.getElementById('kb-doc-empty-hint');
    const countSpan = document.getElementById('detail-doc-count');

    countSpan.textContent = documents.length;

    if (documents.length === 0) {
        container.innerHTML = '';
        emptyHint.style.display = 'block';
        return;
    }

    emptyHint.style.display = 'none';

    const fileIcons = {'docx': 'DOC', 'xlsx': 'XLS', 'pdf': 'PDF'};

    container.innerHTML = documents.map(doc => `
        <div class="kb-doc-item" data-doc-id="${doc.doc_id}">
            <div class="kb-doc-icon">${fileIcons[doc.file_type] || 'FILE'}</div>
            <div class="kb-doc-info">
                <div class="kb-doc-name">${escapeHtml(doc.filename)}</div>
                <div class="kb-doc-meta">
                    <span>${formatFileSize(doc.file_size)}</span>
                    <span>${formatDate(doc.uploaded_at)}</span>
                </div>
            </div>
            <span class="kb-doc-status ${doc.parsed_status}">
                ${doc.parsed_status === 'success' ? '✓ 已解析' : doc.parsed_status === 'error' ? '✗ 解析失败' : '⏳ 处理中'}
            </span>
            <div class="kb-doc-actions">
                <button class="icon-btn preview-btn" title="预览" onclick="previewDocument('${doc.doc_id}')">预览</button>
                <button class="icon-btn delete-btn" title="删除" onclick="deleteDocument('${doc.doc_id}', '${escapeHtml(doc.filename)}')">删除</button>
            </div>
        </div>
    `).join('');
}

async function saveKbSettings() {
    if (!currentKbId) return;

    const name = document.getElementById('edit-kb-name-input').value.trim();
    const desc = document.getElementById('edit-kb-desc-input').value.trim();

    if (!name) {
        alert('请输入知识库名称');
        return;
    }

    try {
        const res = await fetch(`/api/knowledge-bases/${currentKbId}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, description: desc})
        });

        if (res.ok) {
            document.getElementById('detail-kb-name').textContent = name;
            alert('保存成功');
        } else {
            const err = await res.json();
            alert('保存失败: ' + (err.detail || '未知错误'));
        }
    } catch (e) {
        alert('网络错误: ' + e.message);
    }
}

const kbFileInput = document.getElementById('kb-file-input');

if (kbFileInput) {
    kbFileInput.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (!files.length) return;

        for (let file of files) {
            await uploadFileToKb(file);
        }

        kbFileInput.value = '';
    });
}

async function uploadFileToKb(file) {
    if (!currentKbId) {
        alert('请先选择一个知识库');
        return;
    }

    const allowedExts = ['.docx', '.xlsx', '.pdf'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExts.includes(ext)) {
        alert(`不支持的文件格式: ${ext}`);
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        alert(`文件过大: ${(file.size / 1024 / 1024).toFixed(1)}MB，最大允许 50MB`);
        return;
    }

    const dropzone = document.getElementById('kb-upload-dropzone');
    const originalContent = dropzone ? dropzone.innerHTML : '';

    if (dropzone) {
        dropzone.innerHTML = `
            <div style="font-size: 32px; margin-bottom: 8px;">⏳</div>
            <div style="color: var(--text-primary); font-size: 14px; margin-bottom: 4px;">正在上传: ${escapeHtml(file.name)}</div>
            <div style="color: var(--text-secondary); font-size: 12px;">请稍候...</div>
        `;
        dropzone.style.pointerEvents = 'none';
        dropzone.style.opacity = '0.6';
    }

    try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`/api/knowledge-bases/${currentKbId}/documents`, {
            method: 'POST',
            body: formData
        });

        if (res.ok) {
            await loadKbDetail(currentKbId);
        } else {
            const err = await res.json();
            alert('上传失败: ' + (err.detail || '未知错误'));
        }
    } catch (e) {
        alert('上传错误: ' + e.message);
    } finally {
        if (dropzone) {
            dropzone.innerHTML = originalContent;
            dropzone.style.pointerEvents = '';
            dropzone.style.opacity = '';
        }
    }
}

async function previewDocument(docId) {
    if (!currentKbId) {
        alert('请先选择一个知识库');
        return;
    }

    try {
        const res = await fetch('/api/knowledge-bases');
        docSelectorState.knowledgeBases = await res.json();

        if (docSelectorState.knowledgeBases.length === 0) {
            alert('暂无可用知识库');
            return;
        }

        docSelectorState.selectedKbId = currentKbId;
        renderKbTabs();

        await loadDocumentsForKb(currentKbId);

        document.getElementById('kb-doc-selector').style.display = 'flex';

        docSelectorState.selectedDocId = docId;
        renderDocListForSelector();
        await loadAndRenderDocPreview(docId);

    } catch (e) {
        console.error('预览文档失败:', e);
        alert('预览失败: ' + e.message);
    }
}

function deleteDocument(docId, filename) {
    pendingDeleteDoc = { docId, filename };
    document.getElementById('delete-doc-message').textContent = 
        `确定要删除文档「${filename}」吗？此操作无法恢复。`;
    document.getElementById('delete-doc-modal').style.display = 'flex';
}

function closeDeleteDocModal() {
    document.getElementById('delete-doc-modal').style.display = 'none';
    pendingDeleteDoc = null;
}

async function confirmDeleteDoc() {
    if (!pendingDeleteDoc || !currentKbId) {
        closeDeleteDocModal();
        return;
    }

    const { docId, filename } = pendingDeleteDoc;

    try {
        const res = await fetch(`/api/knowledge-bases/${currentKbId}/documents/${docId}`, {
            method: 'DELETE'
        });

        if (res.ok) {
            closeDeleteDocModal();
            await loadKbDetail(currentKbId);
        } else {
            alert('删除失败');
        }
    } catch (e) {
        alert('删除错误: ' + e.message);
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

const kbUploadDropzone = document.getElementById('kb-upload-dropzone');
if (kbUploadDropzone) {
    kbUploadDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        kbUploadDropzone.classList.add('dragover');
    });

    kbUploadDropzone.addEventListener('dragleave', () => {
        kbUploadDropzone.classList.remove('dragover');
    });

    kbUploadDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        kbUploadDropzone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (!files.length) return;

        for (let file of files) {
            uploadFileToKb(file);
        }
    });
}

document.addEventListener('click', (e) => {
    const card = e.target.closest('.kb-card');
    if (card && !e.target.closest('.kb-delete-btn')) {
        const kbId = card.dataset.kbId;
        if (kbId) openKbDetail(kbId);
    }
});

// ==========================================
// 知识库文档选择器 (Task 9)
// ==========================================
let docSelectorState = {
    knowledgeBases: [],
    selectedKbId: null,
    documents: [],
    selectedDocId: null,
    parsedData: null,
    isFullscreen: false,
    _savedModalStyle: null
};

function renderCitationTag(citation, editable) {
    const tag = document.createElement('div');
    tag.className = 'citation-tag' + (editable ? '' : ' citation-tag-readonly');
    tag.dataset.kbId = citation.kb_id;
    tag.dataset.docId = citation.doc_id;
    tag.dataset.charStart = citation.char_start;
    tag.dataset.charEnd = citation.char_end;

    const nameSpan = document.createElement('span');
    nameSpan.className = 'citation-tag-filename';
    nameSpan.textContent = citation.doc_name;
    nameSpan.title = citation.kb_name + ' / ' + citation.doc_name;

    const rangeSpan = document.createElement('span');
    rangeSpan.className = 'citation-tag-range';
    rangeSpan.textContent = ' :' + citation.char_start + '～' + citation.char_end;

    tag.appendChild(nameSpan);
    tag.appendChild(rangeSpan);

    if (editable) {
        const closeBtn = document.createElement('button');
        closeBtn.className = 'citation-tag-close';
        closeBtn.innerHTML = '<svg width="10" height="10" viewBox="0 0 12 12"><path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="2" fill="none"/></svg>';
        closeBtn.onclick = function(e) {
            e.stopPropagation();
            removeCitationTag(citation);
        };
        tag.appendChild(closeBtn);
    }

    tag.onclick = function(e) {
        if (e.target.closest('.citation-tag-close')) return;
        showCitationPreview(citation);
    };

    return tag;
}

function renderCitationTags(citations, editable) {
    if (editable) {
        const container = document.getElementById('citation-tags-container');
        if (!container) return null;
        container.innerHTML = '';
        if (!citations || citations.length === 0) return null;
        citations.forEach(function(c) {
            container.appendChild(renderCitationTag(c, true));
        });
        return null;
    } else {
        const wrapper = document.createElement('div');
        wrapper.className = 'citation-tags-container';
        if (citations && citations.length > 0) {
            citations.forEach(function(c) {
                wrapper.appendChild(renderCitationTag(c, false));
            });
        }
        return wrapper;
    }
}

function removeCitationTag(citation) {
    const idx = state.pendingCitations.findIndex(function(c) {
        return c.kb_id === citation.kb_id && c.doc_id === citation.doc_id &&
               c.char_start === citation.char_start && c.char_end === citation.char_end;
    });
    if (idx > -1) {
        state.pendingCitations.splice(idx, 1);
    }
    renderCitationTags(state.pendingCitations, true);
}

function showCitationPreview(citation) {
    const modal = document.getElementById('citation-preview-modal');
    const titleEl = document.getElementById('citation-preview-title');
    const bodyEl = document.getElementById('citation-preview-body');

    if (!modal || !titleEl || !bodyEl) return;

    titleEl.textContent = citation.doc_name + ' :' + citation.char_start + '～' + citation.char_end;
    bodyEl.textContent = '加载中...';
    bodyEl.style.color = 'var(--text-secondary)';
    modal.style.display = 'flex';

    fetch('/api/knowledge-bases/' + citation.kb_id + '/documents/' + citation.doc_id + '/content?start=' + citation.char_start + '&end=' + citation.char_end)
        .then(function(res) { return res.json(); })
        .then(function(data) {
            if (data.error) {
                bodyEl.textContent = '加载失败: ' + data.error;
            } else {
                bodyEl.textContent = data.content || '(无内容)';
                bodyEl.style.color = 'var(--text-primary)';
            }
        })
        .catch(function(err) {
            bodyEl.textContent = '加载失败: ' + err.message;
        });
}

function closeCitationPreview() {
    var modal = document.getElementById('citation-preview-modal');
    if (modal) modal.style.display = 'none';
}

async function openKnowledgeBaseSelector() {
    closeUploadMenu();

    try {
        const res = await fetch('/api/knowledge-bases');
        docSelectorState.knowledgeBases = await res.json();

        if (docSelectorState.knowledgeBases.length === 0) {
            alert('暂无可用知识库，请先在知识库管理中创建');
            return;
        }

        const firstKb = docSelectorState.knowledgeBases.find(kb => kb.document_count > 0) || docSelectorState.knowledgeBases[0];
        docSelectorState.selectedKbId = firstKb.kb_id;

        renderKbTabs();
        await loadDocumentsForKb(firstKb.kb_id);

        document.getElementById('kb-doc-selector').style.display = 'flex';

    } catch (e) {
        console.error('加载知识库失败:', e);
        alert('加载失败');
    }
}

function closeKbDocSelector() {
    document.getElementById('kb-doc-selector').style.display = 'none';
    var btn = document.getElementById('kb-select-all-btn');
    if (btn) btn.style.display = 'none';
    clearDocSelection();
}

function renderKbTabs() {
    const container = document.getElementById('kb-selector-tabs');
    container.innerHTML = docSelectorState.knowledgeBases.map(kb => `
        <div class="kb-tab-item ${kb.kb_id === docSelectorState.selectedKbId ? 'active' : ''}"
             data-kb-id="${kb.kb_id}"
             onclick="selectKbTab('${kb.kb_id}')"
             title="${escapeHtml(kb.name)} (${kb.document_count}个文档)">
            ${escapeHtml(kb.name)}
        </div>
    `).join('');
}

async function selectKbTab(kbId) {
    if (kbId === docSelectorState.selectedKbId) return;

    docSelectorState.selectedKbId = kbId;
    docSelectorState.selectedDocId = null;
    docSelectorState.parsedData = null;

    renderKbTabs();
    resetPreviewArea();

    await loadDocumentsForKb(kbId);
}

async function loadDocumentsForKb(kbId) {
    try {
        const res = await fetch(`/api/knowledge-bases/${kbId}/documents`);
        docSelectorState.documents = await res.json();

        renderDocListForSelector();

    } catch (e) {
        console.error('加载文档列表失败:', e);
    }
}

function renderDocListForSelector() {
    const container = document.getElementById('kb-selector-doc-list');
    const emptyHint = document.getElementById('kb-selector-empty-hint');

    if (docSelectorState.documents.length === 0) {
        container.innerHTML = '';
        emptyHint.style.display = 'block';
        return;
    }

    emptyHint.style.display = 'none';

    const fileIcons = {'docx': 'DOC', 'xlsx': 'XLS', 'pdf': 'PDF'};

    container.innerHTML = docSelectorState.documents.map(doc => `
        <div class="selector-doc-item ${doc.doc_id === docSelectorState.selectedDocId ? 'selected' : ''}"
             data-doc-id="${doc.doc_id}"
             onclick="selectDocumentInSelector('${doc.doc_id}')">
            <div class="kb-doc-icon">${fileIcons[doc.file_type] || 'FILE'}</div>
            <div style="font-size:13px; color:var(--text-primary); word-break:break-all;">${escapeHtml(doc.filename)}</div>
            <div style="font-size:11px; color:var(--text-secondary); margin-top:2px;">${formatFileSize(doc.file_size)} · ${formatDate(doc.uploaded_at)}</div>
        </div>
    `).join('');
}

async function selectDocumentInSelector(docId) {
    docSelectorState.selectedDocId = docId;

    renderDocListForSelector();

    await loadAndRenderDocPreview(docId);
}

async function loadAndRenderDocPreview(docId) {
    const previewEl = document.getElementById('kb-doc-preview');
    previewEl.innerHTML = '<div style="text-align:center; padding:40px; color:var(--text-secondary);">⏳ 正在加载...</div>';

    try {
        const kbId = docSelectorState.selectedKbId;
        const res = await fetch(`/api/knowledge-bases/${kbId}/documents/${docId}`);
        const data = await res.json();

        docSelectorState.parsedData = data.parsed_data;

        const filename = data.filename || '未知文档';
        document.getElementById('kb-preview-title').textContent = filename;
        document.getElementById('fullscreen-preview-title').textContent = filename;

        if (data.parsed_data && data.parsed_data.sections) {
            renderParsedContent(data.parsed_data.sections, previewEl);
        } else {
            previewEl.innerHTML = `<p style="color:var(--text-secondary);">该文档尚未完成解析或解析失败。</p>`;
        }

    } catch (e) {
        previewEl.innerHTML = `<p style="color:#ff4d4f;">加载失败: ${e.message}</p>`;
    }
}

function renderParsedContent(sections, container) {
    let html = '';

    for (let si = 0; si < sections.length; si++) {
        const section = sections[si];
        let sectionHtml = '';
        switch (section.type) {
            case 'heading':
                const level = Math.min(Math.max(section.level || 1, 1), 3);
                sectionHtml += `<h${level}>${escapeHtml(section.content)}</h${level}>`;
                break;

            case 'paragraph':
                sectionHtml += `<p>${escapeHtml(section.content).replace(/\n/g, '<br>')}</p>`;
                break;

            case 'table':
                if (section.headers && section.headers.length > 0) {
                    sectionHtml += '<table><thead><tr>';
                    section.headers.forEach(h => {
                        sectionHtml += `<th>${escapeHtml(h)}</th>`;
                    });
                    sectionHtml += '</tr></thead><tbody>';

                    if (section.rows && section.rows.length > 0) {
                        section.rows.forEach(row => {
                            sectionHtml += '<tr>';
                            row.forEach(cell => {
                                sectionHtml += `<td>${escapeHtml(cell)}</td>`;
                            });
                            sectionHtml += '</tr>';
                        });
                    }

                    sectionHtml += '</tbody></table>';
                } else if (section.markdown) {
                    sectionHtml += `<pre style="white-space:pre-wrap;">${escapeHtml(section.markdown)}</pre>`;
                }
                break;

            default:
                if (section.content) {
                    sectionHtml += `<p>${escapeHtml(section.content)}</p>`;
                }
        }
        html += `<span data-section-index="${si}">${sectionHtml}</span>`;
    }

    container.innerHTML = html;

    docSelectorState._sectionOffsets = buildSectionOffsetMap(sections);

    var selectAllBtn = document.getElementById('kb-select-all-btn');
    if (!selectAllBtn) {
        selectAllBtn = document.createElement('button');
        selectAllBtn.id = 'kb-select-all-btn';
        selectAllBtn.textContent = '全选引用';
        selectAllBtn.className = 'btn-secondary';
        selectAllBtn.style.cssText = 'padding:4px 12px; font-size:12px;';
        selectAllBtn.onclick = function(e) { e.stopPropagation(); selectAllForCitation(); };
        var fullscreenBtn = document.querySelector('[onclick="toggleDocPreviewFullscreen()"]');
        if (fullscreenBtn && fullscreenBtn.parentElement) {
            fullscreenBtn.parentElement.insertBefore(selectAllBtn, fullscreenBtn);
        } else {
            container.parentNode.insertBefore(selectAllBtn, container);
        }
    }
    selectAllBtn.style.display = 'inline-block';
}

function resetPreviewArea() {
    document.getElementById('kb-doc-preview').innerHTML = `
        <div style="text-align:center; color:var(--text-secondary); padding:40px;">
            请从左侧选择一个文档
        </div>
    `;
    document.getElementById('kb-preview-title').textContent = '文档预览';
    updateSelectedTextLength(0);
}

function toggleDocPreviewFullscreen() {
    const modalEl = document.getElementById('kb-doc-selector');
    const modalContentEl = modalEl.querySelector('.modal-content.kb-doc-selector-content');

    if (!docSelectorState.isFullscreen) {
        docSelectorState._savedModalStyle = {
            borderRadius: modalContentEl.style.borderRadius,
            width: modalContentEl.style.width,
            maxWidth: modalContentEl.style.maxWidth,
            maxHeight: modalContentEl.style.maxHeight,
            height: modalContentEl.style.height
        };

        modalContentEl.style.borderRadius = '0';
        modalContentEl.style.width = '100vw';
        modalContentEl.style.maxWidth = '100vw';
        modalContentEl.style.maxHeight = '100vh';
        modalContentEl.style.height = '100vh';

        docSelectorState.isFullscreen = true;

        const btn = event.target;
        if (btn) btn.textContent = '退出全屏';
    } else {
        if (docSelectorState._savedModalStyle) {
            modalContentEl.style.borderRadius = docSelectorState._savedModalStyle.borderRadius || '';
            modalContentEl.style.width = docSelectorState._savedModalStyle.width || '';
            modalContentEl.style.maxWidth = docSelectorState._savedModalStyle.maxWidth || '';
            modalContentEl.style.maxHeight = docSelectorState._savedModalStyle.maxHeight || '';
            modalContentEl.style.height = docSelectorState._savedModalStyle.height || '';
        }

        docSelectorState.isFullscreen = false;

        const btn = event.target;
        if (btn) btn.textContent = '全屏';
    }
}

function updateSelectedTextLength(length) {
    document.getElementById('selected-text-length').textContent = length;
    document.getElementById('confirm-doc-btn').disabled = length <= 0;
}

function clearDocSelection() {
    if (window.getSelection) {
        window.getSelection().removeAllRanges();
    }
    updateSelectedTextLength(0);
}

function buildSectionOffsetMap(sections) {
    var offsets = [];
    var pos = 0;
    for (var i = 0; i < sections.length; i++) {
        offsets.push(pos);
        var s = sections[i];
        if (s.type === 'heading') {
            var level = Math.min(Math.max(s.level || 1, 1), 3);
            pos += 1 + level + 1 + (s.content || '').length + 1;
        } else if (s.type === 'paragraph') {
            pos += (s.content || '').length + 1;
        } else if (s.type === 'table' && s.markdown) {
            pos += 1 + (s.markdown || '').length + 1;
        } else if (s.content) {
            pos += (s.content || '').length + 1;
        }
    }
    return offsets;
}

function getCharOffsetInSection(sectionEl, node, offset) {
    var treeWalker = document.createTreeWalker(sectionEl, NodeFilter.SHOW_TEXT, null, false);
    var charOffset = 0;
    var current;
    while ((current = treeWalker.nextNode())) {
        if (current === node) {
            return charOffset + offset;
        }
        charOffset += current.textContent.length;
    }
    return charOffset + offset;
}

function findSectionIndex(node) {
    var el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
    while (el) {
        if (el.hasAttribute && el.hasAttribute('data-section-index')) {
            return parseInt(el.getAttribute('data-section-index'), 10);
        }
        el = el.parentElement;
    }
    return -1;
}

function getCharOffsetInPlainPreview(previewEl, node, offset, sectionOffsets) {
    var sectionIdx = findSectionIndex(node);
    if (sectionIdx < 0 || !sectionOffsets || sectionIdx >= sectionOffsets.length) {
        var tw = document.createTreeWalker(previewEl, NodeFilter.SHOW_TEXT, null, false);
        var off = 0;
        var cur;
        while ((cur = tw.nextNode())) {
            if (cur === node) return off + offset;
            off += cur.textContent.length;
        }
        return off + offset;
    }
    var sectionEl = previewEl.querySelector('[data-section-index="' + sectionIdx + '"]');
    if (!sectionEl) return sectionOffsets[sectionIdx] + offset;
    var intraOffset = getCharOffsetInSection(sectionEl, node, offset);
    return sectionOffsets[sectionIdx] + intraOffset;
}

function selectAllForCitation() {
    const currentDoc = docSelectorState.documents.find(d => d.doc_id === docSelectorState.selectedDocId);
    const currentKb = docSelectorState.knowledgeBases.find(k => k.kb_id === docSelectorState.selectedKbId);
    if (!currentDoc || !currentKb) return;

    const plainText = docSelectorState.parsedData ? (docSelectorState.parsedData.plain_text_preview || '') : '';
    const citation = {
        kb_id: currentKb.kb_id,
        kb_name: currentKb.name,
        doc_id: currentDoc.doc_id,
        doc_name: currentDoc.filename,
        char_start: 0,
        char_end: plainText.length,
        full_text: plainText
    };
    state.pendingCitations.push(citation);
    renderCitationTags(state.pendingCitations, true);
    closeKbDocSelector();
}

function confirmDocumentSelection() {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (!selectedText) {
        alert('请先在预览区域选择要引用的文本');
        return;
    }

    const currentDoc = docSelectorState.documents.find(d => d.doc_id === docSelectorState.selectedDocId);
    const currentKb = docSelectorState.knowledgeBases.find(k => k.kb_id === docSelectorState.selectedKbId);

    if (!currentDoc || !currentKb) return;

    const plainText = docSelectorState.parsedData ? (docSelectorState.parsedData.plain_text_preview || '') : '';
    let charStart = 0;
    let charEnd = plainText.length;

    try {
        const range = selection.getRangeAt(0);
        const previewEl = document.getElementById('kb-doc-preview');
        const offsets = docSelectorState._sectionOffsets || [];
        if (previewEl && previewEl.contains(range.startContainer) && previewEl.contains(range.endContainer)) {
            charStart = getCharOffsetInPlainPreview(previewEl, range.startContainer, range.startOffset, offsets);
            charEnd = getCharOffsetInPlainPreview(previewEl, range.endContainer, range.endOffset, offsets);
            if (charEnd < charStart) { var t = charEnd; charEnd = charStart; charStart = t; }
        } else {
            var idx = plainText.indexOf(selectedText);
            if (idx !== -1) { charStart = idx; charEnd = idx + selectedText.length; }
        }
    } catch(e) {
        var idx2 = plainText.indexOf(selectedText);
        if (idx2 !== -1) { charStart = idx2; charEnd = idx2 + selectedText.length; }
    }

    const citation = {
        kb_id: currentKb.kb_id,
        kb_name: currentKb.name,
        doc_id: currentDoc.doc_id,
        doc_name: currentDoc.filename,
        char_start: charStart,
        char_end: charEnd,
        full_text: selectedText
    };

    state.pendingCitations.push(citation);
    renderCitationTags(state.pendingCitations, true);

    closeKbDocSelector();
}

document.addEventListener('mouseup', handleTextSelection);
document.addEventListener('keyup', handleTextSelection);

function handleTextSelection() {
    const selector = document.getElementById('kb-doc-selector');
    if (!selector || selector.style.display === 'none') return;

    const selection = window.getSelection();
    const text = selection.toString().trim();

    if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const previewEl = document.getElementById('kb-doc-preview');
        if (previewEl && previewEl.contains(range.commonAncestorContainer)) {
            updateSelectedTextLength(text.length);
            return;
        }
    }
}