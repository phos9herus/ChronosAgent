// static/js/chat.js

// ==========================================
// 全局状态树
// ==========================================
const state = {
    userProfile: { display_name: "User", avatar_mode: "circle", avatar_circle: "", avatar_bg: "" },
    roles: [], // 缓存所有的角色列表
    currentRoleId: null,
    currentRoleMeta: {}, // 当前角色的 meta
    isGenerating: false,
    selectedImages: [],
    activeAiText: "",
    activeAiThoughtText: "",
    enableThink: false
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
    thinkToggle: document.getElementById('think-toggle-btn')
};

let ws = null;
let currentAiBubble = null;
let currentAiThoughtNode = null;
let typingIndicatorNode = null;
let reconnectAttempts = 0;

marked.setOptions({ highlight: null });

function smoothScrollToBottom() {
    requestAnimationFrame(() => { dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight; });
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
// 核心切换与流式对话
// ==========================================
async function selectRole(id) {
    if (dom.rightDrawer) dom.rightDrawer.classList.remove('open');
    if (dom.leftDrawer) dom.leftDrawer.classList.remove('open');

    state.currentRoleId = id;
    renderRoleList(); // 刷新激活态

    const oldContainer = dom.chatMessages;
    const newContainer = oldContainer.cloneNode(false);
    oldContainer.parentNode.replaceChild(newContainer, oldContainer);
    dom.chatMessages = newContainer;

    state.isGenerating = false;
    currentAiBubble = null;
    currentAiThoughtNode = null;
    dom.userInput.disabled = false;
    dom.sendBtn.disabled = false;

    try {
        const resSettings = await fetch(`/api/roles/${id}/settings`);
        state.currentRoleMeta = await resSettings.json();
        document.getElementById('current-role-title').innerText = state.currentRoleMeta.display_name;

        updateThinkUI(!!state.currentRoleMeta.enable_think);
        await loadChatHistory(id);
    } catch (e) {
        dom.chatMessages.innerHTML = `<div class="system-hint" style="color:#ff4d4f">加载失败</div>`;
    }
}

async function loadChatHistory(roleId) {
    const res = await fetch(`/api/roles/${roleId}/history`);
    if (!res.ok) return;
    const history = await res.json();
    dom.chatMessages.innerHTML = '';

    if (history.length === 0) {
        dom.chatMessages.innerHTML = `<div class="system-hint">已连接，开始对话吧。</div>`;
        return;
    }
    history.forEach(msg => {
        if (msg.role === 'user') appendUserMessage(msg.content, msg.images || []);
        else if (msg.role !== 'system') appendAIMessage(msg.content);
    });
    const boundary = document.createElement('div');
    boundary.className = 'system-hint'; boundary.innerText = '--- 历史记忆 ---';
    dom.chatMessages.appendChild(boundary);
    smoothScrollToBottom();
}

function appendUserMessage(text, images) {
    const row = document.createElement('div'); row.className = 'message-row user';
    const bubble = document.createElement('div'); bubble.className = 'message-bubble';
    if (images && images.length > 0) {
        const imgC = document.createElement('div'); imgC.className = 'message-images';
        images.forEach(img => { const el = document.createElement('img'); el.src = img; imgC.appendChild(el); });
        bubble.appendChild(imgC);
    }
    if (text) { const t = document.createElement('div'); t.innerText = text; bubble.appendChild(t); }

    const avatar = document.createElement('div'); avatar.className = 'msg-avatar';
    // 强制聊天气泡显示 1:1 圆形头像，忽略 gradient 模式
    avatar.innerHTML = renderAvatarDOM('circle', state.userProfile.avatar_circle, null, state.userProfile.display_name);

    row.appendChild(bubble); row.appendChild(avatar);
    dom.chatMessages.appendChild(row); smoothScrollToBottom();
}

function appendAIMessage(content) {
    const row = document.createElement('div'); row.className = 'message-row ai';
    const bubble = document.createElement('div'); bubble.className = 'message-bubble';
    bubble.innerHTML = `<div class="answer-content markdown-body">${marked.parse(content || "")}</div>`;

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

// WebSocket 接收流
function initGlobalWebSocket(onReady) {
    if (ws && (ws.readyState === 1 || ws.readyState === 0)) { if(onReady) onReady(); return; }
    ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/chat`);
    ws.onopen = () => { reconnectAttempts = 0; if(onReady) onReady(); };
    ws.onmessage = (e) => {
        const data = JSON.parse(e.data);
        if (data.role_id && data.role_id !== state.currentRoleId) return;
        if (data.msg_type !== "status" || data.content === "[DONE]") hideTypingIndicator();

        if (data.msg_type === "status" && data.content === "[DONE]") {
            if (currentAiBubble) currentAiBubble.innerHTML = marked.parse(state.activeAiText);
            state.isGenerating = false; currentAiBubble = null; state.activeAiText = ""; state.activeAiThoughtText = "";
        } else if (data.msg_type === "error") {
            alert(data.content); state.isGenerating = false;
        } else if (data.msg_type === "usage") {
            const bar = currentAiBubble?.parentElement.querySelector('.token-usage-bar');
            if (bar) { bar.innerHTML = `<span>总计消耗: ${data.content.total} Tokens</span>`; bar.classList.add('visible'); }
        } else {
            if (!currentAiBubble) createAiStreamRow();
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
    ws.onclose = () => { setTimeout(() => initGlobalWebSocket(), 2000); };
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

async function sendMessage() {
    if (!state.currentRoleId || state.isGenerating) return;
    const text = dom.userInput.value.trim();
    if (!text && state.selectedImages.length === 0) return;

    appendUserMessage(text, state.selectedImages);
    showTypingIndicator();

    const payload = { role_id: state.currentRoleId, user_input: text, images: state.selectedImages, enable_think: state.enableThink };
    state.isGenerating = true; dom.userInput.value = ''; dom.previewArea.innerHTML = ''; state.selectedImages = [];
    ws.send(JSON.stringify(payload));
}
dom.sendBtn.onclick = sendMessage;
dom.userInput.onkeydown = (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } };

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

    pb.style.width = '60%'; pt.innerText = "同步用户与角色信息...";
    await fetchUserProfile(); await fetchRoles(); await new Promise(r=>setTimeout(r,300));

    pb.style.width = '90%'; pt.innerText = "建立神经连接...";
    await new Promise(r => initGlobalWebSocket(r));

    pb.style.width = '100%'; pt.innerText = "就绪。";
    setTimeout(() => { ol.classList.add('hidden'); setTimeout(() => ol.style.display = 'none', 600); }, 500);
}

window.onload = runBootSequence;