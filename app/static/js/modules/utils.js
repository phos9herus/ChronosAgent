// modules/utils.js — 纯工具函数

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

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}
