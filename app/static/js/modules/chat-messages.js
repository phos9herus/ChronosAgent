// modules/chat-messages.js — 消息气泡渲染 + 历史加载

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
