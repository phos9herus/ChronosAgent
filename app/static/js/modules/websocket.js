// modules/websocket.js — WebSocket 连接管理

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
