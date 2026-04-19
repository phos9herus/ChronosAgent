// modules/input-ui.js — 输入气泡 + 发送 + 功能开关(T/S/D)

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
