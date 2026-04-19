// modules/roles.js — 角色列表 + 对话管理 + 侧边栏

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
// 创建角色弹窗控制 + 模式切换实时预览
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

// 监听模式切换时的实时预览与侧边栏"热更新"
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
