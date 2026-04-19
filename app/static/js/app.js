// app.js — 入口文件：事件绑定初始化
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
