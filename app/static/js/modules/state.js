// modules/state.js — 全局状态树、DOM缓存、全局变量、常量

const state = {
    userProfile: { display_name: "User", avatar_mode: "circle", avatar_circle: "", avatar_bg: "" },
    roles: [],
    currentRoleId: null,
    currentRoleMeta: {},
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
    depthRecallMode: "off",
    models: {},
    pendingCitations: []
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
