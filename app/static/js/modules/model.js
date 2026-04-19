// modules/model.js — 模型选择器 + 启动序列

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
