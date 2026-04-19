// modules/stats.js — 统计页面 (Chart.js)

let statsCharts = {};

const statsDom = {
    mainPage: document.getElementById('stats-main-page'),
    modelsPage: document.getElementById('stats-models-page'),
    modelDetailPage: document.getElementById('stats-model-detail-page'),
    rolesPage: document.getElementById('stats-roles-page'),
    roleDetailPage: document.getElementById('stats-role-detail-page'),
    usagePage: document.getElementById('stats-usage-page'),
    tokenDetailPage: document.getElementById('stats-token-detail-page'),
    modelsList: document.getElementById('stats-models-list'),
    modelDetailTitle: document.getElementById('stats-model-detail-title'),
    modelDetailContent: document.getElementById('stats-model-detail-content'),
    rolesList: document.getElementById('stats-roles-list'),
    roleDetailTitle: document.getElementById('stats-role-detail-title'),
    roleDetailContent: document.getElementById('stats-role-detail-content'),
    usageContent: document.getElementById('stats-usage-content'),
    tokenDetailContent: document.getElementById('stats-token-detail-content')
};

let statsCurrentModelId = null;

const statsDrawer = document.getElementById('stats-drawer');
if (document.getElementById('open-stats-drawer')) {
    document.getElementById('open-stats-drawer').onclick = () => {
        dom.rightDrawer.classList.remove('open');
        dom.leftDrawer.classList.remove('open');
        statsDrawer.classList.add('open');
        showStatsPage('main');
    };
}
if (document.getElementById('close-stats-drawer')) {
    document.getElementById('close-stats-drawer').onclick = () => statsDrawer.classList.remove('open');
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
