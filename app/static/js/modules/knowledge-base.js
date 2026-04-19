// modules/knowledge-base.js — 知识库CRUD + 文档管理

let kbState = {
    list: [],
    pendingDeleteId: null
};

function openKbDrawer() {
    dom.rightDrawer.classList.remove('open');
    loadKnowledgeBases();
    document.getElementById('kb-drawer').classList.add('open');
}

function closeKbDrawer() {
    document.getElementById('kb-drawer').classList.remove('open');
}

async function loadKnowledgeBases() {
    try {
        const res = await fetch('/api/knowledge-bases');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        kbState.list = await res.json();
        renderKbList();
        updateKbStats();
        renderSidebarKbList();
    } catch (e) {
        console.error('加载知识库失败:', e);
        const container = document.getElementById('kb-management-list');
        const emptyHint = document.getElementById('kb-empty-hint');
        if (container) {
            container.innerHTML = '<div style="text-align:center; padding:20px; color:#ff4d4f;">加载失败，请检查网络连接</div>';
        }
        if (emptyHint) emptyHint.style.display = 'none';
    }
}

function renderKbList() {
    const container = document.getElementById('kb-management-list');
    const emptyHint = document.getElementById('kb-empty-hint');

    if (kbState.list.length === 0) {
        container.innerHTML = '';
        emptyHint.style.display = 'block';
        return;
    }

    emptyHint.style.display = 'none';
    container.innerHTML = kbState.list.map(kb => `
        <div class="kb-card" data-kb-id="${kb.kb_id}">
            <div class="kb-card-header">
                <span class="kb-name">${escapeHtml(kb.name)}</span>
                <span class="kb-doc-count">${kb.document_count} 个文档</span>
            </div>
            ${kb.description ? `<div class="kb-card-body"><p class="kb-description">${escapeHtml(kb.description)}</p></div>` : ''}
            <div class="kb-card-footer">
                <span>${formatDate(kb.created_at)}</span>
                <button class="kb-delete-btn" onclick="event.stopPropagation(); showDeleteKbDialog('${kb.kb_id}', '${escapeHtml(kb.name)}')">删除</button>
            </div>
        </div>
    `).join('');
}

function renderSidebarKbList() {
    const container = document.getElementById('kb-list');
    if (!container) return;

    if (kbState.list.length === 0) {
        container.innerHTML = '';
        return;
    }

    container.innerHTML = kbState.list.map(kb => `
        <div class="kb-sidebar-item flex-item" data-kb-id="${kb.kb_id}" onclick="openKbDrawer(); openKbDetail('${kb.kb_id}')" style="background: linear-gradient(135deg, var(--bg-l2), var(--bg-l3));">
            <span class="kb-sidebar-name">${escapeHtml(kb.name)}</span>
            <span class="kb-sidebar-count">${kb.document_count}文档</span>
        </div>
    `).join('');
}

function updateKbStats() {
    const total = kbState.list.length;
    const docTotal = kbState.list.reduce((sum, kb) => sum + kb.document_count, 0);
    document.getElementById('kb-total-count').textContent = total;
    document.getElementById('kb-doc-total-count').textContent = docTotal;
}

function showCreateKbDialog() {
    document.getElementById('new-kb-name').value = '';
    document.getElementById('new-kb-desc').value = '';
    document.getElementById('create-kb-modal').style.display = 'flex';
}

function closeCreateKbModal() {
    document.getElementById('create-kb-modal').style.display = 'none';
}

async function confirmCreateKb() {
    const name = document.getElementById('new-kb-name').value.trim();
    const desc = document.getElementById('new-kb-desc').value.trim();

    if (!name) {
        alert('请输入知识库名称');
        return;
    }

    try {
        const res = await fetch('/api/knowledge-bases', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, description: desc})
        });

        if (res.ok) {
            closeCreateKbModal();
            await loadKnowledgeBases();
        } else {
            const err = await res.json();
            alert('创建失败: ' + (err.detail || '未知错误'));
        }
    } catch (e) {
        alert('网络错误: ' + e.message);
    }
}

function showDeleteKbDialog(kbId, kbName) {
    kbState.pendingDeleteId = kbId;
    document.getElementById('delete-kb-message').textContent =
        `确定要删除知识库「${kbName}」吗？此操作将同时删除该知识库下的所有文档，且无法恢复。`;
    document.getElementById('delete-kb-modal').style.display = 'flex';
}

function closeDeleteKbModal() {
    document.getElementById('delete-kb-modal').style.display = 'none';
    kbState.pendingDeleteId = null;
}

async function confirmDeleteKb() {
    if (!kbState.pendingDeleteId) return;

    try {
        const res = await fetch(`/api/knowledge-bases/${kbState.pendingDeleteId}`, {
            method: 'DELETE'
        });

        if (res.ok) {
            closeDeleteKbModal();
            await loadKnowledgeBases();
        } else {
            alert('删除失败');
        }
    } catch (e) {
        alert('网络错误: ' + e.message);
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

// 知识库事件绑定
document.getElementById('btn-create-kb').addEventListener('click', () => {
    openKbDrawer();
    setTimeout(showCreateKbDialog, 300);
});

document.getElementById('btn-create-kb-drawer').addEventListener('click', showCreateKbDialog);

// ===== 知识库详情视图 =====
let currentKbId = null;
let pendingDeleteDoc = null;  // { docId, filename }

function openKbDetail(kbId) {
    currentKbId = kbId;

    document.getElementById('kb-main-view').style.display = 'none';
    const detailView = document.getElementById('kb-detail-view');
    detailView.style.display = 'block';

    loadKbDetail(kbId);
}

function backToKbList() {
    currentKbId = null;
    document.getElementById('kb-detail-view').style.display = 'none';
    document.getElementById('kb-main-view').style.display = 'block';
}

async function loadKbDetail(kbId) {
    try {
        const res = await fetch(`/api/knowledge-bases/${kbId}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const kb = await res.json();

        document.getElementById('detail-kb-name').textContent = kb.name;
        document.getElementById('edit-kb-name-input').value = kb.name || '';
        document.getElementById('edit-kb-desc-input').value = kb.description || '';

        renderDocList(kb.documents || []);

    } catch (e) {
        console.error('加载知识库详情失败:', e);
        const container = document.getElementById('kb-detail-doc-list');
        if (container) {
            container.innerHTML = '<div style="text-align:center; padding:20px; color:#ff4d4f;">加载失败，请检查网络连接</div>';
        }
    }
}

function renderDocList(documents) {
    const container = document.getElementById('kb-detail-doc-list');
    const emptyHint = document.getElementById('kb-doc-empty-hint');
    const countSpan = document.getElementById('detail-doc-count');

    countSpan.textContent = documents.length;

    if (documents.length === 0) {
        container.innerHTML = '';
        emptyHint.style.display = 'block';
        return;
    }

    emptyHint.style.display = 'none';

    const fileIcons = {'docx': 'DOC', 'xlsx': 'XLS', 'pdf': 'PDF'};

    container.innerHTML = documents.map(doc => `
        <div class="kb-doc-item" data-doc-id="${doc.doc_id}">
            <div class="kb-doc-icon">${fileIcons[doc.file_type] || 'FILE'}</div>
            <div class="kb-doc-info">
                <div class="kb-doc-name">${escapeHtml(doc.filename)}</div>
                <div class="kb-doc-meta">
                    <span>${formatFileSize(doc.file_size)}</span>
                    <span>${formatDate(doc.uploaded_at)}</span>
                </div>
            </div>
            <span class="kb-doc-status ${doc.parsed_status}">
                ${doc.parsed_status === 'success' ? '✓ 已解析' : doc.parsed_status === 'error' ? '✗ 解析失败' : '⏳ 处理中'}
            </span>
            <div class="kb-doc-actions">
                <button class="icon-btn preview-btn" title="预览" onclick="previewDocument('${doc.doc_id}')">预览</button>
                <button class="icon-btn delete-btn" title="删除" onclick="deleteDocument('${doc.doc_id}', '${escapeHtml(doc.filename)}')">删除</button>
            </div>
        </div>
    `).join('');
}

async function saveKbSettings() {
    if (!currentKbId) return;

    const name = document.getElementById('edit-kb-name-input').value.trim();
    const desc = document.getElementById('edit-kb-desc-input').value.trim();

    if (!name) {
        alert('请输入知识库名称');
        return;
    }

    try {
        const res = await fetch(`/api/knowledge-bases/${currentKbId}`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({name, description: desc})
        });

        if (res.ok) {
            document.getElementById('detail-kb-name').textContent = name;
            alert('保存成功');
        } else {
            const err = await res.json();
            alert('保存失败: ' + (err.detail || '未知错误'));
        }
    } catch (e) {
        alert('网络错误: ' + e.message);
    }
}

const kbFileInput = document.getElementById('kb-file-input');

if (kbFileInput) {
    kbFileInput.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (!files.length) return;

        for (let file of files) {
            await uploadFileToKb(file);
        }

        kbFileInput.value = '';
    });
}

async function uploadFileToKb(file) {
    if (!currentKbId) {
        alert('请先选择一个知识库');
        return;
    }

    const allowedExts = ['.docx', '.xlsx', '.pdf'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowedExts.includes(ext)) {
        alert(`不支持的文件格式: ${ext}`);
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        alert(`文件过大: ${(file.size / 1024 / 1024).toFixed(1)}MB，最大允许 50MB`);
        return;
    }

    const dropzone = document.getElementById('kb-upload-dropzone');
    const originalContent = dropzone ? dropzone.innerHTML : '';

    if (dropzone) {
        dropzone.innerHTML = `
            <div style="font-size: 32px; margin-bottom: 8px;">⏳</div>
            <div style="color: var(--text-primary); font-size: 14px; margin-bottom: 4px;">正在上传: ${escapeHtml(file.name)}</div>
            <div style="color: var(--text-secondary); font-size: 12px;">请稍候...</div>
        `;
        dropzone.style.pointerEvents = 'none';
        dropzone.style.opacity = '0.6';
    }

    try {
        const formData = new FormData();
        formData.append('file', file);

        const res = await fetch(`/api/knowledge-bases/${currentKbId}/documents`, {
            method: 'POST',
            body: formData
        });

        if (res.ok) {
            await loadKbDetail(currentKbId);
        } else {
            const err = await res.json();
            alert('上传失败: ' + (err.detail || '未知错误'));
        }
    } catch (e) {
        alert('上传错误: ' + e.message);
    } finally {
        if (dropzone) {
            dropzone.innerHTML = originalContent;
            dropzone.style.pointerEvents = '';
            dropzone.style.opacity = '';
        }
    }
}

async function previewDocument(docId) {
    if (!currentKbId) {
        alert('请先选择一个知识库');
        return;
    }

    try {
        const res = await fetch('/api/knowledge-bases');
        docSelectorState.knowledgeBases = await res.json();

        if (docSelectorState.knowledgeBases.length === 0) {
            alert('暂无可用知识库');
            return;
        }

        docSelectorState.selectedKbId = currentKbId;
        renderKbTabs();

        await loadDocumentsForKb(currentKbId);

        document.getElementById('kb-doc-selector').style.display = 'flex';

        docSelectorState.selectedDocId = docId;
        renderDocListForSelector();
        await loadAndRenderDocPreview(docId);

    } catch (e) {
        console.error('预览文档失败:', e);
        alert('预览失败: ' + e.message);
    }
}

function deleteDocument(docId, filename) {
    pendingDeleteDoc = { docId, filename };
    document.getElementById('delete-doc-message').textContent = 
        `确定要删除文档「${filename}」吗？此操作无法恢复。`;
    document.getElementById('delete-doc-modal').style.display = 'flex';
}

function closeDeleteDocModal() {
    document.getElementById('delete-doc-modal').style.display = 'none';
    pendingDeleteDoc = null;
}

async function confirmDeleteDoc() {
    if (!pendingDeleteDoc || !currentKbId) {
        closeDeleteDocModal();
        return;
    }

    const { docId, filename } = pendingDeleteDoc;

    try {
        const res = await fetch(`/api/knowledge-bases/${currentKbId}/documents/${docId}`, {
            method: 'DELETE'
        });

        if (res.ok) {
            closeDeleteDocModal();
            await loadKbDetail(currentKbId);
        } else {
            alert('删除失败');
        }
    } catch (e) {
        alert('删除错误: ' + e.message);
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1024 / 1024).toFixed(1) + ' MB';
}

const kbUploadDropzone = document.getElementById('kb-upload-dropzone');
if (kbUploadDropzone) {
    kbUploadDropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        kbUploadDropzone.classList.add('dragover');
    });

    kbUploadDropzone.addEventListener('dragleave', () => {
        kbUploadDropzone.classList.remove('dragover');
    });

    kbUploadDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        kbUploadDropzone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (!files.length) return;

        for (let file of files) {
            uploadFileToKb(file);
        }
    });
}

document.addEventListener('click', (e) => {
    const card = e.target.closest('.kb-card');
    if (card && !e.target.closest('.kb-delete-btn')) {
        const kbId = card.dataset.kbId;
        if (kbId) openKbDetail(kbId);
    }
});
