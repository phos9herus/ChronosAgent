// modules/doc-selector.js — 文档选择器 + 引用Tag + 字符偏移计算

// ==========================================
// 知识库文档选择器 (Task 9)
// ==========================================
let docSelectorState = {
    knowledgeBases: [],
    selectedKbId: null,
    documents: [],
    selectedDocId: null,
    parsedData: null,
    isFullscreen: false,
    _savedModalStyle: null
};

function renderCitationTag(citation, editable) {
    const tag = document.createElement('div');
    tag.className = 'citation-tag' + (editable ? '' : ' citation-tag-readonly');
    tag.dataset.kbId = citation.kb_id;
    tag.dataset.docId = citation.doc_id;
    tag.dataset.charStart = citation.char_start;
    tag.dataset.charEnd = citation.char_end;

    const nameSpan = document.createElement('span');
    nameSpan.className = 'citation-tag-filename';
    nameSpan.textContent = citation.doc_name;
    nameSpan.title = citation.kb_name + ' / ' + citation.doc_name;

    const rangeSpan = document.createElement('span');
    rangeSpan.className = 'citation-tag-range';
    rangeSpan.textContent = ' :' + citation.char_start + '～' + citation.char_end;

    tag.appendChild(nameSpan);
    tag.appendChild(rangeSpan);

    if (editable) {
        const closeBtn = document.createElement('button');
        closeBtn.className = 'citation-tag-close';
        closeBtn.innerHTML = '<svg width="10" height="10" viewBox="0 0 12 12"><path d="M1 1l10 10M11 1L1 11" stroke="currentColor" stroke-width="2" fill="none"/></svg>';
        closeBtn.onclick = function(e) {
            e.stopPropagation();
            removeCitationTag(citation);
        };
        tag.appendChild(closeBtn);
    }

    tag.onclick = function(e) {
        if (e.target.closest('.citation-tag-close')) return;
        showCitationPreview(citation);
    };

    return tag;
}

function renderCitationTags(citations, editable) {
    if (editable) {
        const container = document.getElementById('citation-tags-container');
        if (!container) return null;
        container.innerHTML = '';
        if (!citations || citations.length === 0) return null;
        citations.forEach(function(c) {
            container.appendChild(renderCitationTag(c, true));
        });
        return null;
    } else {
        const wrapper = document.createElement('div');
        wrapper.className = 'citation-tags-container';
        if (citations && citations.length > 0) {
            citations.forEach(function(c) {
                wrapper.appendChild(renderCitationTag(c, false));
            });
        }
        return wrapper;
    }
}

function removeCitationTag(citation) {
    const idx = state.pendingCitations.findIndex(function(c) {
        return c.kb_id === citation.kb_id && c.doc_id === citation.doc_id &&
               c.char_start === citation.char_start && c.char_end === citation.char_end;
    });
    if (idx > -1) {
        state.pendingCitations.splice(idx, 1);
    }
    renderCitationTags(state.pendingCitations, true);
}

function showCitationPreview(citation) {
    const modal = document.getElementById('citation-preview-modal');
    const titleEl = document.getElementById('citation-preview-title');
    const bodyEl = document.getElementById('citation-preview-body');

    if (!modal || !titleEl || !bodyEl) return;

    titleEl.textContent = citation.doc_name + ' :' + citation.char_start + '～' + citation.char_end;
    bodyEl.textContent = '加载中...';
    bodyEl.style.color = 'var(--text-secondary)';
    modal.style.display = 'flex';

    fetch('/api/knowledge-bases/' + citation.kb_id + '/documents/' + citation.doc_id + '/content?start=' + citation.char_start + '&end=' + citation.char_end)
        .then(function(res) { return res.json(); })
        .then(function(data) {
            if (data.error) {
                bodyEl.textContent = '加载失败: ' + data.error;
            } else {
                bodyEl.textContent = data.content || '(无内容)';
                bodyEl.style.color = 'var(--text-primary)';
            }
        })
        .catch(function(err) {
            bodyEl.textContent = '加载失败: ' + err.message;
        });
}

function closeCitationPreview() {
    var modal = document.getElementById('citation-preview-modal');
    if (modal) modal.style.display = 'none';
}

async function openKnowledgeBaseSelector() {
    closeUploadMenu();

    try {
        const res = await fetch('/api/knowledge-bases');
        docSelectorState.knowledgeBases = await res.json();

        if (docSelectorState.knowledgeBases.length === 0) {
            alert('暂无可用知识库，请先在知识库管理中创建');
            return;
        }

        const firstKb = docSelectorState.knowledgeBases.find(kb => kb.document_count > 0) || docSelectorState.knowledgeBases[0];
        docSelectorState.selectedKbId = firstKb.kb_id;

        renderKbTabs();
        await loadDocumentsForKb(firstKb.kb_id);

        document.getElementById('kb-doc-selector').style.display = 'flex';

    } catch (e) {
        console.error('加载知识库失败:', e);
        alert('加载失败');
    }
}

function closeKbDocSelector() {
    document.getElementById('kb-doc-selector').style.display = 'none';
    var btn = document.getElementById('kb-select-all-btn');
    if (btn) btn.style.display = 'none';
    clearDocSelection();
}

function renderKbTabs() {
    const container = document.getElementById('kb-selector-tabs');
    container.innerHTML = docSelectorState.knowledgeBases.map(kb => `
        <div class="kb-tab-item ${kb.kb_id === docSelectorState.selectedKbId ? 'active' : ''}"
             data-kb-id="${kb.kb_id}"
             onclick="selectKbTab('${kb.kb_id}')"
             title="${escapeHtml(kb.name)} (${kb.document_count}个文档)">
            ${escapeHtml(kb.name)}
        </div>
    `).join('');
}

async function selectKbTab(kbId) {
    if (kbId === docSelectorState.selectedKbId) return;

    docSelectorState.selectedKbId = kbId;
    docSelectorState.selectedDocId = null;
    docSelectorState.parsedData = null;

    renderKbTabs();
    resetPreviewArea();

    await loadDocumentsForKb(kbId);
}

async function loadDocumentsForKb(kbId) {
    try {
        const res = await fetch(`/api/knowledge-bases/${kbId}/documents`);
        docSelectorState.documents = await res.json();

        renderDocListForSelector();

    } catch (e) {
        console.error('加载文档列表失败:', e);
    }
}

function renderDocListForSelector() {
    const container = document.getElementById('kb-selector-doc-list');
    const emptyHint = document.getElementById('kb-selector-empty-hint');

    if (docSelectorState.documents.length === 0) {
        container.innerHTML = '';
        emptyHint.style.display = 'block';
        return;
    }

    emptyHint.style.display = 'none';

    const fileIcons = {'docx': 'DOC', 'xlsx': 'XLS', 'pdf': 'PDF'};

    container.innerHTML = docSelectorState.documents.map(doc => `
        <div class="selector-doc-item ${doc.doc_id === docSelectorState.selectedDocId ? 'selected' : ''}"
             data-doc-id="${doc.doc_id}"
             onclick="selectDocumentInSelector('${doc.doc_id}')">
            <div class="kb-doc-icon">${fileIcons[doc.file_type] || 'FILE'}</div>
            <div style="font-size:13px; color:var(--text-primary); word-break:break-all;">${escapeHtml(doc.filename)}</div>
            <div style="font-size:11px; color:var(--text-secondary); margin-top:2px;">${formatFileSize(doc.file_size)} · ${formatDate(doc.uploaded_at)}</div>
        </div>
    `).join('');
}

async function selectDocumentInSelector(docId) {
    docSelectorState.selectedDocId = docId;

    renderDocListForSelector();

    await loadAndRenderDocPreview(docId);
}

async function loadAndRenderDocPreview(docId) {
    const previewEl = document.getElementById('kb-doc-preview');
    previewEl.innerHTML = '<div style="text-align:center; padding:40px; color:var(--text-secondary);">⏳ 正在加载...</div>';

    try {
        const kbId = docSelectorState.selectedKbId;
        const res = await fetch(`/api/knowledge-bases/${kbId}/documents/${docId}`);
        const data = await res.json();

        docSelectorState.parsedData = data.parsed_data;

        const filename = data.filename || '未知文档';
        document.getElementById('kb-preview-title').textContent = filename;
        document.getElementById('fullscreen-preview-title').textContent = filename;

        if (data.parsed_data && data.parsed_data.sections) {
            renderParsedContent(data.parsed_data.sections, previewEl);
        } else {
            previewEl.innerHTML = `<p style="color:var(--text-secondary);">该文档尚未完成解析或解析失败。</p>`;
        }

    } catch (e) {
        previewEl.innerHTML = `<p style="color:#ff4d4f;">加载失败: ${e.message}</p>`;
    }
}

function renderParsedContent(sections, container) {
    let html = '';

    for (let si = 0; si < sections.length; si++) {
        const section = sections[si];
        let sectionHtml = '';
        switch (section.type) {
            case 'heading':
                const level = Math.min(Math.max(section.level || 1, 1), 3);
                sectionHtml += `<h${level}>${escapeHtml(section.content)}</h${level}>`;
                break;

            case 'paragraph':
                sectionHtml += `<p>${escapeHtml(section.content).replace(/\n/g, '<br>')}</p>`;
                break;

            case 'table':
                if (section.headers && section.headers.length > 0) {
                    sectionHtml += '<table><thead><tr>';
                    section.headers.forEach(h => {
                        sectionHtml += `<th>${escapeHtml(h)}</th>`;
                    });
                    sectionHtml += '</tr></thead><tbody>';

                    if (section.rows && section.rows.length > 0) {
                        section.rows.forEach(row => {
                            sectionHtml += '<tr>';
                            row.forEach(cell => {
                                sectionHtml += `<td>${escapeHtml(cell)}</td>`;
                            });
                            sectionHtml += '</tr>';
                        });
                    }

                    sectionHtml += '</tbody></table>';
                } else if (section.markdown) {
                    sectionHtml += `<pre style="white-space:pre-wrap;">${escapeHtml(section.markdown)}</pre>`;
                }
                break;

            default:
                if (section.content) {
                    sectionHtml += `<p>${escapeHtml(section.content)}</p>`;
                }
        }
        html += `<span data-section-index="${si}">${sectionHtml}</span>`;
    }

    container.innerHTML = html;

    docSelectorState._sectionOffsets = buildSectionOffsetMap(sections);

    var selectAllBtn = document.getElementById('kb-select-all-btn');
    if (!selectAllBtn) {
        selectAllBtn = document.createElement('button');
        selectAllBtn.id = 'kb-select-all-btn';
        selectAllBtn.textContent = '全选引用';
        selectAllBtn.className = 'btn-secondary';
        selectAllBtn.style.cssText = 'padding:4px 12px; font-size:12px;';
        selectAllBtn.onclick = function(e) { e.stopPropagation(); selectAllForCitation(); };
        var fullscreenBtn = document.querySelector('[onclick="toggleDocPreviewFullscreen()"]');
        if (fullscreenBtn && fullscreenBtn.parentElement) {
            fullscreenBtn.parentElement.insertBefore(selectAllBtn, fullscreenBtn);
        } else {
            container.parentNode.insertBefore(selectAllBtn, container);
        }
    }
    selectAllBtn.style.display = 'inline-block';
}

function resetPreviewArea() {
    document.getElementById('kb-doc-preview').innerHTML = `
        <div style="text-align:center; color:var(--text-secondary); padding:40px;">
            请从左侧选择一个文档
        </div>
    `;
    document.getElementById('kb-preview-title').textContent = '文档预览';
    updateSelectedTextLength(0);
}

function toggleDocPreviewFullscreen() {
    const modalEl = document.getElementById('kb-doc-selector');
    const modalContentEl = modalEl.querySelector('.modal-content.kb-doc-selector-content');

    if (!docSelectorState.isFullscreen) {
        docSelectorState._savedModalStyle = {
            borderRadius: modalContentEl.style.borderRadius,
            width: modalContentEl.style.width,
            maxWidth: modalContentEl.style.maxWidth,
            maxHeight: modalContentEl.style.maxHeight,
            height: modalContentEl.style.height
        };

        modalContentEl.style.borderRadius = '0';
        modalContentEl.style.width = '100vw';
        modalContentEl.style.maxWidth = '100vw';
        modalContentEl.style.maxHeight = '100vh';
        modalContentEl.style.height = '100vh';

        docSelectorState.isFullscreen = true;

        const btn = event.target;
        if (btn) btn.textContent = '退出全屏';
    } else {
        if (docSelectorState._savedModalStyle) {
            modalContentEl.style.borderRadius = docSelectorState._savedModalStyle.borderRadius || '';
            modalContentEl.style.width = docSelectorState._savedModalStyle.width || '';
            modalContentEl.style.maxWidth = docSelectorState._savedModalStyle.maxWidth || '';
            modalContentEl.style.maxHeight = docSelectorState._savedModalStyle.maxHeight || '';
            modalContentEl.style.height = docSelectorState._savedModalStyle.height || '';
        }

        docSelectorState.isFullscreen = false;

        const btn = event.target;
        if (btn) btn.textContent = '全屏';
    }
}

function updateSelectedTextLength(length) {
    document.getElementById('selected-text-length').textContent = length;
    document.getElementById('confirm-doc-btn').disabled = length <= 0;
}

function clearDocSelection() {
    if (window.getSelection) {
        window.getSelection().removeAllRanges();
    }
    updateSelectedTextLength(0);
}

function buildSectionOffsetMap(sections) {
    var offsets = [];
    var pos = 0;
    for (var i = 0; i < sections.length; i++) {
        offsets.push(pos);
        var s = sections[i];
        if (s.type === 'heading') {
            var level = Math.min(Math.max(s.level || 1, 1), 3);
            pos += 1 + level + 1 + (s.content || '').length + 1;
        } else if (s.type === 'paragraph') {
            pos += (s.content || '').length + 1;
        } else if (s.type === 'table' && s.markdown) {
            pos += 1 + (s.markdown || '').length + 1;
        } else if (s.content) {
            pos += (s.content || '').length + 1;
        }
    }
    return offsets;
}

function getCharOffsetInSection(sectionEl, node, offset) {
    var treeWalker = document.createTreeWalker(sectionEl, NodeFilter.SHOW_TEXT, null, false);
    var charOffset = 0;
    var current;
    while ((current = treeWalker.nextNode())) {
        if (current === node) {
            return charOffset + offset;
        }
        charOffset += current.textContent.length;
    }
    return charOffset + offset;
}

function findSectionIndex(node) {
    var el = node.nodeType === Node.TEXT_NODE ? node.parentElement : node;
    while (el) {
        if (el.hasAttribute && el.hasAttribute('data-section-index')) {
            return parseInt(el.getAttribute('data-section-index'), 10);
        }
        el = el.parentElement;
    }
    return -1;
}

function getCharOffsetInPlainPreview(previewEl, node, offset, sectionOffsets) {
    var sectionIdx = findSectionIndex(node);
    if (sectionIdx < 0 || !sectionOffsets || sectionIdx >= sectionOffsets.length) {
        var tw = document.createTreeWalker(previewEl, NodeFilter.SHOW_TEXT, null, false);
        var off = 0;
        var cur;
        while ((cur = tw.nextNode())) {
            if (cur === node) return off + offset;
            off += cur.textContent.length;
        }
        return off + offset;
    }
    var sectionEl = previewEl.querySelector('[data-section-index="' + sectionIdx + '"]');
    if (!sectionEl) return sectionOffsets[sectionIdx] + offset;
    var intraOffset = getCharOffsetInSection(sectionEl, node, offset);
    return sectionOffsets[sectionIdx] + intraOffset;
}

function selectAllForCitation() {
    const currentDoc = docSelectorState.documents.find(d => d.doc_id === docSelectorState.selectedDocId);
    const currentKb = docSelectorState.knowledgeBases.find(k => k.kb_id === docSelectorState.selectedKbId);
    if (!currentDoc || !currentKb) return;

    const plainText = docSelectorState.parsedData ? (docSelectorState.parsedData.plain_text_preview || '') : '';
    const citation = {
        kb_id: currentKb.kb_id,
        kb_name: currentKb.name,
        doc_id: currentDoc.doc_id,
        doc_name: currentDoc.filename,
        char_start: 0,
        char_end: plainText.length,
        full_text: plainText
    };
    state.pendingCitations.push(citation);
    renderCitationTags(state.pendingCitations, true);
    closeKbDocSelector();
}

function confirmDocumentSelection() {
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();

    if (!selectedText) {
        alert('请先在预览区域选择要引用的文本');
        return;
    }

    const currentDoc = docSelectorState.documents.find(d => d.doc_id === docSelectorState.selectedDocId);
    const currentKb = docSelectorState.knowledgeBases.find(k => k.kb_id === docSelectorState.selectedKbId);

    if (!currentDoc || !currentKb) return;

    const plainText = docSelectorState.parsedData ? (docSelectorState.parsedData.plain_text_preview || '') : '';
    let charStart = 0;
    let charEnd = plainText.length;

    try {
        const range = selection.getRangeAt(0);
        const previewEl = document.getElementById('kb-doc-preview');
        const offsets = docSelectorState._sectionOffsets || [];
        if (previewEl && previewEl.contains(range.startContainer) && previewEl.contains(range.endContainer)) {
            charStart = getCharOffsetInPlainPreview(previewEl, range.startContainer, range.startOffset, offsets);
            charEnd = getCharOffsetInPlainPreview(previewEl, range.endContainer, range.endOffset, offsets);
            if (charEnd < charStart) { var t = charEnd; charEnd = charStart; charStart = t; }
        } else {
            var idx = plainText.indexOf(selectedText);
            if (idx !== -1) { charStart = idx; charEnd = idx + selectedText.length; }
        }
    } catch(e) {
        var idx2 = plainText.indexOf(selectedText);
        if (idx2 !== -1) { charStart = idx2; charEnd = idx2 + selectedText.length; }
    }

    const citation = {
        kb_id: currentKb.kb_id,
        kb_name: currentKb.name,
        doc_id: currentDoc.doc_id,
        doc_name: currentDoc.filename,
        char_start: charStart,
        char_end: charEnd,
        full_text: selectedText
    };

    state.pendingCitations.push(citation);
    renderCitationTags(state.pendingCitations, true);

    closeKbDocSelector();
}

document.addEventListener('mouseup', handleTextSelection);
document.addEventListener('keyup', handleTextSelection);

function handleTextSelection() {
    const selector = document.getElementById('kb-doc-selector');
    if (!selector || selector.style.display === 'none') return;

    const selection = window.getSelection();
    const text = selection.toString().trim();

    if (selection.rangeCount > 0) {
        const range = selection.getRangeAt(0);
        const previewEl = document.getElementById('kb-doc-preview');
        if (previewEl && previewEl.contains(range.commonAncestorContainer)) {
            updateSelectedTextLength(text.length);
            return;
        }
    }
}
