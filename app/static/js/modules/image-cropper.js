// modules/image-cropper.js — 图片上传 + Canvas裁剪器

const uploadTrigger = document.getElementById('upload-trigger');
const uploadMenu = document.getElementById('upload-menu');
const uploadDropdown = document.getElementById('upload-dropdown');

function toggleUploadMenu(event) {
    event.stopPropagation();
    const isShowing = uploadMenu.classList.contains('show');
    if (isShowing) {
        closeUploadMenu();
    } else {
        uploadMenu.style.display = 'block';
        void uploadMenu.offsetWidth;
        uploadMenu.classList.add('show');
    }
}

function closeUploadMenu() {
    uploadMenu.classList.remove('show');
    setTimeout(() => {
        if (!uploadMenu.classList.contains('show')) {
            uploadMenu.style.display = 'none';
        }
    }, 200);
}

function triggerImageUpload() {
    closeUploadMenu();
    document.getElementById('file-input').click();
}

if (uploadTrigger) {
    uploadTrigger.addEventListener('click', toggleUploadMenu);
}

document.addEventListener('click', (e) => {
    if (uploadDropdown && !uploadDropdown.contains(e.target)) {
        closeUploadMenu();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeUploadMenu();
    }
});

// ==========================================
// 图片上传与预览
// ==========================================
dom.fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    for (let f of files) {
        if (!f.type.startsWith('image/')) continue;
        const reader = new FileReader();
        reader.onload = (ev) => {
            const b64 = ev.target.result; state.selectedImages.push(b64);
            const wrap = document.createElement('div'); wrap.className = 'preview-img';
            wrap.innerHTML = `<img src="${b64}"><button class="remove-btn">x</button>`;
            wrap.querySelector('button').onclick = () => { wrap.remove(); state.selectedImages = state.selectedImages.filter(i => i !== b64); };
            dom.previewArea.appendChild(wrap);
        };
        reader.readAsDataURL(f);
    }
    dom.fileInput.value = "";
});

// ==========================================
// 纯手工 Vanilla JS 图片裁剪引擎 (二段式状态机)
// ==========================================
let cropper = {
    modal: document.getElementById('cropper-modal'),
    canvas: document.getElementById('cropper-canvas'),
    ctx: document.getElementById('cropper-canvas').getContext('2d'),
    box: document.getElementById('cropper-box'),
    fileInput: document.getElementById('cropper-file-input'),
    img: new Image(),
    targetType: null, // 'user' | 'role' | 'create_role'
    roleId: null,

    // 两段式缓存
    step: 1,
    cachedCircleBase64: null,
    cachedBgBase64: null,

    // 拖拽与缩放状态
    isDragging: false,
    startX: 0, startY: 0,
    imgX: 0, imgY: 0, imgScale: 1
};

function openCropper(targetType, roleId = null) {
    cropper.targetType = targetType;
    cropper.roleId = roleId;
    cropper.step = 1;

    // 第一步强制锁定为 1:1 圆形
    cropper.box.style.width = '240px';
    cropper.box.style.height = '240px';
    cropper.box.style.borderRadius = '50%';

    const btn = document.getElementById('cropper-confirm-btn');
    if (targetType === 'create_role') {
        btn.innerText = "确认裁剪";
    } else {
        btn.innerText = "下一步: 截取横向卡片背景 (1/2)";
    }

    cropper.ctx.clearRect(0, 0, cropper.canvas.width, cropper.canvas.height);
    cropper.modal.style.display = 'flex';
    cropper.fileInput.click();
}

function closeCropper() { cropper.modal.style.display = 'none'; cropper.fileInput.value = ''; }

cropper.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
        cropper.img.onload = () => { initCropperCanvas(); };
        cropper.img.src = ev.target.result;
    };
    reader.readAsDataURL(file);
});

function initCropperCanvas() {
    const body = document.querySelector('.cropper-body');
    const cw = body.clientWidth, ch = body.clientHeight;
    cropper.canvas.width = cw; cropper.canvas.height = ch;

    const scaleX = cw / cropper.img.width;
    const scaleY = ch / cropper.img.height;
    cropper.imgScale = Math.max(scaleX, scaleY) * 1.1;

    cropper.imgX = (cw - cropper.img.width * cropper.imgScale) / 2;
    cropper.imgY = (ch - cropper.img.height * cropper.imgScale) / 2;

    drawCropper();
}

function drawCropper() {
    cropper.ctx.clearRect(0, 0, cropper.canvas.width, cropper.canvas.height);
    cropper.ctx.drawImage(cropper.img, cropper.imgX, cropper.imgY, cropper.img.width * cropper.imgScale, cropper.img.height * cropper.imgScale);
}

const cropperBody = document.querySelector('.cropper-body');
cropperBody.addEventListener('mousedown', dragStart);
cropperBody.addEventListener('mousemove', dragMove);
window.addEventListener('mouseup', dragEnd);

cropperBody.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = cropper.canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const zoomFactor = 1.1;
    const direction = e.deltaY < 0 ? 1 : -1;
    let newScale = direction > 0 ? cropper.imgScale * zoomFactor : cropper.imgScale / zoomFactor;

    const minScaleX = cropper.box.offsetWidth / cropper.img.width;
    const minScaleY = cropper.box.offsetHeight / cropper.img.height;
    const minScale = Math.max(minScaleX, minScaleY);
    const maxScale = 5.0;

    if (newScale < minScale) newScale = minScale;
    if (newScale > maxScale) newScale = maxScale;

    cropper.imgX = mouseX - (mouseX - cropper.imgX) * (newScale / cropper.imgScale);
    cropper.imgY = mouseY - (mouseY - cropper.imgY) * (newScale / cropper.imgScale);
    cropper.imgScale = newScale;

    drawCropper();
}, { passive: false });

function dragStart(e) { cropper.isDragging = true; cropper.startX = e.clientX; cropper.startY = e.clientY; }
function dragMove(e) {
    if (!cropper.isDragging) return;
    const dx = e.clientX - cropper.startX; const dy = e.clientY - cropper.startY;
    cropper.imgX += dx; cropper.imgY += dy;
    cropper.startX = e.clientX; cropper.startY = e.clientY;
    drawCropper();
}
function dragEnd() { cropper.isDragging = false; }

// 执行二段式裁剪上传引擎
async function confirmCrop() {
    if (!cropper.img.src) return alert("请先选择图片");

    const boxRect = cropper.box.getBoundingClientRect();
    const canvasRect = cropper.canvas.getBoundingClientRect();
    const cropX = boxRect.left - canvasRect.left;
    const cropY = boxRect.top - canvasRect.top;
    const cropW = boxRect.width;
    const cropH = boxRect.height;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = cropW; offCanvas.height = cropH;
    const offCtx = offCanvas.getContext('2d');
    offCtx.drawImage(cropper.canvas, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
    const base64Data = offCanvas.toDataURL('image/png');

    const btn = document.getElementById('cropper-confirm-btn');

    if (cropper.targetType === 'create_role') {
        window._pendingCreateAvatar = base64Data;
        document.getElementById('create-role-avatar-preview').innerHTML = `<img src="${base64Data}" style="border-radius:50%; width:100%; height:100%; object-fit:cover;">`;
        closeCropper();
        return;
    }

    if (cropper.step === 1) {
        cropper.cachedCircleBase64 = base64Data;
        cropper.step = 2;

        // 瞬间切换到横版卡片裁剪状态
        cropper.box.style.width = '400px';
        cropper.box.style.height = '100px';
        cropper.box.style.borderRadius = '8px';
        btn.innerText = "确认并上传 (2/2)";
        return;
    }

    if (cropper.step === 2) {
        cropper.cachedBgBase64 = base64Data;
        btn.innerText = "上传中..."; btn.disabled = true;

        try {
            const payload = {
                target_type: cropper.targetType,
                role_id: cropper.roleId,
                image_circle_base64: cropper.cachedCircleBase64,
                image_bg_base64: cropper.cachedBgBase64
            };
            const res = await fetch('/api/upload_avatar', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });

            if (res.ok) {
                closeCropper();
                const jsonRes = await res.json();
                const paths = jsonRes.paths;

                const previewEl = document.getElementById(cropper.targetType === 'user' ? 'set-user-avatar-preview' : 'set-role-avatar-preview');
                const currentMode = document.getElementById(cropper.targetType === 'user' ? 'set-user-avatar-mode' : 'set-role-avatar-mode').value;

                if (cropper.targetType === 'user') {
                    state.userProfile.avatar_circle = paths.avatar_circle;
                    state.userProfile.avatar_bg = paths.avatar_bg;
                    renderUserSidebar(); // 【直接重绘前端，绝不拉取旧数据】
                }
                if (cropper.targetType === 'role') {
                    state.currentRoleMeta.avatar_circle = paths.avatar_circle;
                    state.currentRoleMeta.avatar_bg = paths.avatar_bg;

                    const rIndex = state.roles.findIndex(r => r.role_id === cropper.roleId);
                    if (rIndex > -1) {
                        state.roles[rIndex].avatar_circle = paths.avatar_circle;
                        state.roles[rIndex].avatar_bg = paths.avatar_bg;
                        renderRoleList(); // 【直接重绘前端，绝不拉取旧数据】
                    }
                }

                // 强制实体预览 DOM 刷新
                if (previewEl) {
                    const meta = cropper.targetType === 'user' ? state.userProfile : state.currentRoleMeta;
                    previewEl.innerHTML = renderPreviewDOM(currentMode, meta.avatar_circle, meta.avatar_bg, meta.display_name);
                }

            } else { alert("上传失败"); }
        } catch(e) { console.error(e); }
        finally { btn.disabled = false; }
    }
}
