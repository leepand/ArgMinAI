// 模态框控制
function closeModal(modalId) {
    document.getElementById(modalId).classList.add('hidden');
}

function showVersionDetails(versionId) {
    fetch(`/api/features/versions/${versionId}`)
        .then(response => response.json())
        .then(data => {
            // 格式化JSON数据
            document.getElementById('sample-data-view').textContent = 
                JSON.stringify(data.sample_data, null, 2);
            document.getElementById('schema-definition-view').textContent = 
                JSON.stringify(data.schema_definition, null, 2);
            
            // 高亮JSON
            hljs.highlightElement(document.getElementById('sample-data-view'));
            hljs.highlightElement(document.getElementById('schema-definition-view'));
            
            // 显示模态框
            document.getElementById('version-details-modal').classList.remove('hidden');
        });
}

// 特征搜索
function searchFeatures() {
    const searchQuery = document.getElementById('feature-search').value.toLowerCase();
    const rows = document.querySelectorAll('#features-table tbody tr');
    
    rows.forEach(row => {
        const name = row.querySelector('td:first-child').textContent.toLowerCase();
        const description = row.querySelector('td:nth-child(2)').textContent.toLowerCase();
        
        if (name.includes(searchQuery) || description.includes(searchQuery)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// 初始化表格排序
function initFeatureTableSort() {
    const table = document.getElementById('features-table');
    const headers = table.querySelectorAll('th[data-sortable]');
    
    headers.forEach(header => {
        header.addEventListener('click', () => {
            const column = header.getAttribute('data-column');
            const direction = header.getAttribute('data-direction') || 'asc';
            const newDirection = direction === 'asc' ? 'desc' : 'asc';
            
            // 更新所有表头
            headers.forEach(h => {
                h.removeAttribute('data-direction');
                h.innerHTML = h.innerHTML.replace(/ ?↑|↓/, '');
            });
            
            // 设置当前表头
            header.setAttribute('data-direction', newDirection);
            header.innerHTML += direction === 'asc' ? ' ↑' : ' ↓';
            
            // 排序表格
            sortTable(table, column, newDirection);
        });
    });
}

function sortTable(table, column, direction) {
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    
    rows.sort((a, b) => {
        const aValue = a.querySelector(`td:nth-child(${column})`).textContent;
        const bValue = b.querySelector(`td:nth-child(${column})`).textContent;
        
        if (direction === 'asc') {
            return aValue.localeCompare(bValue);
        } else {
            return bValue.localeCompare(aValue);
        }
    });
    
    // 重新插入排序后的行
    rows.forEach(row => tbody.appendChild(row));
}

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initFeatureTableSort();
    
    // 绑定搜索事件
    const searchInput = document.getElementById('feature-search');
    if (searchInput) {
        searchInput.addEventListener('input', searchFeatures);
    }
    
    // 绑定版本详情按钮
    document.querySelectorAll('.version-details-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const versionId = this.getAttribute('data-version-id');
            showVersionDetails(versionId);
        });
    });
});